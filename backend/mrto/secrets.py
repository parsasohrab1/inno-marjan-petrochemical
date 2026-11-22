"""Secret management — single source of truth for runtime secrets.

Provides a `get_secret(name, default=None)` resolver with a pluggable
backend chain, and `require_secret(name)` that raises a clear error if a
required secret is missing. The backend order is:

  1. Environment variable (always first — fastest, least privileged).
  2. File on disk pointed to by `<NAME>_FILE` (Docker/K8s secret mount).
  3. HashiCorp Vault via the `hvac` library when MRTO_VAULT_ADDR is set
     and the library is installed. Falls back silently otherwise.
  4. SOPS-encrypted file when MRTO_SOPS_FILE is set and `sops` is on PATH.

The chain is evaluated once per secret on first access and cached; tests
can reset the cache via `reset_cache()`. All secret values are read
exactly once and never logged; on failure we log only the *name*, never
the attempted value.

Design goals:
- Zero new dependencies required for the default path (env + file). The
  Vault/SOPS backends are optional and degrade to a warning if their
  libs/CLIs are missing.
- A single call site replaces scattered `os.environ.get(...)` reads,
  so the audit story is "all secrets flow through this module".
- The `MRTO_ALLOW_INSECURE_JWT` dev bypass is *not* applied here — it
  stays in app.py because it is a policy decision about JWT length, not
  a secret-resolution concern.
"""
from __future__ import annotations

import os
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

log = logging.getLogger("mrto.secrets")

_CACHE: dict[str, str] = {}


def reset_cache() -> None:
    """Clear the secret cache. Mainly for tests."""
    _CACHE.clear()


# ────────────────────────────────────────────────────────────────────
# Backend: environment variable
# ────────────────────────────────────────────────────────────────────
def _from_env(name: str) -> Optional[str]:
    return os.environ.get(name)


# ────────────────────────────────────────────────────────────────────
# Backend: file on disk (Docker/K8s secret mount convention)
# ────────────────────────────────────────────────────────────────────
def _from_file(name: str) -> Optional[str]:
    """Read the path in `<NAME>_FILE` and return its trimmed contents.

    This is the standard convention for mounting secrets as files in
    Docker Swarm / Kubernetes: instead of putting the secret value in
    an env var, the operator mounts a file and points the app at it via
    `<NAME>_FILE=/mnt/secrets/jwt_secret`. The file is read on first
    access and the value cached in-memory.
    """
    file_path = os.environ.get(f"{name}_FILE")
    if not file_path:
        return None
    p = Path(file_path)
    if not p.is_file():
        log.warning("secret file %s pointed to by %s_FILE does not exist",
                    file_path, name)
        return None
    try:
        return p.read_text(encoding="utf-8").strip()
    except OSError as e:
        log.warning("failed to read secret file %s: %s", file_path, e)
        return None


# ────────────────────────────────────────────────────────────────────
# Backend: HashiCorp Vault (optional — requires `hvac`)
# ────────────────────────────────────────────────────────────────────
_VAULT_CLIENT = None


def _vault_client():
    """Return a cached Vault client or None if not configured/available."""
    global _VAULT_CLIENT
    if _VAULT_CLIENT is not None:
        return _VAULT_CLIENT
    addr = os.environ.get("MRTO_VAULT_ADDR")
    if not addr:
        return None
    try:
        import hvac  # type: ignore
    except ImportError:
        log.warning("MRTO_VAULT_ADDR is set but `hvac` is not installed; "
                    "Vault backend disabled. pip install hvac to enable.")
        return None
    token = (os.environ.get("MRTO_VAULT_TOKEN")
             or _from_file("MRTO_VAULT_TOKEN"))
    try:
        c = hvac.Client(url=addr, token=token)
        if not c.is_authenticated():
            log.warning("Vault at %s did not authenticate the provided token",
                        addr)
            return None
        _VAULT_CLIENT = c
        return c
    except Exception as e:  # pragma: no cover — network/credentials errors
        log.warning("Vault client init failed for %s: %s", addr, e)
        return None


def _from_vault(name: str) -> Optional[str]:
    """Read a secret from Vault. The path is MRTO_VAULT_PATH (default
    `secret/mrto`) and the key is the lowercased env name."""
    c = _vault_client()
    if c is None:
        return None
    mount = os.environ.get("MRTO_VAULT_MOUNT", "secret")
    path = os.environ.get("MRTO_VAULT_PATH", "mrto")
    try:
        # kv v2 read path: <mount>/data/<path>
        resp = c.secrets.kv.v2.read_secret_version(
            path=path, mount_point=mount
        )
        data = resp["data"]["data"]
        key = name.lower().removeprefix("mrto_")
        return data.get(key) or data.get(name)
    except Exception as e:  # pragma: no cover
        log.warning("Vault read failed for %s/%s: %s", mount, path, e)
        return None


# ────────────────────────────────────────────────────────────────────
# Backend: SOPS-encrypted file (optional — requires `sops` on PATH)
# ────────────────────────────────────────────────────────────────────
def _from_sops(name: str) -> Optional[str]:
    """Decrypt the SOPS file at MRTO_SOPS_FILE and look up the key.

    SOPS keeps the entire secrets file encrypted at rest; we shell out
    to `sops -d` which handles AWS KMS / GCP KMS / age keys based on the
    file's metadata. The decrypted output is parsed as YAML/JSON and
    the value for `name` (case-insensitive) is returned.
    """
    sops_file = os.environ.get("MRTO_SOPS_FILE")
    if not sops_file or not shutil.which("sops"):
        return None
    try:
        out = subprocess.run(
            ["sops", "-d", sops_file],
            capture_output=True, text=True, check=True, timeout=10,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        log.warning("sops decrypt of %s failed: %s", sops_file, e)
        return None
    # Parse — support both .yaml and .json.
    text = out.stdout
    import json
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
            data = yaml.safe_load(text)
        except Exception:
            return None
    if not isinstance(data, dict):
        return None
    return data.get(name) or data.get(name.lower())


# ────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────
def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """Resolve a secret from the configured backend chain.

    Order: env → file → Vault → SOPS. First hit wins; the result is
    cached for the process lifetime (use reset_cache() in tests).
    """
    if name in _CACHE:
        return _CACHE[name]
    for backend in (_from_env, _from_file, _from_vault, _from_sops):
        try:
            val = backend(name)
        except Exception as e:  # defensive — never let a backend crash boot
            log.warning("secret backend %s raised for %s: %s",
                        backend.__name__, name, e)
            continue
        if val is not None and val != "":
            _CACHE[name] = val
            return val
    if default is not None:
        _CACHE[name] = default
        return default
    return None


def require_secret(name: str) -> str:
    """Like get_secret but raises RuntimeError if the value is missing
    or empty. Use for secrets the app cannot start without."""
    v = get_secret(name)
    if not v:
        raise RuntimeError(
            f"Required secret {name!r} is not set. Provide it via the {name} "
            f"env var, a {name}_FILE mount, Vault, or SOPS. See "
            f"docs/DEPLOYMENT.md for the secrets-management guide."
        )
    return v
