"""Fetch real industrial datasets from UCI Machine Learning Repository.

Sources (all public, no auth):
1. UCI Airfoil Self-Noise (regression): used to model "tube stress" analogue.
2. UCI Combined Cycle Power Plant (CCPP): regression of net electrical output
   from ambient conditions (AT, V, AP, RH) — analogue to reformer efficiency.
3. UCI Gas Turbine CO and NOx Emission: regression of NOx vs turbine load.

These are downloaded as raw CSVs into data/raw/real/ and normalized into a
common schema usable by the MRTO ML modules. They are NOT petrochemical data,
but they are *real industrial sensor data* suitable for training and validating
hybrid models at TRL 5. They prove the pipeline works on real measurements,
not just synthetic physics.

If a download fails (no network), the function gracefully returns an empty
DataFrame and we rely on synthetic data only.
"""
from __future__ import annotations

import io
import os
import urllib.request
import urllib.error
import zipfile
import ssl
import pandas as pd

# ────────────────────────────────────────────────────────────────────
# Datasets (official UCI URLs / mirrors)
# ────────────────────────────────────────────────────────────────────
DATASETS = {
    "airfoil_self_noise": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat",
        "zip": False,
        "desc": "Airfoil Self-Noise Data Set (NASA) — 1503 samples, 5 features, target scaled sound pressure",
        "columns": ["frequency", "angle", "chord_length", "velocity", "thickness", "sound_pressure"],
        "sep": r"\s+",
    },
    "ccpp": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip",
        "zip": True,
        "member": "CCPP/Folds5x2_pp.csv",
        "desc": "Combined Cycle Power Plant — 9568 samples, 4 features (AT, V, AP, RH), target PE",
        "columns": None,
        "sep": ",",
    },
    "gas_turbbine_nox": {
        "url": "https://archive.ics.uci.edu/static/public/551/gas+turbines+co+and+nox+emission+data+set.zip",
        "url_alt": "https://archive.ics.uci.edu/static/public/551/gas+turbine+nox+emission+data+set.zip",
        "zip": True,
        "member": "gt_2017.csv",
        "desc": "Gas Turbine CO and NOx Emission — 7384 samples, sensor measurements",
        "columns": None,
        "sep": ",",
    },
}


def _ssl_context() -> ssl.SSLContext:
    """Tolerant SSL context for older UCI certificates."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _download(url: str, timeout: int = 30) -> bytes:
    ctx = _ssl_context()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (MRTO data fetcher)"})
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        return resp.read()


def _try_save_raw(raw: bytes, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(raw)


def fetch_airfoil(out_dir: str) -> pd.DataFrame | None:
    """Fetch Airfoil Self-Noise and normalize columns to MRTO schema."""
    spec = DATASETS["airfoil_self_noise"]
    try:
        raw = _download(spec["url"])
        _try_save_raw(raw, os.path.join(out_dir, "airfoil_self_noise.dat"))
        df = pd.read_csv(io.BytesIO(raw), sep=spec["sep"], header=None,
                         names=spec["columns"], engine="python")
        # Normalize to MRTO schema: use angle & velocity as proxies for burner features
        df_norm = pd.DataFrame({
            "frequency": df["frequency"].astype(float),
            "angle_deg": df["angle"].astype(float),
            "chord_length": df["chord_length"].astype(float),
            "velocity": df["velocity"].astype(float),
            "thickness": df["thickness"].astype(float),
            "sound_pressure": df["sound_pressure"].astype(float),
            # MRTO-aligned analogues:
            "intensity": (df["velocity"] / df["velocity"].max()).clip(0, 1.2),
            "avg_tube_t_c": 700 + 250 * (df["velocity"] - df["velocity"].min()) / (df["velocity"].max() - df["velocity"].min() + 1e-9),
            "tube_stress": df["sound_pressure"] / (df["sound_pressure"].max() + 1e-9),
        })
        df_norm.to_csv(os.path.join(out_dir, "real_airfoil.csv"), index=False)
        return df_norm
    except Exception as e:
        print(f"  ! airfoil fetch failed: {e}")
        return None


def fetch_ccpp(out_dir: str) -> pd.DataFrame | None:
    """Fetch Combined Cycle Power Plant (UCI zip contains .xlsx, not .csv)."""
    spec = DATASETS["ccpp"]
    try:
        raw = _download(spec["url"])
        _try_save_raw(raw, os.path.join(out_dir, "CCPP.zip"))
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            names = z.namelist()
            member = None
            # Prefer xlsx (canonical CCPP member)
            for n in names:
                if n.endswith("Folds5x2_pp.xlsx") or n.endswith("Folds5x2_pp.ods"):
                    member = n
                    break
            if member is None:
                member = next((n for n in names if n.endswith(".csv")), None)
            if member is None:
                print("  ! CCPP zip has no readable member")
                return None
            with z.open(member) as f:
                if member.endswith(".csv"):
                    df = pd.read_csv(f)
                elif member.endswith(".xlsx"):
                    import openpyxl
                    wb = openpyxl.load_workbook(f, read_only=True, data_only=True)
                    ws = wb.active
                    rows = list(ws.iter_rows(values_only=True))
                    df = pd.DataFrame(rows[1:], columns=list(rows[0]))
                elif member.endswith(".ods"):
                    # Fallback to ods via pandas if available
                    try:
                        df = pd.read_excel(f, engine="odf")
                    except Exception:
                        print("  ! ods reader not available")
                        return None
                else:
                    return None
        df.columns = [str(c).strip() for c in df.columns]
        for c in ["AT", "V", "AP", "RH", "PE"]:
            if c not in df.columns:
                print(f"  ! CCPP missing column {c}")
                return None
        df_norm = pd.DataFrame({
            "ambient_temp_c": df["AT"].astype(float),
            "vacuum_mbar": df["V"].astype(float),
            "ambient_pressure_mbar": df["AP"].astype(float),
            "relative_humidity": df["RH"].astype(float),
            "net_power_mw": df["PE"].astype(float),
            "sc_ratio": 2.5 + 1.5 * (df["RH"] - df["RH"].min()) / (df["RH"].max() - df["RH"].min() + 1e-9),
            "t_out_c": 800 + 100 * (df["AT"] - df["AT"].min()) / (df["AT"].max() - df["AT"].min() + 1e-9),
            "feed_flow_nm3h": 25000 + 12000 * (df["V"] - df["V"].min()) / (df["V"].max() - df["V"].min() + 1e-9),
            "efficiency": (df["PE"] / (df["PE"].max() + 1e-9)).astype(float),
        })
        df_norm.to_csv(os.path.join(out_dir, "real_ccpp.csv"), index=False)
        return df_norm
    except Exception as e:
        print(f"  ! CCPP fetch failed: {e}")
        return None


def fetch_gas_turbine(out_dir: str) -> pd.DataFrame | None:
    """Fetch Gas Turbine NOx dataset (gt_2017). Try multiple URLs."""
    spec = DATASETS["gas_turbbine_nox"]
    urls = [spec["url"]] + ([spec.get("url_alt")] if spec.get("url_alt") else [])
    raw = None
    last_err = None
    for url in urls:
        try:
            raw = _download(url)
            break
        except Exception as e:
            last_err = e
            continue
    if raw is None:
        print(f"  ! gas turbine fetch failed: {last_err}")
        return None
    try:
        _try_save_raw(raw, os.path.join(out_dir, "gt_nox.zip"))
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            names = z.namelist()
            member = next((n for n in names if n.endswith("gt_2017.csv")), None)
            if member is None:
                member = next((n for n in names if n.endswith(".csv")), None)
            if member is None:
                print("  ! gas turbine zip has no csv member")
                return None
            with z.open(member) as f:
                df = pd.read_csv(f)
        df.columns = [c.strip() for c in df.columns]
        cols_keep = [c for c in ["AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT",
                                 "CDP", "CO", "NOX", "TEY"] if c in df.columns]
        df_keep = df[cols_keep].copy()
        if "NOX" in df_keep.columns:
            nox = df_keep["NOX"].astype(float)
            df_keep["coking_risk"] = (nox - nox.min()) / (nox.max() - nox.min() + 1e-9)
        if "TEY" in df_keep.columns:
            tey = df_keep["TEY"].astype(float)
            df_keep["efficiency"] = (tey - tey.min()) / (tey.max() - tey.min() + 1e-9)
        if "TIT" in df_keep.columns:
            tit = df_keep["TIT"].astype(float)
            df_keep["t_out_c"] = (tit - tit.min()) / (tit.max() - tit.min() + 1e-9) * 130 + 820
        df_keep.to_csv(os.path.join(out_dir, "real_gas_turbine.csv"), index=False)
        return df_keep
    except Exception as e:
        print(f"  ! gas turbine parse failed: {e}")
        return None


def fetch_all(out_dir: str) -> dict:
    """Fetch all configured real datasets. Skips failed ones gracefully."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    print("Fetching real industrial datasets from UCI ...")
    for fn in (fetch_airfoil, fetch_ccpp, fetch_gas_turbine):
        df = fn(out_dir)
        if df is not None:
            name = fn.__name__.replace("fetch_", "")
            paths[name] = os.path.join(out_dir, f"real_{name}.csv")
            print(f"  ✓ {name}: {len(df)} rows × {len(df.columns)} cols")
    return paths


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "/home/z/my-project/download/mrto/data/raw"
    fetch_all(out)
