"""Pytest configuration: path setup for MRTO tests."""
import os
import sys

# Add backend directory to path so `from mrto import ...` works
BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BACKEND)

# SECURITY: tests need a valid JWT secret (the app fails fast if unset).
# Use a dedicated test secret; bypass the >=32-char guard via the allow flag.
os.environ.setdefault("MRTO_JWT_SECRET", "test-jwt-secret-for-pytest-only-do-not-use-in-prod")
os.environ.setdefault("MRTO_ALLOW_INSECURE_JWT", "1")
# Keep the admin password deterministic across test runs (otherwise the
# random password printed at first boot would break login tests).
os.environ.setdefault("MRTO_ADMIN_PASS", "admin123")
# Allow anonymous read in tests so existing GET tests don't need a token.
os.environ.setdefault("MRTO_ALLOW_ANON", "1")

# Use temporary SQLite DB for tests
import tempfile
TEST_DB_DIR = tempfile.mkdtemp(prefix="mrto_test_")
os.environ["MRTO_DB_PATH"] = os.path.join(TEST_DB_DIR, "test.sqlite3")
os.environ["MRTO_MODELS_DIR"] = os.path.join(os.path.dirname(__file__), "..", "models")
