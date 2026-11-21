"""Pytest configuration: path setup for MRTO tests."""
import os
import sys

# Add backend directory to path so `from mrto import ...` works
BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BACKEND)

# Use temporary SQLite DB for tests
import tempfile
TEST_DB_DIR = tempfile.mkdtemp(prefix="mrto_test_")
os.environ["MRTO_DB_PATH"] = os.path.join(TEST_DB_DIR, "test.sqlite3")
os.environ["MRTO_MODELS_DIR"] = os.path.join(os.path.dirname(__file__), "..", "models")
