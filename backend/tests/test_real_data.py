"""Tests for the real-data fetcher (UCI datasets)."""
import os
import pytest
import pandas as pd

from mrto import real_data as rd


def test_datasets_have_required_keys():
    """The DATASETS registry must list all expected datasets."""
    assert "airfoil_self_noise" in rd.DATASETS
    assert "ccpp" in rd.DATASETS
    assert "gas_turbbine_nox" in rd.DATASETS
    for k, v in rd.DATASETS.items():
        assert "url" in v and "desc" in v


def test_fetch_airfoil_returns_dataframe_or_none(tmp_path):
    """If network unavailable, function returns None gracefully."""
    df = rd.fetch_airfoil(str(tmp_path))
    if df is None:
        pytest.skip("Network unavailable — airfoil fetch returned None")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 100
    # Schema check
    expected = {"intensity", "avg_tube_t_c", "tube_stress"}
    assert expected.issubset(df.columns)


def test_fetch_all_returns_dict(tmp_path):
    paths = rd.fetch_all(str(tmp_path))
    assert isinstance(paths, dict)
    # If network failed, dict is empty — that's allowed
    for k, p in paths.items():
        assert os.path.exists(p)
        assert os.path.getsize(p) > 0


def test_fetch_handles_missing_dir(tmp_path):
    """Function should create directory if missing."""
    new_dir = str(tmp_path / "newsubdir")
    rd.fetch_all(new_dir)
    assert os.path.isdir(new_dir)
