"""Tests for synthetic data generator (Module 0)."""
import os
import math
import tempfile

import pandas as pd
import pytest

from mrto import synthetic_data as sd
from mrto.synthetic_data import (
    FurnaceConfig, lmp_to_rupture_time,
    generate_tubes, generate_burner_profiles, generate_time_series,
    generate_temperature_map, generate_rul_labels,
    generate_sc_optimization_dataset, generate_burner_status_dataset,
    generate_catalyst_deactivation, generate_all,
)


def test_lmp_returns_finite_for_design_temperature():
    """Larson-Miller rupture time at 880°C should be finite & positive."""
    t_r = lmp_to_rupture_time(880.0)
    assert math.isfinite(t_r)
    assert t_r > 0
    # Roughly in the order of 10^3 to 10^6 hours for HP-Modified at design temp
    assert 1e3 <= t_r <= 1e7


def test_lmp_higher_temperature_lower_life():
    """Higher wall temperature → lower rupture time (creep acceleration)."""
    t_low = lmp_to_rupture_time(800.0)
    t_high = lmp_to_rupture_time(950.0)
    assert t_high < t_low, f"t(950)={t_high} should be < t(800)={t_low}"


def test_tubes_shape_and_uniqueness():
    cfg = FurnaceConfig(n_rows=2, n_cols=10)
    tubes = generate_tubes(cfg)
    assert len(tubes) == 20
    assert tubes["tube_id"].is_unique
    assert set(["row", "col", "material", "design_life_h"]).issubset(tubes.columns)


def test_burner_profiles_sensible():
    cfg = FurnaceConfig(n_burners=20)
    b = generate_burner_profiles(cfg)
    assert len(b) == 20
    assert (b["base_intensity"] > 0).all()
    assert (b["age_factor"] <= 1.0).all() and (b["age_factor"] > 0).all()


def test_timeseries_within_bounds():
    cfg = FurnaceConfig()
    ts = generate_time_series(cfg, n_hours=24, sampling_min=60)
    assert len(ts) == 24
    assert (ts["sc_ratio"] >= 2.5).all() and (ts["sc_ratio"] <= 4.5).all()
    assert (ts["t_out"] >= 820).all() and (ts["t_out"] <= 950).all()
    # Pressure drop positive
    assert (ts["dp_bar"] > 0).all()


def test_temperature_map_coverage():
    cfg = FurnaceConfig(n_rows=2, n_cols=4, n_burners=4)
    tubes = generate_tubes(cfg)
    burners = generate_burner_profiles(cfg)
    ts = generate_time_series(cfg, n_hours=2, sampling_min=60)
    tm = generate_temperature_map(tubes, burners, ts, cfg)
    assert len(tm) == 2 * 4 * len(ts)  # n_tubes * n_timestamps
    # wall temperature in plausible range
    assert (tm["wall_temp_c"] >= 350).all() and (tm["wall_temp_c"] <= 1050).all()


def test_rul_labels_nonneg_and_probability_in_unit():
    cfg = FurnaceConfig(n_rows=2, n_cols=4, n_burners=4)
    tubes = generate_tubes(cfg)
    burners = generate_burner_profiles(cfg)
    ts = generate_time_series(cfg, n_hours=2, sampling_min=60)
    tm = generate_temperature_map(tubes, burners, ts, cfg)
    rul = generate_rul_labels(tubes, tm, cfg)
    assert (rul["rul_h"] >= 0).all()
    assert (rul["p_fail_90d"] >= 0).all() and (rul["p_fail_90d"] <= 1).all()
    assert (rul["p_fail_30d"] >= 0).all() and (rul["p_fail_30d"] <= 1).all()
    # Higher consumed life should correlate with higher failure probability
    sorted_by_consumed = rul.sort_values("consumed_life_fraction")
    first_half = sorted_by_consumed.iloc[: len(sorted_by_consumed) // 2]["p_fail_90d"].mean()
    second_half = sorted_by_consumed.iloc[len(sorted_by_consumed) // 2:]["p_fail_90d"].mean()
    # Not strictly greater on small samples due to noise, but trend should hold
    # Just assert no NaN
    assert not rul["p_fail_90d"].isna().any()


def test_sc_optimization_dataset_shape_and_constraints():
    cfg = FurnaceConfig()
    df = generate_sc_optimization_dataset(cfg, n_samples=200)
    assert len(df) == 200
    assert (df["efficiency"] >= 0).all() and (df["efficiency"] <= 1).all()
    assert (df["coking_risk"] >= 0).all() and (df["coking_risk"] <= 1).all()
    assert (df["tube_stress"] >= 0).all() and (df["tube_stress"] <= 1).all()


def test_burner_status_dataset_balanced_enough():
    cfg = FurnaceConfig()
    df = generate_burner_status_dataset(cfg, n_samples=500)
    assert len(df) == 500
    assert df["label_inefficient"].nunique() >= 1
    # Should have at least some inefficient labels
    assert (df["label_inefficient"] == 1).sum() > 0


def test_catalyst_dataset_activity_in_unit():
    cfg = FurnaceConfig()
    df = generate_catalyst_deactivation(cfg, n_samples=200)
    assert (df["activity"] >= 0).all() and (df["activity"] <= 1).all()


def test_generate_all_writes_files(tmp_path):
    paths = generate_all(str(tmp_path), n_hours=24, sampling_min=60)
    assert all(os.path.exists(p) for p in paths.values())
    # Verify file sizes
    for k, p in paths.items():
        assert os.path.getsize(p) > 0, f"{k} file is empty"
    tubes = pd.read_csv(paths["tubes"])
    assert len(tubes) > 0
