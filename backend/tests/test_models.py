"""Tests for the four ML modules (training + inference)."""
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from mrto import synthetic_data as sd
from mrto.models import (
    TemperatureMapModel, RULModel, OptimizationModel, BurnerCatalystModel,
    save_model, load_model,
)


@pytest.fixture(scope="module")
def dataset():
    """Generate a small synthetic dataset once for all model tests."""
    cfg = sd.FurnaceConfig(n_rows=2, n_cols=10, n_burners=10)
    tubes = sd.generate_tubes(cfg)
    burners = sd.generate_burner_profiles(cfg)
    ts = sd.generate_time_series(cfg, n_hours=24, sampling_min=60)
    temp_map = sd.generate_temperature_map(tubes, burners, ts, cfg)
    rul_df = sd.generate_rul_labels(tubes, temp_map, cfg)
    sc_df = sd.generate_sc_optimization_dataset(cfg, n_samples=800)
    burner_df = sd.generate_burner_status_dataset(cfg, n_samples=600)
    cat_df = sd.generate_catalyst_deactivation(cfg, n_samples=400)
    return {
        "cfg": cfg, "tubes": tubes, "burners": burners, "ts": ts,
        "temp_map": temp_map, "rul_df": rul_df, "sc_df": sc_df,
        "burner_df": burner_df, "cat_df": cat_df,
    }


def test_module1_train_and_predict(dataset):
    m = TemperatureMapModel()
    metrics = m.fit(dataset["temp_map"], dataset["ts"], dataset["tubes"], dataset["burners"], test_size=0.3)
    assert metrics.rmse > 0
    assert metrics.n_test > 0
    # Predict a single row
    feat = pd.DataFrame([{
        "burner_intensity_avg": 1.0, "burner_intensity_min": 0.9, "burner_intensity_max": 1.1,
        "sc_ratio": 3.0, "t_out": 880.0, "feed_flow_nm3h": 32000,
        "fuel_flow_nm3h": 11500, "row_factor": 0.02, "col_norm": 0.5, "anomaly_offset": 0.0,
    }])
    pred = m.predict(feat)
    assert pred.shape == (1,)
    assert 400 < pred[0] < 1050


def test_module1_srs_accuracy_target(dataset):
    """SRS SR-T1-ACC: RMSE ≤ 15 °C."""
    m = TemperatureMapModel()
    metrics = m.fit(dataset["temp_map"], dataset["ts"], dataset["tubes"], dataset["burners"], test_size=0.3)
    # We expect < 15 °C RMSE on synthetic data (low noise by construction)
    assert metrics.rmse <= 15.0, f"RMSE {metrics.rmse:.2f} exceeds 15 °C target"


def test_module2_train_and_predict(dataset):
    m = RULModel()
    metrics = m.fit(dataset["rul_df"], dataset["tubes"], test_size=0.3)
    assert metrics.mae_hours >= 0
    assert metrics.n_test > 0
    feat = pd.DataFrame([{
        "avg_wall_temp_c": 880.0, "consumed_life_fraction": 0.5,
        "installed_at_days_ago": 200, "row": 1, "col": 1,
    }])
    rul = m.predict_rul(feat)
    prob = m.predict_failure_prob(feat)
    assert rul.shape == (1,)
    assert rul[0] >= 0
    assert prob.shape == (1,)
    assert 0 <= prob[0] <= 1


def test_module3_train_and_optimize(dataset):
    m = OptimizationModel()
    metrics = m.fit(dataset["sc_df"], test_size=0.25)
    assert metrics.eff_rmse > 0
    # Optimize from current operating point
    result = m.optimize({
        "sc_ratio": 3.3, "t_out_c": 880.0, "feed_flow_nm3h": 32000,
        "cat_age": 0.3, "fuel_flow_nm3h": 11500,
    })
    assert "sc_ratio_opt" in result
    assert "efficiency_pred" in result
    assert 2.5 <= result["sc_ratio_opt"] <= 4.5
    assert 820 <= result["t_out_c_opt"] <= 950
    # Coking risk within constraint
    assert result["coking_risk_pred"] <= 0.5


def test_module4_train_and_predict(dataset):
    m = BurnerCatalystModel()
    metrics = m.fit(dataset["burner_df"], dataset["cat_df"], test_size=0.25)
    assert metrics.auc >= 0.5
    # Predict inefficient burner
    b_feat = pd.DataFrame([{
        "intensity": 0.6, "fuel_flow_nm3h": 250, "angle_deg": 0,
        "age_h": 8000, "avg_tube_t_c": 920,
    }])
    pred = m.predict_inefficient(b_feat)
    assert pred.shape == (1,)
    assert pred[0] in (0, 1)
    # Predict catalyst activity
    c_feat = pd.DataFrame([{
        "age_h": 10000, "t_out_avg_c": 880, "sc_avg": 3.0, "dp_bar": 1.5,
    }])
    act = m.predict_activity(c_feat)
    assert act.shape == (1,)
    assert 0 <= act[0] <= 1.5  # tolerance for slight over-prediction


def test_model_persistence_roundtrip(tmp_path, dataset):
    m = OptimizationModel()
    m.fit(dataset["sc_df"], test_size=0.3)
    p = str(tmp_path / "opt.joblib")
    save_model(m, p)
    m2 = load_model(p)
    feat = pd.DataFrame([{
        "sc_ratio": 3.0, "t_out_c": 880, "feed_flow_nm3h": 32000,
        "cat_age": 0.3, "fuel_flow_nm3h": 11500,
    }])
    a = m.predict_efficiency(feat)[0]
    b = m2.predict_efficiency(feat)[0]
    assert abs(a - b) < 1e-6, "Model round-trip failed"


def test_optimization_feasibility_constraint(dataset):
    """The optimizer must never return S/C below 2.5 or coking risk above 0.5."""
    m = OptimizationModel()
    m.fit(dataset["sc_df"], test_size=0.2)
    for sc_init in [2.6, 3.0, 3.5, 4.0]:
        r = m.optimize({"sc_ratio": sc_init, "t_out_c": 880, "feed_flow_nm3h": 32000,
                         "cat_age": 0.3, "fuel_flow_nm3h": 11500})
        assert r["sc_ratio_opt"] >= 2.5
        assert r["coking_risk_pred"] <= 0.5
