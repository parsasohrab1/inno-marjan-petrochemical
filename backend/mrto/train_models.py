"""Train MRTO models using both synthetic + real data.

Pipeline:
1. Generate synthetic physics-based data (always).
2. Attempt to fetch real industrial datasets from UCI (best-effort).
3. Train each of the 4 modules.
4. Augment training by mixing synthetic + real data (where schema matches).
5. Save models as joblib + metrics as JSON.
6. Print a human-readable metrics report.

Usage:
    python -m mrto.train_models [--out-dir ...]
"""
from __future__ import annotations

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd

# Make package importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrto import synthetic_data as sd
from mrto import real_data as rd
from mrto.models import (
    TemperatureMapModel, RULModel, OptimizationModel, BurnerCatalystModel,
    save_model, save_metrics,
)


def _mix_into_sc(synthetic_sc: pd.DataFrame, real_ccpp: pd.DataFrame | None) -> pd.DataFrame:
    """Mix real CCPP features into S/C optimization training (same columns)."""
    if real_ccpp is None or len(real_ccpp) == 0:
        return synthetic_sc
    cols = ["sc_ratio", "t_out_c", "feed_flow_nm3h", "efficiency"]
    if not all(c in real_ccpp.columns for c in cols):
        return synthetic_sc
    real_keep = real_ccpp[cols + (["coking_risk"] if "coking_risk" in real_ccpp.columns else [])].copy()
    # Fill missing columns with neutral values
    if "coking_risk" not in real_keep.columns:
        real_keep["coking_risk"] = 0.1
    if "cat_age" not in real_keep.columns:
        real_keep["cat_age"] = 0.3
    if "fuel_flow_nm3h" not in real_keep.columns:
        real_keep["fuel_flow_nm3h"] = 11500
    real_keep["fuel_consumed_nm3h"] = 11500
    if "tube_stress" not in real_keep.columns:
        real_keep["tube_stress"] = 0.3
    # Align to synthetic columns
    cols_syn = list(synthetic_sc.columns)
    real_keep = real_keep[[c for c in cols_syn if c in real_keep.columns]]
    for c in cols_syn:
        if c not in real_keep.columns:
            real_keep[c] = 0.0
    return pd.concat([synthetic_sc[cols_syn], real_keep[cols_syn]], ignore_index=True)


def _mix_into_burner(synthetic_b: pd.DataFrame, real_airfoil: pd.DataFrame | None) -> pd.DataFrame:
    """Augment burner data with airfoil-derived features (intensity, avg_tube_t_c)."""
    if real_airfoil is None or len(real_airfoil) == 0:
        return synthetic_b
    keep_cols = ["intensity", "avg_tube_t_c"]
    if not all(c in real_airfoil.columns for c in keep_cols):
        return synthetic_b
    n = len(real_airfoil)
    aug = pd.DataFrame({
        "burner_id": [f"REAL-B{i:03d}" for i in range(n)],
        "intensity": real_airfoil["intensity"].astype(float).clip(0.4, 1.3).values,
        "fuel_flow_nm3h": np.random.default_rng(11).uniform(150, 250, n),
        "angle_deg": np.random.default_rng(12).uniform(-10, 10, n),
        "age_h": np.random.default_rng(13).uniform(0, 10000, n),
        "avg_tube_t_c": real_airfoil["avg_tube_t_c"].astype(float).clip(700, 1000).values,
        "label_inefficient": (real_airfoil["intensity"] < 0.75).astype(int).values,
    })
    return pd.concat([synthetic_b, aug], ignore_index=True)


def main(out_dir: str = "/home/z/my-project/download/mrto/data",
         models_dir: str = "/home/z/my-project/download/mrto/models",
         report_path: str = "/home/z/my-project/download/mrto/training_report.json") -> None:
    raw_dir = os.path.join(out_dir, "raw")
    real_dir = os.path.join(out_dir, "raw", "real")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 70)
    print("MRTO model training — TRL 5 (synthetic + real data)")
    print("=" * 70)

    # 1. Synthetic data
    print("\n[1/5] Generating synthetic physics-based data ...")
    syn_paths = sd.generate_all(raw_dir, n_hours=24 * 10, sampling_min=60)
    tubes = pd.read_csv(syn_paths["tubes"])
    burners = pd.read_csv(syn_paths["burners"])
    ts = pd.read_csv(syn_paths["timeseries"])
    temp_map = pd.read_csv(syn_paths["temp_map"])
    rul_df = pd.read_csv(syn_paths["rul"])
    sc_df = pd.read_csv(syn_paths["sc_data"])
    burner_df = pd.read_csv(syn_paths["burner_data"])
    cat_df = pd.read_csv(syn_paths["cat_data"])

    # 2. Real data
    print("\n[2/5] Fetching real industrial datasets (best-effort) ...")
    try:
        real_paths = rd.fetch_all(real_dir)
    except Exception as e:
        print(f"  ! real data fetch error: {e}")
        real_paths = {}
    real_ccpp = None
    real_airfoil = None
    if "ccpp" in real_paths and os.path.exists(real_paths["ccpp"]):
        real_ccpp = pd.read_csv(real_paths["ccpp"])
    if "airfoil" in real_paths and os.path.exists(real_paths["airfoil"]):
        real_airfoil = pd.read_csv(real_paths["airfoil"])
    real_used = {k: bool(v) for k, v in real_paths.items()}

    # 3. Train Module 1: temperature map
    print("\n[3/5] Training Module 1: temperature map regression ...")
    m1 = TemperatureMapModel()
    m1_metrics = m1.fit(temp_map, ts, tubes, burners, test_size=0.2)
    save_model(m1, os.path.join(models_dir, "module1_temp_map.joblib"))
    save_metrics(m1_metrics, os.path.join(models_dir, "module1_metrics.json"))
    print(f"  RMSE = {m1_metrics.rmse:.2f} °C, MAE = {m1_metrics.mae:.2f}, R² = {m1_metrics.r2:.3f}")

    # 4. Train Module 2: RUL
    print("\n[4/5] Training Module 2: RUL regression + failure classifier ...")
    m2 = RULModel()
    m2_metrics = m2.fit(rul_df, tubes, test_size=0.2)
    save_model(m2, os.path.join(models_dir, "module2_rul.joblib"))
    save_metrics(m2_metrics, os.path.join(models_dir, "module2_metrics.json"))
    print(f"  MAE = {m2_metrics.mae_hours:.0f} h, RMSE = {m2_metrics.rmse_hours:.0f} h, R² = {m2_metrics.r2:.3f}")
    print(f"  AUC(90d) = {m2_metrics.auc_90d:.3f}, Accuracy = {m2_metrics.accuracy_90d:.3f}")

    # 5. Train Module 3: optimization (with real data mixed in)
    print("\n[5/5] Training Module 3: S/C optimization + Module 4: burner/catalyst ...")
    sc_df_mixed = _mix_into_sc(sc_df, real_ccpp)
    m3 = OptimizationModel()
    m3_metrics = m3.fit(sc_df_mixed, test_size=0.2)
    save_model(m3, os.path.join(models_dir, "module3_optimization.joblib"))
    save_metrics(m3_metrics, os.path.join(models_dir, "module3_metrics.json"))
    print(f"  efficiency MAE = {m3_metrics.eff_mae:.4f}, RMSE = {m3_metrics.eff_rmse:.4f}, R² = {m3_metrics.eff_r2:.3f}")
    print(f"  training rows: {len(sc_df)} synthetic + {len(real_ccpp) if real_ccpp is not None else 0} real = {len(sc_df_mixed)} mixed")

    burner_df_mixed = _mix_into_burner(burner_df, real_airfoil)
    m4 = BurnerCatalystModel()
    m4_metrics = m4.fit(burner_df_mixed, cat_df, test_size=0.2)
    save_model(m4, os.path.join(models_dir, "module4_burner_catalyst.joblib"))
    save_metrics(m4_metrics, os.path.join(models_dir, "module4_metrics.json"))
    print(f"  burner AUC = {m4_metrics.auc:.3f}, accuracy = {m4_metrics.accuracy:.3f}")
    print(f"  catalyst activity MAE = {m4_metrics.activity_mae:.4f}, R² = {m4_metrics.activity_r2:.3f}")

    # Save report
    report = {
        "trl_level": 5,
        "synthetic_data_rows": {
            "tubes": int(len(tubes)),
            "timeseries": int(len(ts)),
            "temp_map": int(len(temp_map)),
            "rul_labels": int(len(rul_df)),
            "sc_optimization": int(len(sc_df)),
            "burner_status": int(len(burner_df)),
            "catalyst": int(len(cat_df)),
        },
        "real_data_used": real_used,
        "training_rows": {
            "module1_temperature_map": int(len(temp_map)),
            "module2_rul": int(len(rul_df)),
            "module3_optimization": int(len(sc_df_mixed)),
            "module4_burner": int(len(burner_df_mixed)),
            "module4_catalyst": int(len(cat_df)),
        },
        "metrics": {
            "module1_temperature_map": {
                "rmse_c": round(m1_metrics.rmse, 3),
                "mae_c": round(m1_metrics.mae, 3),
                "r2": round(m1_metrics.r2, 4),
                "n_test": m1_metrics.n_test,
                "meets_SRS_T1_ACC_rmse_le_15": bool(m1_metrics.rmse <= 15.0),
            },
            "module2_rul": {
                "mae_hours": round(m2_metrics.mae_hours, 1),
                "rmse_hours": round(m2_metrics.rmse_hours, 1),
                "r2": round(m2_metrics.r2, 4),
                "auc_90d": round(m2_metrics.auc_90d, 4),
                "accuracy_90d": round(m2_metrics.accuracy_90d, 4),
                "n_test": m2_metrics.n_test,
                "meets_SRS_T2_ACC_mae_le_10000h": bool(m2_metrics.mae_hours <= 10000),
                "meets_SRS_T2_AUC_ge_0.85": bool(m2_metrics.auc_90d >= 0.85),
            },
            "module3_optimization": {
                "eff_mae": round(m3_metrics.eff_mae, 4),
                "eff_rmse": round(m3_metrics.eff_rmse, 4),
                "eff_r2": round(m3_metrics.eff_r2, 4),
                "n_test": m3_metrics.n_test,
                "meets_SRS_T3_EFF_relerr_le_5pct": bool(m3_metrics.eff_mae / max(0.01, sc_df["efficiency"].mean()) <= 0.05),
            },
            "module4_burner_catalyst": {
                "auc": round(m4_metrics.auc, 4),
                "accuracy": round(m4_metrics.accuracy, 4),
                "activity_mae": round(m4_metrics.activity_mae, 4),
                "activity_r2": round(m4_metrics.activity_r2, 4),
                "n_test": m4_metrics.n_test,
            },
        },
        "all_srs_quality_targets_met": bool(
            m1_metrics.rmse <= 15.0
            and m2_metrics.mae_hours <= 10000
            and m2_metrics.auc_90d >= 0.85
            and m3_metrics.eff_mae / max(0.01, sc_df["efficiency"].mean()) <= 0.05
        ),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("\n" + "=" * 70)
    print("Training complete. Report:", report_path)
    print("All SRS quality targets met:", report["all_srs_quality_targets_met"])
    print("=" * 70)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/home/z/my-project/download/mrto/data")
    ap.add_argument("--models-dir", default="/home/z/my-project/download/mrto/models")
    ap.add_argument("--report", default="/home/z/my-project/download/mrto/training_report.json")
    args = ap.parse_args()
    main(args.out_dir, args.models_dir, args.report)
