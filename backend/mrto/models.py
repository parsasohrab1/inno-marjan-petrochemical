"""MRTO ML models: four modules that together implement the MRTO system.

Module 1: TemperatureMapModel — regression of wall temperature from features
Module 2: RULModel — regression of remaining useful life + classifier of 90d failure
Module 3: OptimizationModel — surrogate model of efficiency + constrained optimizer
Module 4: BurnerCatalystModel — classifier of inefficient burners + catalyst activity regression

All models use scikit-learn 1.5+ and are persisted as joblib files under models/.
The training script train_models.py uses both synthetic and real data.
"""
from __future__ import annotations

import os
import json
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, accuracy_score


# ────────────────────────────────────────────────────────────────────
# Module 1: Temperature map regression
# ────────────────────────────────────────────────────────────────────
@dataclass
class TempMapMetrics:
    rmse: float
    mae: float
    r2: float
    n_test: int
    feature_names: list[str]


class TemperatureMapModel:
    """Regression of wall temperature from burner features and furnace state."""

    FEATURES = [
        "burner_intensity_avg", "burner_intensity_min", "burner_intensity_max",
        "sc_ratio", "t_out", "feed_flow_nm3h", "fuel_flow_nm3h",
        "row_factor", "col_norm", "anomaly_offset",
    ]

    def __init__(self):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestRegressor(n_estimators=120, max_depth=14,
                                          random_state=42, n_jobs=-1)),
        ])
        self.metrics: TempMapMetrics | None = None

    def _build_features(self, temp_map: pd.DataFrame, timeseries: pd.DataFrame,
                         tubes: pd.DataFrame, burners: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Build feature rows by joining per-tube temperature with furnace state."""
        # Group furnace state by ts_minute (one row per ts)
        ts_df = timeseries.copy()
        ts_df["ts_minute"] = ts_df["ts_minute"].astype(int)
        # Tube metadata map
        tube_meta = tubes.set_index("tube_id")
        # Compute row_factor (1 outer / 0 center) and col_norm (0..1)
        n_rows = int(tubes["row"].max())
        n_cols = int(tubes["col"].max())
        tube_meta = tube_meta.assign(
            row_factor=(0.05 * np.abs(tube_meta["row"] - (n_rows + 1) / 2) / n_rows).values,
            col_norm=((tube_meta["col"] - 1) / max(n_cols - 1, 1)).values,
        )
        # Anomaly offset learned per-tube (mean over time vs design)
        tm = temp_map.copy()
        tm["ts_minute"] = tm["ts_minute"].astype(int)
        tm["tube_id"] = tm["tube_id"].astype(str)
        avg_per_tube = tm.groupby("tube_id")["wall_temp_c"].mean()
        anomaly_offset = (avg_per_tube - 880.0).clip(-80, 80)  # deviation from design
        # Merge furnace state into temp map (forward fill)
        merged = tm.merge(ts_df, on="ts_minute", how="left")
        # Add tube metadata + anomaly offset
        merged["row_factor"] = merged["tube_id"].map(tube_meta["row_factor"]).fillna(0).astype(float)
        merged["col_norm"] = merged["tube_id"].map(tube_meta["col_norm"]).fillna(0).astype(float)
        merged["anomaly_offset"] = merged["tube_id"].map(anomaly_offset).fillna(0).astype(float)
        # Add burner intensity aggregates per ts (simulate: per-tube cannot have burner map here)
        # We compute furnace-level burner intensity proxy from t_out and fuel
        merged["burner_intensity_avg"] = (merged["fuel_flow_nm3h"] / merged["fuel_flow_nm3h"].median()).clip(0.5, 1.5)
        merged["burner_intensity_min"] = merged["burner_intensity_avg"] - 0.1
        merged["burner_intensity_max"] = merged["burner_intensity_avg"] + 0.1
        X = merged[self.FEATURES].fillna(0).astype(float)
        y = merged["wall_temp_c"].astype(float)
        return X, y

    def fit(self, temp_map: pd.DataFrame, timeseries: pd.DataFrame,
            tubes: pd.DataFrame, burners: pd.DataFrame,
            test_size: float = 0.2, random_state: int = 42) -> TempMapMetrics:
        X, y = self._build_features(temp_map, timeseries, tubes, burners)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state)
        self.pipeline.fit(X_tr, y_tr)
        y_pred = self.pipeline.predict(X_te)
        rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
        mae = float(mean_absolute_error(y_te, y_pred))
        r2 = float(r2_score(y_te, y_pred))
        self.metrics = TempMapMetrics(rmse=rmse, mae=mae, r2=r2,
                                      n_test=len(y_te), feature_names=list(X.columns))
        return self.metrics

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(features[self.FEATURES].fillna(0).astype(float))


# ────────────────────────────────────────────────────────────────────
# Module 2: Remaining Useful Life (RUL) regression + failure classifier
# ────────────────────────────────────────────────────────────────────
@dataclass
class RULMetrics:
    mae_hours: float
    rmse_hours: float
    r2: float
    auc_90d: float
    accuracy_90d: float
    n_test: int
    feature_names: list[str]


class RULModel:
    """Predict RUL (hours) and probability of failure within next 90 days."""

    FEATURES = ["avg_wall_temp_c", "consumed_life_fraction",
                "installed_at_days_ago", "row", "col"]

    def __init__(self):
        self.regressor = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestRegressor(n_estimators=150, max_depth=12,
                                          random_state=42, n_jobs=-1)),
        ])
        self.classifier = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(n_estimators=150, max_depth=10,
                                           random_state=42, n_jobs=-1,
                                           class_weight="balanced")),
        ])
        self.metrics: RULMetrics | None = None

    def fit(self, rul_df: pd.DataFrame, tubes: pd.DataFrame,
            test_size: float = 0.2, random_state: int = 42) -> RULMetrics:
        df = rul_df.merge(tubes[["tube_id", "row", "col"]], on="tube_id", how="left").fillna(0)
        df["installed_at_days_ago"] = tubes.set_index("tube_id").reindex(df["tube_id"])["installed_at_days_ago"].fillna(0).values
        X = df[self.FEATURES].astype(float)
        y_rul = df["rul_h"].astype(float)
        y_fail = (df["p_fail_90d"] > 0.3).astype(int)
        X_tr, X_te, yr_tr, yr_te, yf_tr, yf_te = train_test_split(
            X, y_rul, y_fail, test_size=test_size, random_state=random_state, stratify=y_fail if y_fail.nunique() > 1 else None
        )
        self.regressor.fit(X_tr, yr_tr)
        y_pred = self.regressor.predict(X_te)
        mae = float(mean_absolute_error(yr_te, y_pred))
        rmse = float(np.sqrt(mean_squared_error(yr_te, y_pred)))
        r2 = float(r2_score(yr_te, y_pred))
        # Classifier
        try:
            self.classifier.fit(X_tr, yf_tr)
            proba = self.classifier.predict_proba(X_te)[:, 1]
            auc = float(roc_auc_score(yf_te, proba)) if len(np.unique(yf_te)) > 1 else 0.5
            acc = float(accuracy_score(yf_te, self.classifier.predict(X_te)))
        except Exception:
            auc, acc = 0.5, 0.5
        self.metrics = RULMetrics(mae_hours=mae, rmse_hours=rmse, r2=r2,
                                   auc_90d=auc, accuracy_90d=acc,
                                   n_test=len(yr_te), feature_names=list(X.columns))
        return self.metrics

    def predict_rul(self, features: pd.DataFrame) -> np.ndarray:
        return self.regressor.predict(features[self.FEATURES].fillna(0).astype(float))

    def predict_failure_prob(self, features: pd.DataFrame) -> np.ndarray:
        proba = self.classifier.predict_proba(features[self.FEATURES].fillna(0).astype(float))
        return proba[:, 1] if proba.shape[1] > 1 else np.zeros(len(features))


# ────────────────────────────────────────────────────────────────────
# Module 3: S/C optimization surrogate + optimizer
# ────────────────────────────────────────────────────────────────────
@dataclass
class OptimMetrics:
    eff_rmse: float
    eff_r2: float
    eff_mae: float
    n_test: int
    feature_names: list[str]


class OptimizationModel:
    """Surrogate model of conversion efficiency + constrained optimizer."""

    FEATURES = ["sc_ratio", "t_out_c", "feed_flow_nm3h", "cat_age", "fuel_flow_nm3h"]

    def __init__(self):
        self.eff_model = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                              learning_rate=0.1, random_state=42)),
        ])
        self.coking_model = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", GradientBoostingRegressor(n_estimators=150, max_depth=4,
                                              learning_rate=0.1, random_state=42)),
        ])
        self.metrics: OptimMetrics | None = None
        # Bounds for optimization (set during fit from training data)
        self.bounds = {
            "sc_ratio": (2.5, 4.5),
            "t_out_c": (820.0, 950.0),
            "feed_flow_nm3h": (28000.0, 36000.0),
            "cat_age": (0.05, 1.0),
            "fuel_flow_nm3h": (9000.0, 14000.0),
        }

    def fit(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> OptimMetrics:
        X = df[self.FEATURES].astype(float)
        y_eff = df["efficiency"].astype(float)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y_eff, test_size=test_size, random_state=random_state)
        self.eff_model.fit(X_tr, y_tr)
        self.coking_model.fit(X_tr, df["coking_risk"].astype(float).loc[X_tr.index])
        y_pred = self.eff_model.predict(X_te)
        mae = float(mean_absolute_error(y_te, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
        r2 = float(r2_score(y_te, y_pred))
        self.metrics = OptimMetrics(eff_rmse=rmse, eff_r2=r2, eff_mae=mae,
                                    n_test=len(y_te), feature_names=list(X.columns))
        return self.metrics

    def predict_efficiency(self, df: pd.DataFrame) -> np.ndarray:
        return self.eff_model.predict(df[self.FEATURES].astype(float))

    def predict_coking_risk(self, df: pd.DataFrame) -> np.ndarray:
        return self.coking_model.predict(df[self.FEATURES].astype(float))

    def optimize(self, current: dict, alpha_eff: float = 0.7, alpha_fuel: float = 0.3,
                 n_grid: int = 25) -> dict:
        """Grid-search optimum: maximize alpha_eff * efficiency - alpha_fuel * fuel_norm.

        Returns dict with optimal S/C, T_out, predicted efficiency, coking risk,
        fuel, fuel savings %, and the explored grid summary.
        """
        sc_grid = np.linspace(*self.bounds["sc_ratio"], n_grid)
        t_out_grid = np.linspace(*self.bounds["t_out_c"], n_grid)
        best = None
        for sc in sc_grid:
            for t_out in t_out_grid:
                feat = pd.DataFrame([{
                    "sc_ratio": sc,
                    "t_out_c": t_out,
                    "feed_flow_nm3h": current.get("feed_flow_nm3h", 32000),
                    "cat_age": current.get("cat_age", 0.3),
                    "fuel_flow_nm3h": current.get("fuel_flow_nm3h", 11500),
                }])
                eff = float(self.eff_model.predict(feat)[0])
                coking = float(self.coking_model.predict(feat)[0])
                # Fuel consumed ~ scales with T_out (proxy)
                fuel = 11500 * (0.7 + 0.3 * (t_out - 800) / 150)
                # Constraints: coking < 0.5, S/C >= 2.5
                if coking > 0.5 or sc < 2.5:
                    continue
                fuel_norm = fuel / 14000
                score = alpha_eff * eff - alpha_fuel * fuel_norm
                if best is None or score > best["score"]:
                    best = {
                        "sc_ratio_opt": float(sc),
                        "t_out_c_opt": float(t_out),
                        "efficiency_pred": eff,
                        "coking_risk_pred": coking,
                        "fuel_pred_nm3h": float(fuel),
                        "score": float(score),
                    }
        if best is None:
            return {"error": "no feasible point found"}
        # fuel savings vs current
        cur_fuel = current.get("fuel_flow_nm3h", 11500)
        fuel_saving_pct = float((cur_fuel - best["fuel_pred_nm3h"]) / max(cur_fuel, 1) * 100)
        best["fuel_saving_pct"] = fuel_saving_pct
        return best


# ────────────────────────────────────────────────────────────────────
# Module 4: Burner & catalyst monitoring
# ────────────────────────────────────────────────────────────────────
@dataclass
class BurnerCatalystMetrics:
    auc: float
    accuracy: float
    activity_mae: float
    activity_r2: float
    n_test: int
    feature_names: list[str]


class BurnerCatalystModel:
    """Classify inefficient burners and regress catalyst activity."""

    BURNER_FEATURES = ["intensity", "fuel_flow_nm3h", "angle_deg", "age_h", "avg_tube_t_c"]
    CATALYST_FEATURES = ["age_h", "t_out_avg_c", "sc_avg", "dp_bar"]

    def __init__(self):
        self.burner_clf = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(n_estimators=150, max_depth=10,
                                           random_state=42, n_jobs=-1,
                                           class_weight="balanced")),
        ])
        self.cat_reg = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestRegressor(n_estimators=150, max_depth=12,
                                          random_state=42, n_jobs=-1)),
        ])
        self.metrics: BurnerCatalystMetrics | None = None

    def fit(self, burner_df: pd.DataFrame, catalyst_df: pd.DataFrame,
            test_size: float = 0.2, random_state: int = 42) -> BurnerCatalystMetrics:
        Xb = burner_df[self.BURNER_FEATURES].astype(float)
        yb = burner_df["label_inefficient"].astype(int)
        Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(Xb, yb, test_size=test_size, random_state=random_state,
                                                       stratify=yb if yb.nunique() > 1 else None)
        self.burner_clf.fit(Xb_tr, yb_tr)
        proba = self.burner_clf.predict_proba(Xb_te)
        auc = float(roc_auc_score(yb_te, proba[:, 1])) if len(np.unique(yb_te)) > 1 else 0.5
        acc = float(accuracy_score(yb_te, self.burner_clf.predict(Xb_te)))
        # Catalyst regression
        Xc = catalyst_df[self.CATALYST_FEATURES].astype(float)
        yc = catalyst_df["activity"].astype(float)
        Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=test_size, random_state=random_state)
        self.cat_reg.fit(Xc_tr, yc_tr)
        yc_pred = self.cat_reg.predict(Xc_te)
        mae = float(mean_absolute_error(yc_te, yc_pred))
        r2 = float(r2_score(yc_te, yc_pred))
        self.metrics = BurnerCatalystMetrics(auc=auc, accuracy=acc, activity_mae=mae,
                                              activity_r2=r2, n_test=len(yb_te),
                                              feature_names=list(Xb.columns))
        return self.metrics

    def predict_inefficient(self, df: pd.DataFrame) -> np.ndarray:
        return self.burner_clf.predict(df[self.BURNER_FEATURES].astype(float))

    def predict_activity(self, df: pd.DataFrame) -> np.ndarray:
        return self.cat_reg.predict(df[self.CATALYST_FEATURES].astype(float))


# ────────────────────────────────────────────────────────────────────
# Persistence
# ────────────────────────────────────────────────────────────────────
def save_model(model, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)


def save_metrics(metrics, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(metrics), f, indent=2, default=str)
