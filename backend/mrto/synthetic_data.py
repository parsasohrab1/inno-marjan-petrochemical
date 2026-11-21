"""Synthetic data generator for MRTO system.

Generates physics-based synthetic data for a methanol steam methane reformer
(SMR) furnace. The data simulates:

1. Tube wall temperature distribution driven by burner heat flux and feed flow.
2. Creep-driven remaining useful life (RUL) using Larson-Miller parameter.
3. Steam-to-carbon (S/C) ratio vs conversion efficiency and coking risk.
4. Burner status, catalyst deactivation, and pressure drop.

The generator is deterministic (seeded) so tests are reproducible.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ────────────────────────────────────────────────────────────────────
# Constants (typical industrial methanol SMR values)
# ────────────────────────────────────────────────────────────────────
SEED = 1405
RNG = np.random.default_rng(SEED)

N_ROWS = 4  # rows of tubes in furnace
N_COLS = 80  # tubes per row
N_TUBES = N_ROWS * N_COLS  # 320 tubes
N_BURNERS = 64  # burners distributed across furnace
T_AMBIENT = 35.0  # °C outside furnace
T_DESIGN = 880.0  # °C nominal design wall temperature
P_DESIGN = 25.0  # bar nominal operating pressure
DESIGN_LIFE_H = 100_000.0  # 100,000 hours nominal tube design life
S_C_LOW = 2.5
S_C_HIGH = 4.5
T_OUT_LOW = 820.0
T_OUT_HIGH = 950.0

# HP-Modified creep constants (illustrative, calibration-grade)
LMP_C = 20.0  # Larson-Miller constant (typical for HP-Nb alloy)
# Rupture time (hours) ~= 10^(LMP/1000 * f - g), with LMP = T_K * (C + log10(t_r))


@dataclass
class FurnaceConfig:
    n_rows: int = N_ROWS
    n_cols: int = N_COLS
    n_burners: int = N_BURNERS
    design_life_h: float = DESIGN_LIFE_H
    t_ambient: float = T_AMBIENT
    t_design: float = T_DESIGN
    p_design: float = P_DESIGN
    seed: int = SEED


def lmp_to_rupture_time(t_wall_c: float, c: float = LMP_C, lmp_target: float = 28500.0) -> float:
    """Approximate rupture time (hours) for a given wall temperature.

    LMP = T_K * (C + log10(t_r)). Solve for t_r given a target LMP for the alloy.
    For HP-Modified at ~880°C design, LMP_target ≈ 28,500 corresponds to ~100,000 h life.
    """
    t_k = t_wall_c + 273.15
    if t_k <= 0:
        return 0.0
    log_t = lmp_target / t_k - c
    if log_t < -20:
        return 0.0
    if log_t > 20:
        return float("inf")
    return 10 ** log_t


def generate_tubes(cfg: FurnaceConfig) -> pd.DataFrame:
    """Generate tube metadata: id, row, col, install date, last replace, material."""
    rng = np.random.default_rng(cfg.seed)
    rows = []
    for i in range(cfg.n_rows * cfg.n_cols):
        row = i // cfg.n_cols
        col = i % cfg.n_cols
        # Random install age in years 0..6 (older near furnace center)
        age_years = float(rng.uniform(0.5, 6.0))
        rows.append({
            "tube_id": f"T{row+1:02d}-{col+1:03d}",
            "row": int(row + 1),
            "col": int(col + 1),
            "material": "HP-Modified",
            "installed_at_days_ago": int(age_years * 365),
            "last_replaced_days_ago": int(age_years * 365),
            "design_life_h": cfg.design_life_h,
        })
    return pd.DataFrame(rows)


def generate_burner_profiles(cfg: FurnaceConfig) -> pd.DataFrame:
    """Each burner has a base intensity, age-based drift, and noise."""
    rng = np.random.default_rng(cfg.seed + 1)
    # Burners arranged in a row, mapped to tube columns
    base = rng.uniform(0.85, 1.15, size=cfg.n_burners)
    age_factor = rng.uniform(0.92, 1.0, size=cfg.n_burners)  # older = weaker
    return pd.DataFrame({
        "burner_id": [f"B{i+1:03d}" for i in range(cfg.n_burners)],
        "base_intensity": base,
        "age_factor": age_factor,
        "nominal_fuel_flow": rng.uniform(180, 240, size=cfg.n_burners),
    })


def _burner_influence(col: int, n_cols: int, n_burners: int) -> np.ndarray:
    """Each tube col influenced by 2 nearest burners (linear interp)."""
    # burners map to columns uniformly
    burner_positions = np.linspace(0, n_cols - 1, n_burners)
    # for each tube col, compute weights to all burners (1/distance)
    diff = np.abs(burner_positions[None, :] - col)
    weights = 1.0 / (diff + 1.5)
    weights = weights / weights.sum()
    return weights


def generate_time_series(cfg: FurnaceConfig, n_hours: int = 24 * 30, sampling_min: int = 5) -> pd.DataFrame:
    """Generate operational time-series: S/C, T_out, feed_flow, fuel_flow, pressures.

    Each row = one sampling timestamp; columns are aggregate furnace values.
    """
    rng = np.random.default_rng(cfg.seed + 2)
    n = int(n_hours * 60 / sampling_min)
    t = np.arange(n) * sampling_min  # minutes
    t_hour = t / 60.0

    # S/C slowly varies 2.7..3.6, daily oscillation + slow trend
    sc = 3.1 + 0.25 * np.sin(2 * np.pi * t_hour / 24) + 0.1 * np.sin(2 * np.pi * t_hour / 168) + rng.normal(0, 0.05, n)
    sc = np.clip(sc, S_C_LOW, S_C_HIGH)
    # T_out varies 850..920 with hourly noise
    t_out = 880 + 18 * np.sin(2 * np.pi * t_hour / 24) + rng.normal(0, 8, n)
    t_out = np.clip(t_out, T_OUT_LOW, T_OUT_HIGH)
    # Feed flow (Nm3/h)
    feed = 32000 + 1200 * np.sin(2 * np.pi * t_hour / 24) + rng.normal(0, 200, n)
    # Fuel flow (Nm3/h)
    fuel = 11500 + 600 * np.sin(2 * np.pi * t_hour / 24) + rng.normal(0, 150, n)
    # Pressure in/out
    p_in = 25.5 + rng.normal(0, 0.15, n)
    p_out = 22.0 + rng.normal(0, 0.15, n)
    dp = p_in - p_out

    return pd.DataFrame({
        "ts_minute": t,
        "ts_hour": t_hour,
        "sc_ratio": sc,
        "t_out": t_out,
        "feed_flow_nm3h": feed,
        "fuel_flow_nm3h": fuel,
        "p_in_bar": p_in,
        "p_out_bar": p_out,
        "dp_bar": dp,
    })


def generate_temperature_map(
    tubes: pd.DataFrame,
    burners: pd.DataFrame,
    ts: pd.DataFrame,
    cfg: FurnaceConfig,
) -> pd.DataFrame:
    """For each tube and each timestamp, simulate wall temperature.

    Wall T = T_ambient + base_delta + burner_effect + feed_effect + noise + anomalies.
    Some tubes randomly have persistent hotspots (coking) or cold spots (blockage).
    """
    rng = np.random.default_rng(cfg.seed + 3)
    n_t = len(ts)
    n_tubes = len(tubes)

    # Persistent anomalies on ~5% of tubes (hotspot) and 2% (coldspot)
    n_hot = int(0.05 * n_tubes)
    n_cold = int(0.02 * n_tubes)
    hot_idx = rng.choice(n_tubes, n_hot, replace=False)
    cold_idx = rng.choice(np.setdiff1d(np.arange(n_tubes), hot_idx), n_cold, replace=False)

    anomaly = np.zeros(n_tubes)
    anomaly[hot_idx] = rng.uniform(35, 65, n_hot)  # °C above baseline
    anomaly[cold_idx] = -rng.uniform(20, 45, n_cold)  # °C below baseline

    # Burner-to-tube column influence matrix
    influence = np.zeros((cfg.n_cols, cfg.n_burners))
    for col in range(cfg.n_cols):
        influence[col] = _burner_influence(col, cfg.n_cols, cfg.n_burners)

    # Burner intensity time-series: varies slightly per timestamp
    burner_intensity = np.empty((n_t, cfg.n_burners))
    base_b = burners["base_intensity"].values
    age_b = burners["age_factor"].values
    # One burner fails (drops to 70%) randomly over time
    fail_burner = rng.integers(0, cfg.n_burners)
    for t_idx in range(n_t):
        # Slow drift + small noise
        drift = 0.05 * np.sin(2 * np.pi * ts["ts_hour"].iloc[t_idx] / 168)
        burner_intensity[t_idx] = base_b * age_b * (1 + drift) + rng.normal(0, 0.02, cfg.n_burners)
        # Inject a failing burner for hours 100..250
        h = ts["ts_hour"].iloc[t_idx]
        if 100 < h < 250:
            burner_intensity[t_idx, fail_burner] *= 0.75

    # Build per-tube wall temperature
    rows = []
    # vectorized over tubes for memory efficiency in chunks
    chunk = 80
    for start in range(0, n_tubes, chunk):
        chunk_ids = tubes["tube_id"].iloc[start:start + chunk].tolist()
        chunk_cols = tubes["col"].iloc[start:start + chunk].values - 1
        chunk_rows = tubes["row"].iloc[start:start + chunk].values - 1
        chunk_anom = anomaly[start:start + chunk]
        # row_factor: outer rows see less direct flame
        row_factor = 1.0 - 0.05 * np.abs(chunk_rows - (cfg.n_rows - 1) / 2) / cfg.n_rows
        for k, tube_id in enumerate(chunk_ids):
            col = int(chunk_cols[k])
            w = influence[col]  # weights over burners (length n_burners)
            # weighted average burner intensity per timestamp
            bi = burner_intensity @ w  # length n_t
            # wall temperature relative to design
            delta = (
                (bi - 1.0) * 80.0  # ±80 °C around design for ±20% intensity
                + row_factor[k] * 10.0  # outer rows slightly cooler
                + chunk_anom[k]  # persistent anomaly
                + 0.02 * (ts["t_out"].values - T_DESIGN)  # furnace T_out effect
                + rng.normal(0, 3.0, n_t)  # measurement noise
            )
            wall_t = T_AMBIENT + (T_DESIGN - T_AMBIENT) + delta
            # clip to plausible
            wall_t = np.clip(wall_t, 350.0, 1050.0)
            for t_idx in range(n_t):
                rows.append({
                    "tube_id": tube_id,
                    "ts_minute": int(ts["ts_minute"].iloc[t_idx]),
                    "wall_temp_c": round(float(wall_t[t_idx]), 2),
                    "source": "ir_simulation",
                })
    return pd.DataFrame(rows)


def generate_rul_labels(tubes: pd.DataFrame, temp_map: pd.DataFrame, cfg: FurnaceConfig) -> pd.DataFrame:
    """Compute cumulative thermal damage and remaining useful life per tube.

    Approach:
    - For each tube, compute average wall temperature over the time window.
    - Estimate rupture time at that temperature using Larson-Miller.
    - Consumed life fraction = elapsed_hours / rupture_time.
    - RUL (hours) = rupture_time - elapsed_hours (clipped at 0).
    - Failure probability in next 90 days = sigmoid of (consumed_fraction - 0.8).
    """
    elapsed_h = tubes["last_replaced_days_ago"].values * 24.0
    grouped = temp_map.groupby("tube_id")["wall_temp_c"].mean()
    rows = []
    for _, tube in tubes.iterrows():
        tid = tube["tube_id"]
        avg_t = float(grouped.get(tid, T_DESIGN))
        t_rupture = lmp_to_rupture_time(avg_t)
        if not math.isfinite(t_rupture) or t_rupture <= 0:
            t_rupture = 5 * DESIGN_LIFE_H
        consumed = float(elapsed_h[tubes.index[tubes["tube_id"] == tid][0]]) / t_rupture
        rul = max(0.0, t_rupture - float(elapsed_h[tubes.index[tubes["tube_id"] == tid][0]]))
        # Failure probability next 90 days = sigmoid
        p_fail_90 = 1.0 / (1.0 + math.exp(-8.0 * (consumed - 0.85)))
        p_fail_30 = 1.0 / (1.0 + math.exp(-10.0 * (consumed - 0.95)))
        rows.append({
            "tube_id": tid,
            "avg_wall_temp_c": round(avg_t, 2),
            "consumed_life_fraction": round(consumed, 4),
            "rul_h": round(rul, 0),
            "p_fail_30d": round(p_fail_30, 4),
            "p_fail_90d": round(p_fail_90, 4),
            "label_rul_class": int(rul < 20000),  # 1 = high risk
        })
    return pd.DataFrame(rows)


def generate_sc_optimization_dataset(cfg: FurnaceConfig, n_samples: int = 4000) -> pd.DataFrame:
    """Generate dataset for S/C optimization: features -> efficiency, fuel, coking risk.

    Physics (calibration-grade):
    - Conversion efficiency eta(CH4) increases with T_out, decreases if S/C too low (coking)
      and if catalyst aged. Peak around S/C ~ 2.8.
    - Fuel consumption increases with T_out and feed.
    - Coking risk increases when S/C < 2.7 and T_out high.
    """
    rng = np.random.default_rng(cfg.seed + 4)
    sc = rng.uniform(S_C_LOW, S_C_HIGH, n_samples)
    t_out = rng.uniform(T_OUT_LOW, T_OUT_HIGH, n_samples)
    feed = rng.uniform(28000, 36000, n_samples)
    cat_age = rng.uniform(0.05, 1.0, n_samples)  # 0 = new, 1 = end-of-life
    fuel = rng.uniform(9000, 14000, n_samples)

    # Equilibrium-like CH4 conversion (illustrative)
    # eta = eta_max * (1 - exp(-a*(T_out-700))) * (1 - b*(2.8 - S/C)^2 when S/C<2.8) * (1 - 0.2*cat_age)
    eta_max = 0.93
    a_t = 0.012
    eta = eta_max * (1 - np.exp(-a_t * (t_out - 700)))
    # S/C penalty: peak at 2.8, quadratic penalty below 2.8 (coking) and slight penalty above 3.5 (energy waste)
    sc_penalty = np.where(
        sc < 2.8,
        1 - 0.35 * (2.8 - sc) ** 2,
        1 - 0.05 * (sc - 3.2) ** 2,
    )
    eta = eta * np.clip(sc_penalty, 0.3, 1.0)
    eta = eta * (1 - 0.18 * cat_age)
    eta = np.clip(eta, 0.0, 0.98)
    # measurement noise
    eta_obs = eta + rng.normal(0, 0.012, n_samples)

    # Fuel consumed to reach T_out (heat balance simplified)
    # Q_in = m_feed * Cp * (T_out - T_in) + heat losses; fuel = Q_in / LHV
    # For synthetic generation we keep an interpretable proxy:
    fuel_obs = fuel * (0.6 + 0.4 * (t_out - 800) / 150) * (0.95 + 0.1 * np.sin(feed / 1000))
    fuel_obs = fuel_obs + rng.normal(0, 200, n_samples)

    # Coking risk: high when S/C < 2.7 and T_out > 920 and cat_age high
    coking_risk = (
        0.5 * np.clip((2.7 - sc) / 0.5, 0, 1)
        + 0.3 * np.clip((t_out - 900) / 50, 0, 1)
        + 0.2 * cat_age
    )
    coking_risk = np.clip(coking_risk + rng.normal(0, 0.04, n_samples), 0, 1)

    # Pressure stress on tubes (proxy): increases with T_out and decreases with S/C
    tube_stress = (
        0.5 * np.clip((t_out - 850) / 100, 0, 1)
        + 0.3 * np.clip((3.5 - sc) / 1.0, 0, 1)
        + 0.2 * cat_age
    )
    tube_stress = np.clip(tube_stress + rng.normal(0, 0.03, n_samples), 0, 1)

    return pd.DataFrame({
        "sc_ratio": sc,
        "t_out_c": t_out,
        "feed_flow_nm3h": feed,
        "cat_age": cat_age,
        "fuel_flow_nm3h": fuel,
        "efficiency": np.round(eta_obs, 4),
        "fuel_consumed_nm3h": np.round(fuel_obs, 1),
        "coking_risk": np.round(coking_risk, 4),
        "tube_stress": np.round(tube_stress, 4),
    })


def generate_burner_status_dataset(cfg: FurnaceConfig, n_samples: int = 3000) -> pd.DataFrame:
    """Generate burner status dataset: burner features + label inefficient."""
    rng = np.random.default_rng(cfg.seed + 5)
    intensity = rng.uniform(0.4, 1.3, n_samples)
    fuel = rng.uniform(120, 280, n_samples)
    angle = rng.uniform(-15, 15, n_samples)
    age_h = rng.uniform(0, 12000, n_samples)
    # Average tube col temperature this burner faces
    avg_tube_t = 800 + 80 * (intensity - 0.7) + rng.normal(0, 12, n_samples)
    # Inefficient if intensity < 0.7 OR (fuel > 220 and intensity < 0.85)
    inefficient = ((intensity < 0.7) | ((fuel > 220) & (intensity < 0.85))).astype(int)
    return pd.DataFrame({
        "burner_id": [f"B{rng.integers(1, cfg.n_burners+1):03d}" for _ in range(n_samples)],
        "intensity": intensity,
        "fuel_flow_nm3h": fuel,
        "angle_deg": angle,
        "age_h": age_h,
        "avg_tube_t_c": avg_tube_t,
        "label_inefficient": inefficient,
    })


def generate_catalyst_deactivation(cfg: FurnaceConfig, n_samples: int = 2000) -> pd.DataFrame:
    """Generate catalyst deactivation dataset."""
    rng = np.random.default_rng(cfg.seed + 6)
    age_h = rng.uniform(0, 30000, n_samples)  # hours since last reload
    t_out_avg = rng.uniform(840, 920, n_samples)
    sc_avg = rng.uniform(S_C_LOW, S_C_HIGH, n_samples)
    dp_bar = rng.uniform(0.8, 3.5, n_samples)  # pressure drop across catalyst bed
    # True activity fraction: decreases with age, dp (sintering/fouling)
    activity = 1.0 / (1.0 + 0.00005 * age_h + 0.4 * (dp_bar - 1.0)) * (1 - 0.0001 * (t_out_avg - 800))
    activity = np.clip(activity + rng.normal(0, 0.02, n_samples), 0.05, 1.0)
    return pd.DataFrame({
        "age_h": age_h,
        "t_out_avg_c": t_out_avg,
        "sc_avg": sc_avg,
        "dp_bar": dp_bar,
        "activity": np.round(activity, 4),
        "label_replace": (activity < 0.4).astype(int),
    })


# ────────────────────────────────────────────────────────────────────
# Main entry
# ────────────────────────────────────────────────────────────────────
def generate_all(out_dir: str, n_hours: int = 24 * 14, sampling_min: int = 30) -> dict:
    """Generate all synthetic datasets and save to out_dir (Parquet + CSV).

    Returns dict of file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    cfg = FurnaceConfig()
    paths = {}

    tubes = generate_tubes(cfg)
    burners = generate_burner_profiles(cfg)
    ts = generate_time_series(cfg, n_hours=n_hours, sampling_min=sampling_min)
    temp_map = generate_temperature_map(tubes, burners, ts, cfg)
    rul = generate_rul_labels(tubes, temp_map, cfg)
    sc_data = generate_sc_optimization_dataset(cfg)
    burner_data = generate_burner_status_dataset(cfg)
    cat_data = generate_catalyst_deactivation(cfg)

    paths["tubes"] = os.path.join(out_dir, "synthetic_tubes.csv")
    paths["burners"] = os.path.join(out_dir, "synthetic_burner_profiles.csv")
    paths["timeseries"] = os.path.join(out_dir, "synthetic_timeseries.csv")
    paths["temp_map"] = os.path.join(out_dir, "synthetic_temp_map.csv")
    paths["rul"] = os.path.join(out_dir, "synthetic_rul.csv")
    paths["sc_data"] = os.path.join(out_dir, "synthetic_sc_opt.csv")
    paths["burner_data"] = os.path.join(out_dir, "synthetic_burner_status.csv")
    paths["cat_data"] = os.path.join(out_dir, "synthetic_catalyst.csv")

    tubes.to_csv(paths["tubes"], index=False)
    burners.to_csv(paths["burners"], index=False)
    ts.to_csv(paths["timeseries"], index=False)
    # temp_map can be large; downsample if needed
    temp_map.to_csv(paths["temp_map"], index=False)
    rul.to_csv(paths["rul"], index=False)
    sc_data.to_csv(paths["sc_data"], index=False)
    burner_data.to_csv(paths["burner_data"], index=False)
    cat_data.to_csv(paths["cat_data"], index=False)

    print(f"✓ synthetic tubes: {len(tubes)} rows")
    print(f"✓ synthetic burners: {len(burners)} rows")
    print(f"✓ synthetic time-series: {len(ts)} rows")
    print(f"✓ synthetic temperature map: {len(temp_map)} rows")
    print(f"✓ synthetic RUL labels: {len(rul)} rows")
    print(f"✓ synthetic S/C dataset: {len(sc_data)} rows")
    print(f"✓ synthetic burner status: {len(burner_data)} rows")
    print(f"✓ synthetic catalyst: {len(cat_data)} rows")
    return paths


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "/home/z/my-project/download/mrto/data/raw"
    generate_all(out)
