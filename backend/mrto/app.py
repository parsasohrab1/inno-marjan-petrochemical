"""FastAPI backend for MRTO system.

Endpoints (see SRS Appendix 8-1):
    GET  /api/v1/health
    POST /api/v1/auth/login
    POST /api/v1/auth/refresh
    POST /api/v1/auth/logout
    GET  /api/v1/tubes
    GET  /api/v1/temperature-map
    GET  /api/v1/rul
    POST /api/v1/optimization
    GET  /api/v1/burners
    GET  /api/v1/alerts
    POST /api/v1/alerts/{id}/ack
    GET  /api/v1/reports/daily
    GET  /api/v1/metrics  (model quality metrics)
    GET  /api/v1/models/versions            (model versioning)
    GET  /api/v1/models/{module}/active
    POST /api/v1/ingest/temperature  (admin ingestion)
"""
from __future__ import annotations

import os
import sys
import json
import time
import xml.etree.ElementTree as ET
import base64
import hmac
import hashlib
import logging
import secrets
import threading
from datetime import datetime, timezone

logger = logging.getLogger("mrto")
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

# Allow running both as module and as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrto import db as dbm
from mrto.market import build_plan, fetch_market
from mrto import models as mlm
from mrto.ratelimit import RateLimiter
from mrto import observability as obs

# Phase 3: read-replica routing. When MRTO_READ_REPLICA_URL is set (Postgres
# deployment), GET endpoints open the connection through db_adapter so
# reads go to the replica and the primary is spared for writes. On SQLite
# (dev/test) the replica flag is ignored and the same DB is used.
try:
    from mrto import db_adapter as _db_adapter
    _HAS_REPLICA = _db_adapter.HAS_REPLICA
except Exception:
    _HAS_REPLICA = False


def _read_conn(app, role: str = "replica"):
    """Return a context manager for a read connection. Routes to the
    replica when one is configured; otherwise falls back to the primary
    (SQLite or single-postgres)."""
    if _HAS_REPLICA:
        return _db_adapter.get_conn(app.state.db_path, role=role)
    return dbm.get_conn(app.state.db_path)

def _resolve_models_dir() -> str:
    """Prefer MRTO_MODELS_DIR when that folder exists.

    Fall back to the repository models/ directory so a Windows launch works
    without the old hardcoded Linux path.
    """
    env = os.environ.get("MRTO_MODELS_DIR")
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
    if env and os.path.isdir(env):
        return env
    if os.path.isdir(repo):
        return repo
    return env or repo


MODELS_DIR = _resolve_models_dir()
METRICS_DIR = MODELS_DIR  # metrics json saved alongside models

# Cache loaded models in-memory to avoid joblib.load on every request.
_MODEL_CACHE: dict[str, object] = {}


def _get_model(name: str):
    """Load and cache a joblib model; returns None if file missing."""
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        return None
    if name not in _MODEL_CACHE:
        try:
            _MODEL_CACHE[name] = mlm.load_model(path)
        except Exception:
            return None
    return _MODEL_CACHE[name]


_HOT_TUBES = ("T01-003", "T02-003")
_DEGRADED_BURNERS = {"B01-L": 0.52, "B03-R": 0.40}
# Bed conditions fed to the Module 4 activity model. Replacement is the
# training label: activity below 0.4.
_CATALYST_BED = {"age_h": 8000.0, "sc_avg": 3.0, "dp_bar": 1.6}
_CATALYST_REPLACE_BELOW = 0.4


def _catalyst_status(activity: float | None) -> str:
    if activity is None:
        return "unknown"
    if activity < _CATALYST_REPLACE_BELOW:
        return "replace"
    if activity < 0.75:
        return "aging"
    return "healthy"


def _light_from_pfail(p_fail: float) -> str:
    """Traffic light aligned with the dashboard risk bands."""
    if p_fail > 0.5:
        return "red"
    if p_fail > 0.2:
        return "orange"
    return "green"


def _catalyst_light(status: str) -> str:
    if status == "replace":
        return "red"
    if status == "aging":
        return "orange"
    if status == "healthy":
        return "green"
    return "grey"


def _temp_light(temp_c: float | None) -> str:
    if temp_c is None:
        return "grey"
    if temp_c >= 930:
        return "red"
    if temp_c >= 880:
        return "orange"
    return "green"


def _worst_light(lights: list[str]) -> str:
    rank = {"red": 0, "orange": 1, "grey": 2, "green": 3}
    if not lights:
        return "grey"
    return min(lights, key=lambda name: rank.get(name, 9))


def _check_status(abs_residual: float, warn_at: float, fail_at: float) -> str:
    if abs_residual > fail_at:
        return "fail"
    if abs_residual > warn_at:
        return "warn"
    return "pass"


def _compute_rul(db_path: str) -> dict:
    """RUL per tube, with a traffic light on each row."""
    model = _get_model("module2_rul.joblib")
    with dbm.get_conn(db_path) as conn:
        tubes = dbm.get_tubes(conn)
        avg_t_map = _avg_wall_temp_per_tube(conn, [t["tube_id"] for t in tubes]) if tubes else {}
    payload: dict | None = None
    if model is not None and tubes:
        try:
            df = pd.DataFrame(tubes)
            df["avg_wall_temp_c"] = df["tube_id"].map(avg_t_map).astype(float)
            df["consumed_life_fraction"] = (df["last_replaced_days_ago"] * 24) / df["design_life_h"]
            rul = model.predict_rul(df)
            pfail = model.predict_failure_prob(df)
            payload = {
                "source": "trained_model",
                "tubes": [
                    {"tube_id": tid, "rul_h": int(r), "p_fail_90d": round(float(p), 4)}
                    for tid, r, p in zip(df["tube_id"], rul, pfail)
                ],
            }
        except Exception:
            logger.exception("rul model failed")
            payload = None
    if payload is None:
        from mrto.synthetic_data import lmp_to_rupture_time
        rows = []
        for t in tubes:
            avg_t = avg_t_map.get(t["tube_id"], 880.0)
            t_rupt = lmp_to_rupture_time(avg_t)
            elapsed = t["last_replaced_days_ago"] * 24
            rul_h = max(0.0, t_rupt - elapsed)
            consumed = elapsed / max(t_rupt, 1)
            p_fail_90d = 1.0 / (1.0 + (1.0 - max(0, min(1, consumed - 0.7))) * 10)
            rows.append({
                "tube_id": t["tube_id"],
                "rul_h": int(rul_h),
                "p_fail_90d": round(p_fail_90d, 4),
            })
        payload = {"source": "physics_fallback", "tubes": rows}
    for row in payload["tubes"]:
        row["light"] = _light_from_pfail(float(row["p_fail_90d"]))
    return payload


def _furnace_burners(conn) -> dict:
    """Latest burner row per burner, plus the catalyst bed the model sees."""
    rows = conn.execute(
        """
        SELECT burner_id, ts_minute, fuel_flow, intensity, angle
        FROM burner_status
        WHERE id IN (
            SELECT MAX(id) FROM burner_status GROUP BY burner_id
        )
        ORDER BY burner_id
        """
    ).fetchall()
    overall_avg = conn.execute(
        """
        SELECT AVG(wall_temp_c) FROM temperature_readings
        WHERE id IN (SELECT MAX(id) FROM temperature_readings GROUP BY tube_id)
        """
    ).fetchone()[0]
    burners_list = [dict(r) for r in rows]
    mean_wall = float(overall_avg) if overall_avg is not None else 880.0
    catalyst_activity = None
    model = _get_model("module4_burner_catalyst.joblib")
    if model is not None:
        try:
            if burners_list:
                df = pd.DataFrame(burners_list).rename(
                    columns={"fuel_flow": "fuel_flow_nm3h", "angle": "angle_deg"}
                )
                df["age_h"] = 2000
                df["avg_tube_t_c"] = mean_wall
                df["intensity"] = df["intensity"].fillna(0.9)
                df["fuel_flow_nm3h"] = df["fuel_flow_nm3h"].fillna(190)
                df["angle_deg"] = df["angle_deg"].fillna(0)
                preds = model.predict_inefficient(df)
                for burner, pred in zip(burners_list, preds):
                    burner["inefficient"] = int(pred)
            activity = model.predict_activity(pd.DataFrame([{
                "age_h": _CATALYST_BED["age_h"],
                "t_out_avg_c": mean_wall,
                "sc_avg": _CATALYST_BED["sc_avg"],
                "dp_bar": _CATALYST_BED["dp_bar"],
            }]))
            catalyst_activity = round(float(activity[0]), 4)
        except Exception:
            logger.exception("burner/catalyst model failed")
    for burner in burners_list:
        burner["light"] = "red" if burner.get("inefficient") else "green"
    catalyst = {
        "activity": catalyst_activity,
        "status": _catalyst_status(catalyst_activity),
        "age_h": _CATALYST_BED["age_h"],
        "sc_avg": _CATALYST_BED["sc_avg"],
        "dp_bar": _CATALYST_BED["dp_bar"],
        "t_out_avg_c": round(mean_wall, 2),
        "replace_below": _CATALYST_REPLACE_BELOW,
        "light": _catalyst_light(_catalyst_status(catalyst_activity)),
    }
    return {
        "burners": burners_list,
        "n": len(burners_list),
        "catalyst_activity": catalyst_activity,
        "mean_wall_temp_c": round(mean_wall, 2),
        "catalyst": catalyst,
    }


def _pm_plan(kind: str, light: str) -> tuple[str, int | None, str]:
    plans = {
        ("tube", "red"): ("بازرسی فوری دیواره و برنامه‌ریزی تعویض", 7, "احتمال خرابی ۹۰روزه بالا است"),
        ("tube", "orange"): ("بازرسی در توقف برنامه‌ریزی‌شده", 30, "احتمال خرابی در محدوده متوسط است"),
        ("tube", "green"): ("بازرسی دوره‌ای", 90, "لوله در محدوده سالم است"),
        ("burner", "red"): ("تنظیم شعله یا تعویض نازل", 3, "مدل مشعل را ناکارآمد علامت زده است"),
        ("burner", "green"): ("بازرسی دوره‌ای شعله", 90, "مشعل در محدوده نرمال است"),
        ("catalyst", "red"): ("برنامه‌ریزی تعویض بستر", 14, "فعالیت زیر آستانه تعویض است"),
        ("catalyst", "orange"): ("نمونه‌گیری فعالیت و پایش افت فشار", 45, "فعالیت افت کرده ولی هنوز به تعویض نرسیده"),
        ("catalyst", "green"): ("پایش دوره‌ای بستر", 90, "بستر در محدوده سالم است"),
        ("catalyst", "grey"): ("داده مدل کاتالیست ناقص است", None, "پیش‌بینی فعالیت در دسترس نیست"),
    }
    return plans.get((kind, light), ("پایش", None, ""))


def _maintenance_view(db_path: str) -> dict:
    rul = _compute_rul(db_path)
    with dbm.get_conn(db_path) as conn:
        furnace = _furnace_burners(conn)
    assets = []
    for tube in rul["tubes"]:
        action, due, reason = _pm_plan("tube", tube["light"])
        assets.append({
            "asset_id": tube["tube_id"],
            "kind": "tube",
            "light": tube["light"],
            "action": action,
            "due_days": due,
            "reason": reason,
            "rul_h": tube["rul_h"],
            "p_fail_90d": tube["p_fail_90d"],
        })
    for burner in furnace["burners"]:
        action, due, reason = _pm_plan("burner", burner["light"])
        assets.append({
            "asset_id": burner["burner_id"],
            "kind": "burner",
            "light": burner["light"],
            "action": action,
            "due_days": due,
            "reason": reason,
            "intensity": burner.get("intensity"),
            "fuel_flow": burner.get("fuel_flow"),
        })
    cat = furnace["catalyst"]
    action, due, reason = _pm_plan("catalyst", cat["light"])
    assets.append({
        "asset_id": "CAT-BED",
        "kind": "catalyst",
        "light": cat["light"],
        "action": action,
        "due_days": due,
        "reason": reason,
        "activity": cat["activity"],
    })
    order = {"red": 0, "orange": 1, "grey": 2, "green": 3}
    assets.sort(key=lambda item: (order.get(item["light"], 9), item["kind"], item["asset_id"]))
    counts = {"red": 0, "orange": 0, "green": 0, "grey": 0}
    for asset in assets:
        counts[asset["light"]] = counts.get(asset["light"], 0) + 1
    bounds = [(0, 20000, "کمتر از ۲۰٬۰۰۰", "red"),
              (20000, 40000, "۲۰ تا ۴۰ هزار", "orange"),
              (40000, 70000, "۴۰ تا ۷۰ هزار", "green"),
              (70000, 10**12, "بیش از ۷۰ هزار", "green")]
    buckets = []
    for low, high, label, light in bounds:
        n = sum(1 for tube in rul["tubes"] if low <= tube["rul_h"] < high)
        buckets.append({"label": label, "n": n, "light": light})
    return {
        "source": rul["source"],
        "counts": counts,
        "assets": assets,
        "rul_buckets": buckets,
    }


def _hotspot_rows(readings: list[dict]) -> tuple[float, list[dict]]:
    temps = [float(r["wall_temp_c"]) for r in readings]
    if not temps:
        return 0.0, []
    mean_t = sum(temps) / len(temps)
    hot = [
        {"tube_id": r["tube_id"], "wall_temp_c": r["wall_temp_c"],
         "delta_c": round(float(r["wall_temp_c"]) - mean_t, 2)}
        for r in readings if float(r["wall_temp_c"]) - mean_t > 30
    ]
    return mean_t, hot


def _digital_twin(db_path: str) -> dict:
    """Live process twin: measured furnace state against the nominal model."""
    nominal_wall = 880.0
    with dbm.get_conn(db_path) as conn:
        readings = dbm.get_temperature_map(conn)
        furnace = _furnace_burners(conn)
        n_tubes = conn.execute("SELECT COUNT(*) FROM tubes").fetchone()[0]
    mean_t, hotspots = _hotspot_rows(readings)
    hot_ids = {h["tube_id"] for h in hotspots}
    tube_nodes = [
        {
            "tube_id": r["tube_id"],
            "wall_temp_c": r["wall_temp_c"],
            "light": "red" if r["tube_id"] in hot_ids else _temp_light(float(r["wall_temp_c"])),
        }
        for r in readings
    ]
    tube_light = _worst_light([t["light"] for t in tube_nodes]) if tube_nodes else "grey"
    burner_light = _worst_light([b["light"] for b in furnace["burners"]]) if furnace["burners"] else "grey"
    cat = furnace["catalyst"]
    primary_light = _worst_light([tube_light, burner_light, cat["light"]])
    inefficient = sum(1 for b in furnace["burners"] if b.get("inefficient"))
    nodes = [
        {
            "id": "feed",
            "name": "گاز طبیعی",
            "subtitle": "از پالایشگاه پارس جنوبی",
            "light": "grey",
            "inlet": "گاز فازهای پارس جنوبی در عسلویه",
            "outlet": "گاز به کمپرسور",
            "duty": "تأمین خوراک مجتمع",
            "note": "سنسور دبی خوراک به تاریخ‌نگار وصل نیست",
            "kpis": [],
        },
        {
            "id": "compress",
            "name": "کمپرسور گاز",
            "group": "pretreat",
            "light": "grey",
            "inlet": "گاز طبیعی کم‌فشار",
            "outlet": "گاز فشرده",
            "duty": "بالا بردن فشار گاز تا حد گوگردزدایی و ریفرمر",
            "note": "بدون قرائت زنده",
            "kpis": [],
        },
        {
            "id": "hds",
            "name": "گوگردزدایی",
            "group": "pretreat",
            "light": "grey",
            "inlet": "گاز فشرده و هیدروژن بازگردانی",
            "outlet": "گاز شیرین",
            "duty": "گرفتن گوگرد تا کاتالیست نیکل مسموم نشود",
            "note": "بدون قرائت زنده",
            "kpis": [],
        },
        {
            "id": "saturator",
            "name": "اشباع با بخار",
            "group": "pretreat",
            "light": "green",
            "inlet": "گاز شیرین و بخار",
            "outlet": "مخلوط گاز و بخار",
            "duty": "تنظیم نسبت بخار به کربن پیش از ریفرمر",
            "note": "نسبت بخار به کربنِ مدل، نه سنسور میدانی",
            "kpis": [{"label": "S/C", "value": cat["sc_avg"], "unit": ""}],
        },
        {
            "id": "primary",
            "name": "ریفرمر اولیه",
            "subtitle": "لوله‌های کاتالیستی نیکل، گرمایش خارجی با مشعل",
            "reaction": "CH₄ + H₂O → CO + 3H₂",
            "light": primary_light,
            "inlet": "مخلوط گاز طبیعی و بخار، به‌اضافه سوخت مشعل",
            "outlet": "گاز تبدیل‌شده داغ",
            "duty": "تبدیل متان با بخار روی کاتالیست نیکل داخل لوله",
            "note": "فقط همین واحد دما، مشعل و فعالیت کاتالیست زنده دارد",
            "kpis": [
                {"label": "میانگین دما", "value": round(mean_t, 1) if readings else None, "unit": "°C"},
                {"label": "لوله‌ها", "value": n_tubes, "unit": ""},
                {"label": "نقاط داغ", "value": len(hotspots), "unit": ""},
                {"label": "فعالیت کاتالیست", "value": None if cat["activity"] is None else round(cat["activity"] * 100, 1), "unit": "%"},
                {"label": "مشعل ناکارآمد", "value": inefficient, "unit": ""},
            ],
            "tubes": tube_nodes,
            "burners": [
                {"burner_id": b["burner_id"], "intensity": b.get("intensity"), "light": b["light"]}
                for b in furnace["burners"]
            ],
        },
        {
            "id": "asu",
            "name": "جداسازی هوا",
            "light": "grey",
            "inlet": "هوای محیط",
            "outlet": "اکسیژن به ریفرمر ثانویه",
            "duty": "تأمین اکسیژن احتراق داخلی اتوترمال",
            "note": "سنسور اکسیژن وصل نیست",
            "kpis": [],
        },
        {
            "id": "secondary",
            "name": "ریفرمر ثانویه",
            "subtitle": "اتوترمال",
            "light": "grey",
            "inlet": "گاز ریفرمر اولیه و اکسیژن",
            "outlet": "گاز سنتز",
            "duty": "تکمیل تبدیل متان و تنظیم نسبت هیدروژن",
            "note": "آنالایزر گاز سنتز وصل نیست",
            "kpis": [],
        },
        {
            "id": "synthesis",
            "name": "سنتز متانول",
            "subtitle": "حلقه سنتز",
            "reaction": "CO + 2H₂ → CH₃OH",
            "light": "grey",
            "inlet": "گاز سنتز فشرده",
            "outlet": "متانول خام",
            "duty": "ساخت متانول روی کاتالیست مس-روی",
            "note": "دبی محصول به توزین وصل نیست",
            "kpis": [],
        },
        {
            "id": "distill",
            "name": "برج تقطیر",
            "light": "grey",
            "inlet": "متانول خام",
            "outlet": "متانول خالص",
            "duty": "جدا کردن آب و ناخالصی از متانول صادراتی",
            "note": "آنالایزر خلوص وصل نیست",
            "kpis": [],
        },
        {
            "id": "storage",
            "name": "مخزن محصول",
            "light": "grey",
            "inlet": "متانول خالص",
            "outlet": "بارگیری صادراتی",
            "duty": "نگهداری متانول تا صادرات",
            "note": "سطح مخزن وصل نیست",
            "kpis": [],
        },
    ]
    residuals = []
    if readings:
        residuals.append({
            "name": "دمای دیواره",
            "measured": round(mean_t, 2),
            "twin": nominal_wall,
            "residual": round(mean_t - nominal_wall, 2),
            "unit": "°C",
        })
        residuals.append({
            "name": "نقاط داغ",
            "measured": len(hotspots),
            "twin": 0,
            "residual": len(hotspots),
            "unit": "",
        })
    if furnace["burners"]:
        residuals.append({
            "name": "مشعل ناکارآمد",
            "measured": inefficient,
            "twin": 0,
            "residual": inefficient,
            "unit": "",
        })
    if cat["activity"] is not None:
        residuals.append({
            "name": "فعالیت نسبت به بستر نو",
            "measured": cat["activity"],
            "twin": 1.0,
            "residual": round(cat["activity"] - 1.0, 4),
            "unit": "",
        })
    return {"nodes": nodes, "residuals": residuals, "nominal_wall_c": nominal_wall}


def _recon_check(check_id: str, name: str, status: str, measured, reference, unit: str) -> dict:
    residual = None
    if isinstance(measured, (int, float)) and isinstance(reference, (int, float)):
        residual = round(float(measured) - float(reference), 4)
    return {
        "id": check_id,
        "name": name,
        "status": status,
        "measured": measured,
        "reference": reference,
        "residual": residual,
        "unit": unit,
    }


def _reconciliation(db_path: str) -> dict:
    """Range, completeness, and balance checks on the live furnace sample."""
    with dbm.get_conn(db_path) as conn:
        tubes = dbm.get_tubes(conn)
        readings = dbm.get_temperature_map(conn)
        furnace = _furnace_burners(conn)
        open_hot = conn.execute(
            "SELECT COUNT(*) FROM alerts WHERE status='open' AND module='module1'"
        ).fetchone()[0]
        dup = conn.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT tube_id FROM temperature_readings
                WHERE ts_minute = (SELECT MAX(ts_minute) FROM temperature_readings)
                GROUP BY tube_id HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
    mean_t, hotspots = _hotspot_rows(readings)
    temps = [float(r["wall_temp_c"]) for r in readings]
    n_tubes = len(tubes)
    n_read = len(readings)
    checks = []
    if n_tubes == 0:
        coverage = "fail"
    elif n_read == n_tubes:
        coverage = "pass"
    elif n_read == 0:
        coverage = "fail"
    else:
        coverage = "warn"
    checks.append(_recon_check(
        "coverage", "کامل بودن نقشه دما", coverage, n_read, n_tubes, "لوله",
    ))
    if not temps:
        span = "fail"
        range_measured = None
    else:
        outside = sum(1 for t in temps if t < 700 or t > 1100)
        hot_band = sum(1 for t in temps if t < 800 or t > 980)
        if outside:
            span = "fail"
            range_measured = outside
        elif hot_band:
            span = "warn"
            range_measured = hot_band
        else:
            span = "pass"
            range_measured = 0
    checks.append(_recon_check(
        "temp_range", "محدوده فیزیکی دما", span, range_measured, 0, "لوله خارج از بازه",
    ))
    if temps:
        balance = _check_status(abs(mean_t - 880.0), 40, 80)
        spread = (sum((t - mean_t) ** 2 for t in temps) / len(temps)) ** 0.5
        spread_status = _check_status(spread, 25, 60)
    else:
        balance = "fail"
        spread = None
        spread_status = "fail"
    checks.append(_recon_check(
        "temp_balance", "تراز دما با مدل نامی ۸۸۰°C", balance,
        round(mean_t, 2) if temps else None, 880.0, "°C",
    ))
    checks.append(_recon_check(
        "temp_spread", "پراکندگی دمای لوله‌ها", spread_status,
        None if spread is None else round(spread, 2), 25.0, "°C",
    ))
    alert_gap = abs(int(open_hot) - len(hotspots))
    alert_status = "pass" if alert_gap == 0 else ("warn" if alert_gap == 1 else "fail")
    checks.append(_recon_check(
        "hotspot_alerts", "تطابق هشدار نقطه داغ", alert_status, int(open_hot), len(hotspots), "مورد",
    ))
    checks.append(_recon_check(
        "duplicates", "یکتایی قرائت آخر هر لوله", "pass" if dup == 0 else "fail", int(dup), 0, "لوله تکراری",
    ))
    fuels = [b.get("fuel_flow") for b in furnace["burners"]]
    if not fuels:
        fuel_status = "fail"
        fuel_bad = None
    else:
        fuel_bad = sum(1 for f in fuels if f is None or not (120 <= float(f) <= 280))
        fuel_status = "pass" if fuel_bad == 0 else "fail"
    checks.append(_recon_check(
        "fuel_balance", "دبی سوخت در بازه آموزش مدل", fuel_status,
        0 if fuel_bad is None else fuel_bad, 0, "مشعل خارج از بازه",
    ))
    activity = furnace["catalyst"]["activity"]
    if activity is None:
        cat_status = "fail"
    elif 0.05 <= activity <= 1.05:
        cat_status = "pass"
    else:
        cat_status = "fail"
    checks.append(_recon_check(
        "catalyst_range", "فعالیت کاتالیست در بازه فیزیکی", cat_status, activity, 1.0, "",
    ))
    weight = {"pass": 1.0, "warn": 0.5, "fail": 0.0}
    score = round(sum(weight[c["status"]] for c in checks) / len(checks), 3)
    return {
        "score": score,
        "n_pass": sum(1 for c in checks if c["status"] == "pass"),
        "n_warn": sum(1 for c in checks if c["status"] == "warn"),
        "n_fail": sum(1 for c in checks if c["status"] == "fail"),
        "checks": checks,
    }


# Stoichiometric CO2 from methane: 44/16 × 0.717 kg/Nm³.
_KG_CO2_PER_NM3_CH4 = 1.972
_FURNACE_FEED_NM3H = 32000.0
_FURNACE_FUEL_BASE_NM3H = 11500.0
_FURNACE_SC = 3.2
# Ideal CH4 → CH3OH: 32 kg methanol per 22.414 Nm³ methane.
_KG_METHANOL_PER_NM3 = 32.0 / 22.414
# Methane lower heating value used to turn Nm³ into GJ.
_LHV_GJ_PER_NM3 = 0.0358
# Johnson Matthey / AFPM: about 1,700°F (927°C) for a 100,000 h tube life.
_TWT_DESIGN_C = 927.0
_TWT_WATCH_C = 912.0
_carbon_opt_cache: dict = {}


def _cached_furnace_opt(db_path: str, current: dict) -> dict | None:
    """Run Module 3 on a coarse grid and reuse the result for a short window."""
    key = (
        db_path,
        round(float(current["t_out_c"]), 0),
        round(float(current["cat_age"]), 2),
        round(float(current["fuel_flow_nm3h"]), 0),
    )
    now = time.time()
    hit = _carbon_opt_cache.get(key)
    if hit and now - hit[0] < 20:
        return hit[1]
    model = _get_model("module3_optimization.joblib")
    if model is None:
        result = None
    else:
        try:
            result = model.optimize(current, alpha_eff=0.7, alpha_fuel=0.3, n_grid=12)
        except Exception:
            logger.exception("furnace carbon optimization failed")
            result = None
    if len(_carbon_opt_cache) > 6:
        _carbon_opt_cache.clear()
    _carbon_opt_cache[key] = (now, result)
    return result


def _carbon_view(db_path: str) -> dict:
    """Fuel and CO2 from live burner flow, plus the furnace optimum."""
    with dbm.get_conn(db_path) as conn:
        furnace = _furnace_burners(conn)
        hist_rows = conn.execute(
            """
            SELECT ts_minute, SUM(fuel_flow) AS fuel_nm3h
            FROM burner_status
            GROUP BY ts_minute
            ORDER BY ts_minute DESC
            LIMIT 36
            """
        ).fetchall()
    history = []
    for row in reversed(hist_rows):
        fuel = float(row["fuel_nm3h"] or 0)
        history.append({
            "ts_minute": int(row["ts_minute"]),
            "fuel_nm3h": round(fuel, 1),
            "co2_kg_h": round(fuel * _KG_CO2_PER_NM3_CH4, 1),
        })
    burners = furnace["burners"]
    healthy = sorted(
        float(b["fuel_flow"]) for b in burners
        if not b.get("inefficient") and b.get("fuel_flow") is not None
    )
    reference = healthy[len(healthy) // 2] if healthy else None
    measured = 0.0
    excess = 0.0
    shares = []
    for burner in burners:
        fuel = float(burner.get("fuel_flow") or 0)
        measured += fuel
        extra = 0.0
        if reference is not None and burner.get("inefficient"):
            extra = max(0.0, fuel - reference)
        excess += extra
        shares.append({
            "burner_id": burner["burner_id"],
            "fuel_nm3h": round(fuel, 1),
            "co2_kg_h": round(fuel * _KG_CO2_PER_NM3_CH4, 1),
            "excess_fuel_nm3h": round(extra, 1),
            "light": burner.get("light", "green"),
        })
    t_out = min(950.0, max(820.0, float(furnace["mean_wall_temp_c"] or 880.0)))
    cat_age = min(1.0, float(furnace["catalyst"]["age_h"]) / 30000.0)
    current = {
        "sc_ratio": _FURNACE_SC,
        "t_out_c": round(t_out, 1),
        "feed_flow_nm3h": _FURNACE_FEED_NM3H,
        "cat_age": round(cat_age, 3),
        "fuel_flow_nm3h": _FURNACE_FUEL_BASE_NM3H,
    }
    opt = _cached_furnace_opt(db_path, current) or {}
    opt_fuel = opt.get("fuel_pred_nm3h")
    current_eff = None
    current_coke = None
    model = _get_model("module3_optimization.joblib")
    if model is not None:
        try:
            feat = pd.DataFrame([current])
            current_eff = float(model.predict_efficiency(feat)[0])
            current_coke = float(model.predict_coking_risk(feat)[0])
        except Exception:
            logger.exception("current furnace efficiency prediction failed")
    opt_eff = opt.get("efficiency_pred")
    methanol_now = None if current_eff is None else (
        _FURNACE_FEED_NM3H * (_KG_METHANOL_PER_NM3 / 1000.0) * current_eff
    )
    methanol_opt = None if opt_eff is None else (
        _FURNACE_FEED_NM3H * (_KG_METHANOL_PER_NM3 / 1000.0) * float(opt_eff)
    )

    def _sec(fuel_nm3h: float, methanol_t_h: float | None) -> float | None:
        if not methanol_t_h:
            return None
        return (_FURNACE_FEED_NM3H + fuel_nm3h) * _LHV_GJ_PER_NM3 / methanol_t_h

    sec_now = _sec(_FURNACE_FUEL_BASE_NM3H, methanol_now)
    sec_opt = None if opt_fuel is None else _sec(float(opt_fuel), methanol_opt)
    base_co2 = _FURNACE_FUEL_BASE_NM3H * _KG_CO2_PER_NM3_CH4
    opt_co2 = None if opt_fuel is None else float(opt_fuel) * _KG_CO2_PER_NM3_CH4
    intensity_now = None if not methanol_now else (base_co2 / 1000.0) / methanol_now
    intensity_opt = None if opt_co2 is None or not methanol_opt else (opt_co2 / 1000.0) / methanol_opt
    co2_measured = measured * _KG_CO2_PER_NM3_CH4
    avoidable = excess * _KG_CO2_PER_NM3_CH4
    if avoidable > 50:
        light = "orange"
    elif opt.get("coking_risk_pred", 0) > 0.5:
        light = "red"
    else:
        light = "green"
    return {
        "emission_factor_kg_per_nm3": _KG_CO2_PER_NM3_CH4,
        "light": light,
        "burner_fuel_nm3h": round(measured, 1),
        "combustion_co2_kg_h": round(co2_measured, 1),
        "avoidable_fuel_nm3h": round(excess, 1),
        "avoidable_co2_kg_h": round(avoidable, 1),
        "healthy_fuel_nm3h": None if reference is None else round(reference, 1),
        "inefficient_burners": sum(1 for b in burners if b.get("inefficient")),
        "combustion_quality": None if not burners else round(1 - sum(1 for b in burners if b.get("inefficient")) / len(burners), 3),
        "burners": shares,
        "history": history,
        "furnace": {
            "baseline_fuel_nm3h": _FURNACE_FUEL_BASE_NM3H,
            "baseline_co2_kg_h": round(base_co2, 1),
            "feed_nm3h": _FURNACE_FEED_NM3H,
            "sc_ratio": current["sc_ratio"],
            "t_out_c": current["t_out_c"],
            "cat_age": current["cat_age"],
            "fuel_opt_nm3h": None if opt_fuel is None else round(float(opt_fuel), 1),
            "co2_opt_kg_h": None if opt_co2 is None else round(opt_co2, 1),
            "co2_saving_kg_h": None if opt_co2 is None else round(base_co2 - opt_co2, 1),
            "fuel_saving_pct": None if opt.get("fuel_saving_pct") is None else round(float(opt["fuel_saving_pct"]), 2),
            "efficiency_pred": None if opt.get("efficiency_pred") is None else round(float(opt["efficiency_pred"]), 4),
            "coking_risk_pred": None if opt.get("coking_risk_pred") is None else round(float(opt["coking_risk_pred"]), 4),
            "sc_ratio_opt": opt.get("sc_ratio_opt"),
            "t_out_c_opt": None if opt.get("t_out_c_opt") is None else round(float(opt["t_out_c_opt"]), 1),
            "carbon_intensity_kg_per_1000nm3": round(base_co2 / _FURNACE_FEED_NM3H * 1000, 2),
            "carbon_intensity_opt_kg_per_1000nm3": None if opt_co2 is None else round(opt_co2 / _FURNACE_FEED_NM3H * 1000, 2),
            "efficiency_now": None if current_eff is None else round(current_eff, 4),
            "coking_risk_now": None if current_coke is None else round(current_coke, 4),
            "methanol_t_h": None if methanol_now is None else round(methanol_now, 2),
            "methanol_opt_t_h": None if methanol_opt is None else round(methanol_opt, 2),
            "sec_gj_per_t": None if sec_now is None else round(sec_now, 2),
            "sec_opt_gj_per_t": None if sec_opt is None else round(sec_opt, 2),
            "co2_t_per_t": None if intensity_now is None else round(intensity_now, 3),
            "co2_opt_t_per_t": None if intensity_opt is None else round(intensity_opt, 3),
            "lhv_gj_per_nm3": _LHV_GJ_PER_NM3,
        },
    }


def _bench_item(item_id: str, name: str, source: str, status: str,
                measured, reference, unit: str, gap) -> dict:
    return {
        "id": item_id,
        "name": name,
        "source": source,
        "status": status,
        "measured": None if measured is None else round(float(measured), 3),
        "reference": reference,
        "unit": unit,
        "gap": None if gap is None else round(float(gap), 3),
    }


def _benchmark_view(db_path: str) -> dict:
    """Compare the live furnace with published methanol and reformer practice."""
    carbon = _carbon_view(db_path)
    furnace = carbon["furnace"]
    with dbm.get_conn(db_path) as conn:
        readings = dbm.get_temperature_map(conn)
    temps = [float(r["wall_temp_c"]) for r in readings]
    max_t = max(temps) if temps else None
    mean_t = sum(temps) / len(temps) if temps else None
    spread = None if max_t is None else max_t - mean_t
    above_design = sum(1 for t in temps if t > _TWT_DESIGN_C)
    if max_t is None:
        twt_status = "fail"
    elif max_t > _TWT_DESIGN_C:
        twt_status = "fail"
    elif max_t > _TWT_WATCH_C:
        twt_status = "warn"
    else:
        twt_status = "pass"
    if spread is None:
        balance_status = "fail"
    elif spread > 50:
        balance_status = "fail"
    elif spread > 30:
        balance_status = "warn"
    else:
        balance_status = "pass"
    intensity = furnace.get("co2_t_per_t")
    if intensity is None:
        co2_status = "fail"
    elif intensity > 0.67:
        co2_status = "fail"
    elif intensity > 0.497:
        co2_status = "warn"
    else:
        co2_status = "pass"
    sec = furnace.get("sec_gj_per_t")
    if sec is None:
        sec_status = "fail"
    elif sec > 36:
        sec_status = "fail"
    elif sec > 30:
        sec_status = "warn"
    else:
        sec_status = "pass"
    coke = furnace.get("coking_risk_now")
    coke_status = "fail" if coke is None or coke > 0.5 else "pass"
    checks = [
        _bench_item(
            "twt_design", "حداکثر دمای دیواره در برابر عمر ۱۰۰٬۰۰۰ ساعت",
            "Johnson Matthey / AFPM: حدود ۹۲۷°C برای عمر طراحی ۱۰۰٬۰۰۰ ساعت",
            twt_status, max_t, _TWT_DESIGN_C, "°C",
            None if max_t is None else max_t - _TWT_DESIGN_C,
        ),
        _bench_item(
            "temp_balance", "فاصله گرم‌ترین لوله از میانگین",
            "نقطه داغ MRTO بالای ۳۰°C؛ اختلاف بزرگ‌تر از ۵۰°C خارج از توازن کوره است",
            balance_status, spread, 30.0, "°C",
            None if spread is None else spread - 30.0,
        ),
        _bench_item(
            "co2_intensity", "شدت CO₂ کوره به ازای تن متانول",
            "IPCC 2006، جدول ۳.۱۲: ۰٫۴۹۷ تن CO₂ بر تن برای ریفرمینگ بخار با ریفرمر اولیه و خوراک گاز طبیعی",
            co2_status, intensity, 0.497, "t/t",
            None if intensity is None else intensity - 0.497,
        ),
        _bench_item(
            "specific_energy", "انرژی گاز طبیعی به ازای تن متانول",
            "نمونه معمول حدود ۳۰ GJ/t و نزدیک بهترین فناوری حدود ۲۰ GJ/t؛ جانسون متی برای SMR حدود ۳۲٫۶ GJ/t",
            sec_status, sec, 30.0, "GJ/t",
            None if sec is None else sec - 30.0,
        ),
        _bench_item(
            "coking", "ریسک رسوب کربن در نقطه کار فعلی",
            "قید مدل ماژول ۳ و رویه بهره‌برداری: ریسک کک زیر ۰٫۵",
            coke_status, coke, 0.5, "",
            None if coke is None else coke - 0.5,
        ),
    ]
    weight = {"pass": 1.0, "warn": 0.5, "fail": 0.0}
    score = round(sum(weight[c["status"]] for c in checks) / len(checks), 3)
    return {
        "score": score,
        "n_pass": sum(1 for c in checks if c["status"] == "pass"),
        "n_warn": sum(1 for c in checks if c["status"] == "warn"),
        "n_fail": sum(1 for c in checks if c["status"] == "fail"),
        "tubes_above_design": above_design,
        "max_wall_c": None if max_t is None else round(max_t, 1),
        "checks": checks,
        "energy_chart": [
            {"label": "نقطه کار", "value": furnace.get("sec_gj_per_t"), "role": "measured"},
            {"label": "نقطه بهینه", "value": furnace.get("sec_opt_gj_per_t"), "role": "optimum"},
            {"label": "جانسون متی SMR", "value": 32.6, "role": "reference"},
            {"label": "نمونه معمول", "value": 30.0, "role": "reference"},
            {"label": "نزدیک BAT", "value": 20.0, "role": "reference"},
        ],
        "carbon_chart": [
            {"label": "نقطه کار", "value": furnace.get("co2_t_per_t"), "role": "measured"},
            {"label": "نقطه بهینه", "value": furnace.get("co2_opt_t_per_t"), "role": "optimum"},
            {"label": "IPCC ریفرمر اولیه", "value": 0.497, "role": "reference"},
            {"label": "IPCC بدون ریفرمر اولیه", "value": 0.67, "role": "reference"},
        ],
        "notes": [
            "تولید متانول از خوراک طراحی ۳۲٬۰۰۰ Nm³/h و بازده مدل، با فرض تبدیل ایده‌آل متان به متانول، برآورد شده است.",
            "انرژی شامل خوراک طراحی به‌اضافه سوخت کوره است و دبی زنده نازل‌ها را دوباره جمع نمی‌کند.",
            "شدت CO₂ فقط CO₂ احتراقی سوخت کوره است و با ضریب کل IPCC هم‌مرز نیست.",
        ],
    }


# Marjan nameplate: 1.65 Mt/y methanol from 1.5 bcm/y South Pars gas.
_MARJAN_METHANOL_T_Y = 1_650_000.0
_MARJAN_GAS_NM3_Y = 1_500_000_000.0


def _scale_to_nameplate(sec_gj_per_t: float | None) -> dict | None:
    """Map a reformer specific energy onto the plant's annual gas and methanol contract."""
    if sec_gj_per_t is None or sec_gj_per_t <= 0:
        return None
    nm3_per_t = float(sec_gj_per_t) / _LHV_GJ_PER_NM3
    gas_nm3 = _MARJAN_METHANOL_T_Y * nm3_per_t
    methanol_t = _MARJAN_GAS_NM3_Y / nm3_per_t
    return {
        "nm3_per_t": round(nm3_per_t, 1),
        "gas_bcm": round(gas_nm3 / 1e9, 3),
        "extra_gas_bcm": round((gas_nm3 - _MARJAN_GAS_NM3_Y) / 1e9, 3),
        "methanol_mt": round(methanol_t / 1e6, 3),
        "methanol_gap_mt": round((_MARJAN_METHANOL_T_Y - methanol_t) / 1e6, 3),
    }


def _plant_balance(db_path: str) -> dict:
    """Two contract questions: gas to hold nameplate product, and product from contract gas."""
    furnace = _carbon_view(db_path)["furnace"]
    design_nm3 = _MARJAN_GAS_NM3_Y / _MARJAN_METHANOL_T_Y
    now = _scale_to_nameplate(furnace.get("sec_gj_per_t"))
    optimum = _scale_to_nameplate(furnace.get("sec_opt_gj_per_t"))
    return {
        "feedstock": "گاز طبیعی فازهای پارس جنوبی، عسلویه",
        "product": "متانول",
        "nameplate": {
            "methanol_mt_per_year": 1.65,
            "gas_bcm_per_year": 1.5,
            "nm3_per_t": round(design_nm3, 1),
            "gj_per_t": round(design_nm3 * _LHV_GJ_PER_NM3, 2),
            "iran_capacity_share_pct": 11,
        },
        "now": now,
        "optimum": optimum,
        "gas_chart": [
            {"label": "قرارداد", "value": 1.5, "role": "reference"},
            {"label": "نقطه کار", "value": None if now is None else now["gas_bcm"], "role": "measured"},
            {"label": "نقطه بهینه", "value": None if optimum is None else optimum["gas_bcm"], "role": "optimum"},
        ],
        "product_chart": [
            {"label": "ظرفیت نامی", "value": 1.65, "role": "reference"},
            {"label": "نقطه کار", "value": None if now is None else now["methanol_mt"], "role": "measured"},
            {"label": "نقطه بهینه", "value": None if optimum is None else optimum["methanol_mt"], "role": "optimum"},
        ],
    }


# Planning assumption for a catalyst order, not a Marjan purchase contract.
_CATALYST_LEAD_DAYS = 180


def _reformer_remaining_h(activity: float | None, age_h: float | None) -> float | None:
    """Hours until activity 0.4, assuming linear loss from 1.0 at hour zero."""
    if activity is None or age_h is None or age_h <= 0:
        return None
    consumed = 1.0 - float(activity)
    if consumed <= 0.02:
        return None
    return max(0.0, float(age_h) * (float(activity) - _CATALYST_REPLACE_BELOW) / consumed)


def _catalyst_health(db_path: str) -> dict:
    """Consumable-catalyst inspection, remaining life, and a purchase window.

    Tube-wall heat and the Module 4 activity are live. Pellet photos and the
    gamma trace are published examples, not a camera or a scan of this plant.
    """
    with dbm.get_conn(db_path) as conn:
        readings = dbm.get_temperature_map(conn)
        furnace = _furnace_burners(conn)
    _mean, hotspots = _hotspot_rows(readings)
    cat = furnace["catalyst"]
    remaining_h = _reformer_remaining_h(cat["activity"], cat["age_h"])
    remaining_days = None if remaining_h is None else int(round(remaining_h / 24))
    buy_now = remaining_days is not None and remaining_days <= _CATALYST_LEAD_DAYS
    order_in_days = None if remaining_days is None else max(0, remaining_days - _CATALYST_LEAD_DAYS)
    n_tubes = len({r["tube_id"] for r in readings})
    if hotspots and any(float(h["wall_temp_c"]) >= 930 or float(h["delta_c"]) >= 50 for h in hotspots):
        wall_light = "red"
    elif hotspots:
        wall_light = "orange"
    elif readings:
        wall_light = "green"
    else:
        wall_light = "grey"
    balance = _plant_balance(db_path)["now"] or {}
    wall_finding = (
        f"{len(hotspots)} لوله بیش از ۳۰ درجه از میانگین دیواره گرم‌تر است. "
        "در ریفرمر بخار، نوار داغ به‌خصوص در ۲۰ تا ۴۰ درصد بالای لوله با کربن یا مسمومیت دیده می‌شود "
        "و فضای خالی کوچک بارگذاری می‌تواند دمای موضعی را حدود ۳۰ درجه بالا ببرد."
        if hotspots else
        "قرائت زنده نوار داغ ندارد. عکس، شکل همین نشانه است برای وقتی دیواره ناهموار شود."
    )
    return {
        "production": {
            "methanol_mt": balance.get("methanol_mt"),
            "methanol_gap_mt": balance.get("methanol_gap_mt"),
            "nameplate_mt": 1.65,
        },
        "inspections": [
            {
                "id": "wall",
                "title": "تفسیر عکس دیواره لوله",
                "practice": "ظاهر لوله در کوره روشن: نوار داغ، نقطه داغ، یا لوله یکدست داغ",
                "light": wall_light,
                "measured": bool(readings),
                "sample_image": "hot-band",
                "n_hot": len(hotspots),
                "finding": wall_finding,
            },
            {
                "id": "pellets",
                "title": "تفسیر ظاهر قرص تخلیه‌شده",
                "practice": "حلقه خاکستری فلزی سالم است؛ توده سیاه و خردشده با کربن و شکست حرارتی هم‌خوان است",
                "light": "grey",
                "measured": False,
                "sample_image": "pellets",
                "finding": "از این واحد قرص تخلیه‌شده‌ای در سامانه نیست. عکس فقط دو ظاهر مرجع را کنار هم می‌گذارد.",
            },
            {
                "id": "gamma",
                "title": "رادیوگرافی گاما",
                "practice": "پروفیل چگالی Tru-Scan: بستر پرتو را می‌گیرد و بخار آن را رد می‌کند",
                "light": "grey",
                "measured": False,
                "sample_image": None,
                "finding": "اسکن گامای بستر سنتز انجام نشده. افت چگالی را، اگر سیگنال عبور کند، کانال یا خردشدگی می‌خوانند.",
                "trace": [
                    {"level": "فضای بالا", "density": 0.12},
                    {"level": "بستر", "density": 0.84},
                    {"level": "ناحیه کم‌چگال", "density": 0.32},
                    {"level": "بستر", "density": 0.8},
                    {"level": "فضای پایین", "density": 0.1},
                ],
            },
        ],
        "charges": [
            {
                "id": "reformer_ni",
                "name": "نیکل ریفرمر اولیه",
                "light": cat["light"],
                "remaining_days": remaining_days,
                "rul_basis": "افت خطی فعالیت مدل از ۱٫۰ تا آستانه تعویض ۰٫۴",
                "purchase": {
                    "amount": 1 if buy_now else 0,
                    "buy_now": buy_now,
                    "when_days": order_in_days,
                    "lead_days": _CATALYST_LEAD_DAYS,
                    "volume_tubes": n_tubes,
                    "volume_mass_kg": None,
                },
            },
            {
                "id": "synthesis_cuzn",
                "name": "مس-روی-آلومینا سنتز متانول",
                "light": "grey",
                "remaining_days": None,
                "literature_life": "۳ تا ۴ سال معمول، برخی شارژها تا بیش از ۷ سال",
                "rul_basis": "سن نصب و اسکن این بستر در سامانه نیست",
                "purchase": {
                    "amount": 0,
                    "buy_now": False,
                    "when_days": None,
                    "lead_days": _CATALYST_LEAD_DAYS,
                    "volume_tubes": None,
                    "volume_mass_kg": None,
                },
            },
        ],
        "sources": [
            {"title": "Johnson Matthey — tube appearance and carbon", "url": "https://matthey.com/documents/d/guest/jm-h2-seminar-optimization-and-troubleshooting-pdf"},
            {"title": "Johnson Matthey — carbon in the top 20–40% of the tube", "url": "https://matthey.com/documents/161599/179189/Carbon-Formation-Webinar-QA.pdf/d51750cf-6081-0025-32c3-8eb3df548d8f"},
            {"title": "Johnson Matthey — voids raise local tube temperature about 30°C", "url": "https://matthey.com/documents/d/guest/jm_training-seminar-2024_catalyst-loading-and-unloading-final-pdf"},
            {"title": "Tracerco Tru-Scan reactor density profile", "url": "https://tracerco.com/services/refineries-and-petrochemical/reactors/"},
            {"title": "Johnson Matthey methanol catalyst life, typically 3–4 years", "url": "https://matthey.com/documents/d/guest/jm-methanol-synthesis-product-brochure-pdf"},
            {"title": "Science Advances — Cu/ZnO life typically 2–6 years", "url": "https://www.science.org/doi/10.1126/sciadv.adk2081"},
            {"title": "Air Liquide — methanol catalyst life up to six years", "url": "https://engineering-technologies.airliquide.com/sites/engineering/files/2022-07/og_02_20_s17_21_do.pdf"},
        ],
    }


def _production_plan(db_path: str) -> dict:
    """Contract-gas production plan under live prices and the asked shocks."""
    furnace = _carbon_view(db_path)["furnace"]
    scaled = _scale_to_nameplate(furnace.get("sec_gj_per_t")) or {}
    plan = build_plan(
        scaled.get("methanol_mt"),
        scaled.get("methanol_gap_mt"),
        furnace.get("sec_gj_per_t"),
        fetch_market(),
    )
    plan["sources"] = [
        {"title": "Methanex regional posted prices", "url": "https://www.methanex.com/our-products/about-methanol/pricing/"},
        {"title": "Henry Hub, gold, crude and Methanex shares", "url": "https://query1.finance.yahoo.com/v8/finance/chart/NG=F?interval=1d&range=1d"},
        {"title": "TGJU free-market dollar", "url": "https://call1.tgju.org/ajax.json"},
        {"title": "TSETMC Zagros close", "url": "https://cdn.tsetmc.com/api/ClosingPrice/GetClosingPriceInfo/13235547361447092"},
        {"title": "Iran methanol capacity, CCFGroup 20 Mar 2026", "url": "https://www.ccfgroup.com/newscenter/newsview.php?Class_ID=D00000&Info_ID=2026032030020"},
        {"title": "Methanex on MTO shutdowns and demand, 30 Oct 2026", "url": "https://www.fool.com/earnings/call-transcripts/2026/10/30/methanex-meoh-q3-2026-earnings-call-transcript/"},
        {"title": "Zagros realized export price and gas cost, H1 2026", "url": "https://roydadnaft.ir/English/19367/"},
    ]
    return plan


def _live_loop(db_path: str) -> None:
    """Append one furnace sample every 5 seconds while MRTO_LIVE_SIM=1."""
    import random
    for _ in range(40):
        time.sleep(1)
        try:
            with dbm.get_conn(db_path) as conn:
                n = conn.execute("SELECT COUNT(*) FROM temperature_readings").fetchone()[0]
            if n:
                break
        except Exception:
            logger.exception("live simulator waiting for readings")
    while True:
        try:
            with dbm.get_conn(db_path) as conn:
                tubes = dbm.get_tubes(conn)
                if not tubes:
                    time.sleep(5)
                    continue
                prev_rows = conn.execute(
                    """
                    SELECT tube_id, wall_temp_c FROM temperature_readings
                    WHERE id IN (SELECT MAX(id) FROM temperature_readings GROUP BY tube_id)
                    """
                ).fetchall()
                prev = {r["tube_id"]: float(r["wall_temp_c"]) for r in prev_rows}
                cur = conn.execute("SELECT MAX(ts_minute) FROM temperature_readings").fetchone()
                ts = int(cur[0] or 0) + 1
                for tube in tubes:
                    tid = tube["tube_id"]
                    base = prev.get(tid, 880.0)
                    temp = base + random.uniform(-1.2, 1.2)
                    if tid in _HOT_TUBES:
                        temp = min(960.0, max(935.0, temp if base >= 920 else 942.0))
                    else:
                        temp = min(910.0, max(840.0, temp))
                    other = conn.execute(
                        "SELECT AVG(wall_temp_c) FROM temperature_readings WHERE tube_id != ?",
                        (tid,),
                    ).fetchone()[0]
                    mean_ref = other if other is not None else 880.0
                    dbm.insert_temperature_reading(conn, tid, ts, round(temp, 1), "live")
                    if temp > mean_ref + 30:
                        open_same = conn.execute(
                            "SELECT 1 FROM alerts WHERE tube_id=? AND module='module1' AND status='open'",
                            (tid,),
                        ).fetchone()
                        if open_same is None:
                            dbm.insert_alert(
                                conn, tid, ts, "high", "module1",
                                f"Hotspot on {tid}: {temp:.1f}°C",
                            )
                latest = conn.execute(
                    """
                    SELECT burner_id, fuel_flow, intensity, angle FROM burner_status
                    WHERE id IN (SELECT MAX(id) FROM burner_status GROUP BY burner_id)
                    """
                ).fetchall()
                for row in latest:
                    bid = row["burner_id"]
                    intensity = float(row["intensity"] or 0.9)
                    fuel = float(row["fuel_flow"] or 190)
                    if bid in _DEGRADED_BURNERS:
                        intensity = _DEGRADED_BURNERS[bid] + random.uniform(-0.02, 0.02)
                        fuel = 250 + random.uniform(-8, 8)
                    else:
                        intensity = min(1.15, max(0.85, intensity + random.uniform(-0.01, 0.01)))
                        fuel = min(210, max(170, fuel + random.uniform(-4, 4)))
                    angle = max(-15.0, min(15.0, float(row["angle"] or 0) + random.uniform(-0.4, 0.4)))
                    conn.execute(
                        "INSERT INTO burner_status (burner_id, ts_minute, fuel_flow, intensity, angle) "
                        "VALUES (?,?,?,?,?)",
                        (bid, ts, round(fuel, 1), round(intensity, 3), round(angle, 2)),
                    )
        except Exception:
            logger.exception("live simulator tick failed")
        time.sleep(5)


def _avg_wall_temp_per_tube(conn, tube_ids: list[str]) -> dict[str, float]:
    """Return the latest average wall temperature per tube from readings.

    Falls back to 880 °C (nominal operating temp) for tubes with no readings,
    so the RUL model always gets a valid input instead of a hardcoded constant.
    """
    if not tube_ids:
        return {}
    placeholders = ",".join("?" * len(tube_ids))
    rows = conn.execute(
        f"SELECT tube_id, AVG(wall_temp_c) AS avg_t FROM ("
        f"  SELECT tube_id, wall_temp_c, ts_minute, "
        f"         ROW_NUMBER() OVER (PARTITION BY tube_id ORDER BY ts_minute DESC) AS rn "
        f"  FROM temperature_readings WHERE tube_id IN ({placeholders})"
        f") WHERE rn <= 5 GROUP BY tube_id",
        tube_ids
    ).fetchall()
    out = {r["tube_id"]: float(r["avg_t"]) for r in rows if r["avg_t"] is not None}
    # Tubes without readings → nominal operating temperature.
    return {tid: out.get(tid, 880.0) for tid in tube_ids}


# ────────────────────────────────────────────────────────────────────
# Metric-path normalization (Phase 3)
# ────────────────────────────────────────────────────────────────────
# Collapse numeric path segments to ":id" so /alerts/5/ack and
# /alerts/99/ack become the same Prometheus label. Without this, every
# alert ID becomes its own time series and cardinality blows up.
import re as _re
_ID_RE = _re.compile(r"/\d+(?=/|$)")


def _metric_path(path: str) -> str:
    return _ID_RE.sub("/:id", path) if path else "/"

# ────────────────────────────────────────────────────────────────────
# Pydantic models
# ────────────────────────────────────────────────────────────────────
class LoginReq(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class RefreshReq(BaseModel):
    refresh_token: str = Field(min_length=10, max_length=2048)


class LogoutReq(BaseModel):
    refresh_token: str = Field(min_length=10, max_length=2048)


class OptimizationReq(BaseModel):
    sc_ratio: float = Field(ge=2.0, le=5.0)
    t_out_c: float = Field(ge=700, le=1000)
    feed_flow_nm3h: float = Field(ge=10000, le=50000)
    cat_age: float = Field(ge=0, le=1.2)
    fuel_flow_nm3h: float = Field(ge=5000, le=20000)
    alpha_eff: float = Field(0.7, ge=0, le=1)
    alpha_fuel: float = Field(0.3, ge=0, le=1)


class TemperatureIngest(BaseModel):
    tube_id: str = Field(min_length=1, max_length=32, pattern=r"^[A-Za-z0-9_\-]+$")
    ts_minute: int = Field(ge=0, le=10_000_000)
    wall_temp_c: float = Field(ge=0, le=1500)
    source: str = Field(default="ir", max_length=32)


class TemperatureBatch(BaseModel):
    readings: list[TemperatureIngest] = Field(min_length=1, max_length=5000)


# ────────────────────────────────────────────────────────────────────
# JWT helpers (simple HMAC-based for test-friendly env)
# ────────────────────────────────────────────────────────────────────
# SECURITY: refuse to run with the well-known insecure default secrets.
# Operators must set MRTO_JWT_SECRET to a strong value (≥32 chars).
_INSECURE_JWT_SECRETS = {
    "mrto-dev-secret-1405",
    "change-me-in-production",
    "staging-jwt-change-me",
    "",
}
# Phase 4: route through the secrets module so the value can come from
# env, a _FILE mount, Vault, or SOPS — not just env.
from mrto import secrets as _sec
JWT_SECRET = _sec.get_secret("MRTO_JWT_SECRET", "") or ""
if JWT_SECRET in _INSECURE_JWT_SECRETS:
    raise RuntimeError(
        "MRTO_JWT_SECRET is unset or matches a known insecure placeholder. "
        "Set it to a strong random value (>=32 chars), e.g. "
        "`export MRTO_JWT_SECRET=$(openssl rand -hex 32)`. "
        "For local tests/dev, set MRTO_ALLOW_INSECURE_JWT=1 to bypass this guard."
    )
if len(JWT_SECRET) < 32 and not os.environ.get("MRTO_ALLOW_INSECURE_JWT"):
    raise RuntimeError(
        f"MRTO_JWT_SECRET is only {len(JWT_SECRET)} chars; need >=32. "
        "Set MRTO_ALLOW_INSECURE_JWT=1 to bypass (dev/tests only)."
    )

JWT_TTL = int(os.environ.get("MRTO_JWT_TTL", 8 * 3600))
REFRESH_TTL = int(os.environ.get("MRTO_REFRESH_TTL", 30 * 24 * 3600))
ALLOW_ANON = os.environ.get("MRTO_ALLOW_ANON", "1") == "1"

_bearer = HTTPBearer(auto_error=False)


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _b64d(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def make_jwt(username: str, role: str, jti: str | None = None,
              kind: str = "access") -> str:
    """Sign a JWT. `kind` is 'access' (short-lived) or 'refresh' (long-lived).

    The refresh token's jti is stored in the DB so it can be revoked /
    rotated. Access tokens are stateless (no DB lookup per request).
    """
    header = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    ttl = JWT_TTL if kind == "access" else REFRESH_TTL
    payload = {
        "sub": username, "role": role, "kind": kind,
        "iat": int(time.time()), "exp": int(time.time()) + ttl,
        "jti": jti or secrets.token_hex(8),
    }
    body = _b64(json.dumps(payload, separators=(",", ":")).encode())
    sig = hmac.new(JWT_SECRET.encode(), f"{header}.{body}".encode(), hashlib.sha256).digest()
    sig_b = _b64(sig)
    return f"{header}.{body}.{sig_b}"


def verify_jwt(token: str, expected_kind: str | None = None) -> dict | None:
    """Verify HMAC-SHA256 JWT. Rejects alg-confusion by pinning HS256.

    If `expected_kind` is set, the token's `kind` claim must match (so a
    refresh token cannot be used as an access token and vice versa).
    """
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header_b, body_b, sig_b = parts
        # Pin the algorithm — reject "none" and any asymmetric alg.
        header = json.loads(_b64d(header_b))
        if header.get("alg") != "HS256":
            return None
        expected = _b64(hmac.new(JWT_SECRET.encode(),
                                f"{header_b}.{body_b}".encode(), hashlib.sha256).digest())
        if not secrets.compare_digest(expected, sig_b):
            return None
        payload = json.loads(_b64d(body_b))
        if payload.get("exp", 0) < time.time():
            return None
        if expected_kind and payload.get("kind") != expected_kind:
            return None
        return payload
    except Exception:
        return None


async def current_user(creds: HTTPAuthorizationCredentials = Depends(_bearer)) -> dict:
    """Resolve the authenticated user from the Bearer access token.

    In production (MRTO_ALLOW_ANON != "1"), a missing/invalid token returns
    401. In dev/test mode (default), missing tokens fall back to anonymous
    read-only viewer so the test suite can run without auth.

    Refresh tokens are explicitly rejected here — they may only be used
    at /api/v1/auth/refresh, never as a bearer credential for data routes.
    """
    if creds is None or not creds.credentials:
        if ALLOW_ANON:
            return {"sub": "anonymous", "role": "viewer"}
        raise HTTPException(status_code=401, detail="Authentication required")
    payload = verify_jwt(creds.credentials, expected_kind="access")
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return payload


# Algorithms match backend/mrto/models.py. Larson-Miller only labels RUL;
# it is not the learned predictor.
_MODEL_ALGORITHMS = [
    {
        "module": "module1_temperature_map",
        "task": "دمای دیواره لوله",
        "algorithm": "RandomForestRegressor",
        "prep": "StandardScaler",
        "settings": "120 درخت، عمق 14",
        "holdout_pct": 20,
        "n_test_field": "n_test",
        "note": "",
    },
    {
        "module": "module2_rul",
        "task": "عمر باقیمانده (ساعت)",
        "algorithm": "RandomForestRegressor",
        "prep": "StandardScaler",
        "settings": "150 درخت، عمق 12",
        "holdout_pct": 20,
        "n_test_field": "n_test",
        "note": "برچسب آموزش از پارامتر لارسن-میلر با ثابت 20 برای آلیاژ HP-Nb ساخته می‌شود.",
    },
    {
        "module": "module2_rul",
        "task": "احتمال خرابی ۹۰ روز",
        "algorithm": "RandomForestClassifier",
        "prep": "StandardScaler",
        "settings": "150 درخت، عمق 10، class_weight=balanced",
        "holdout_pct": 20,
        "n_test_field": "n_test",
        "note": "برچسب کلاس از آستانه احتمال خرابی 0.3 روی همان برچسب لارسن-میلر است.",
    },
    {
        "module": "module3_optimization",
        "task": "بازده تبدیل",
        "algorithm": "GradientBoostingRegressor",
        "prep": "StandardScaler",
        "settings": "200 درخت، عمق 4، نرخ یادگیری 0.1",
        "holdout_pct": 20,
        "n_test_field": "n_test",
        "note": "",
    },
    {
        "module": "module3_optimization",
        "task": "ریسک رسوب کک",
        "algorithm": "GradientBoostingRegressor",
        "prep": "StandardScaler",
        "settings": "150 درخت، عمق 4، نرخ یادگیری 0.1",
        "holdout_pct": 20,
        "n_test_field": "n_test",
        "note": "همان جداسازی آموزش و آزمون مدل بازده.",
    },
    {
        "module": "module3_optimization",
        "task": "جستجوی نقطه کار",
        "algorithm": "جستجوی شبکه‌ای مقید",
        "prep": "—",
        "settings": "S/C حداقل 2.5 و ریسک کک حداکثر 0.5",
        "holdout_pct": None,
        "n_test_field": None,
        "note": "",
    },
    {
        "module": "module4_burner_catalyst",
        "task": "مشعل ناکارآمد",
        "algorithm": "RandomForestClassifier",
        "prep": "StandardScaler",
        "settings": "150 درخت، عمق 10، class_weight=balanced",
        "holdout_pct": 20,
        "n_test_field": "n_test",
        "note": "",
    },
    {
        "module": "module4_burner_catalyst",
        "task": "فعالیت کاتالیست",
        "algorithm": "RandomForestRegressor",
        "prep": "StandardScaler",
        "settings": "150 درخت، عمق 12",
        "holdout_pct": 20,
        "n_test_field": None,
        "note": "شمارش نمونه آزمون این رگرسور جدا در فایل معیار ذخیره نشده است.",
    },
]

_COV_FILE_FA = {
    "models.py": "مدل‌های یادگیری",
    "app.py": "سرویس API",
    "db.py": "پایگاه داده",
    "synthetic_data.py": "داده مصنوعی",
    "real_data.py": "داده واقعی",
    "train_models.py": "اسکریپت آموزش",
    "secrets.py": "رمز و کلید",
    "ratelimit.py": "محدودیت نرخ",
    "observability.py": "سلامت و متریک",
    "migrations.py": "مهاجرت طرح",
    "db_adapter.py": "اتصال پایگاه",
    "__init__.py": "بسته",
}

_TEST_GROUP = {
    "test_models.py": "آزمون مدل",
    "test_synthetic_data.py": "داده مصنوعی",
    "test_real_data.py": "داده واقعی",
    "test_db.py": "پایگاه داده",
    "test_api.py": "رابط API",
    "test_phase3.py": "سلامت و مشاهده‌پذیری",
    "test_phase4.py": "رمز و پشتیبان",
}

_TEST_TITLES = {
    "test_module1_train_and_predict": "آموزش و پیش‌بینی دمای دیواره",
    "test_module1_srs_accuracy_target": "هدف SRS خطای دمای دیواره",
    "test_module2_train_and_predict": "آموزش عمر باقیمانده و احتمال خرابی",
    "test_module3_train_and_optimize": "آموزش بازده و جستجوی نقطه کار",
    "test_module4_train_and_predict": "آموزش مشعل و فعالیت کاتالیست",
    "test_model_persistence_roundtrip": "ذخیره و بارگذاری مدل",
    "test_optimization_feasibility_constraint": "قید S/C و ریسک کک",
    "test_lmp_returns_finite_for_design_temperature": "عمر لارسن-میلر در دمای طراحی",
    "test_lmp_higher_temperature_lower_life": "کاهش عمر با افزایش دما",
    "test_tubes_shape_and_uniqueness": "یکتایی شناسه لوله",
    "test_burner_profiles_sensible": "محدوده شدت مشعل",
    "test_timeseries_within_bounds": "محدوده سری زمانی کوره",
    "test_temperature_map_coverage": "پوشش نقشه دما برای هر لوله",
    "test_rul_labels_nonneg_and_probability_in_unit": "عمر نامنفی و احتمال در بازه صفر تا یک",
    "test_sc_optimization_dataset_shape_and_constraints": "شکل داده بهینه‌سازی و قیدها",
    "test_burner_status_dataset_balanced_enough": "توازن برچسب مشعل",
    "test_catalyst_dataset_activity_in_unit": "فعالیت کاتالیست در بازه مجاز",
    "test_generate_all_writes_files": "نوشتن فایل‌های داده مصنوعی",
    "test_datasets_have_required_keys": "کلیدهای داده واقعی",
    "test_fetch_airfoil_returns_dataframe_or_none": "خواندن داده ایرفویل",
    "test_fetch_all_returns_dict": "خواندن همه داده‌های واقعی",
    "test_fetch_handles_missing_dir": "پوشه دادهٔ ناموجود",
    "test_init_creates_schema": "ساخت طرح پایگاه",
    "test_seed_admin_user": "کاربر مدیر اولیه",
    "test_seed_demo_tubes": "لوله‌های نمونه",
    "test_insert_and_retrieve_temperature": "ثبت و خواندن دما",
    "test_insert_and_ack_alert": "ثبت و تأیید هشدار",
    "test_ack_nonexistent_alert_returns_false": "تأیید هشدار ناموجود",
    "test_audit_log": "رد ممیزی",
    "test_get_tubes_by_row": "لوله‌ها بر اساس ردیف",
    "test_hotspot_alert_generated_on_ingest": "هشدار نقطه داغ هنگام ثبت",
    "test_health": "سلامت سرویس",
    "test_root": "ریشه سرویس",
    "test_login_success": "ورود درست",
    "test_login_failure": "ورود نادرست",
    "test_get_tubes": "فهرست لوله‌ها",
    "test_get_tubes_filtered_by_row": "فیلتر ردیف لوله",
    "test_ingest_temperature_and_hotspot_alert": "ثبت دما و هشدار نقطه داغ",
    "test_ingest_creates_missing_tube": "ساخت لولهٔ ناموجود",
    "test_ingest_rejects_anonymous": "رد ثبت ناشناس",
    "test_ingest_rejects_bad_input": "رد ورودی خارج از محدوده",
    "test_temperature_map_endpoint": "نقشه دما",
    "test_rul_endpoint_fallback": "عمر باقیمانده و چراغ ریسک",
    "test_alert_ack_workflow": "گردش تأیید هشدار",
    "test_alert_ack_nonexistent": "هشدار ناموجود در API",
    "test_burners_endpoint_empty": "مشعل بدون قرائت",
    "test_metrics_endpoint": "معیار کیفیت مدل",
    "test_daily_report": "گزارش روزانه",
    "test_invalid_optimization_input": "ورودی نامعتبر بهینه‌سازی",
    "test_openapi_docs_available": "مستند OpenAPI",
    "test_jwt_tampered_payload_rejected": "رد توکن دستکاری‌شده",
    "test_jwt_expired_rejected": "رد توکن منقضی",
    "test_jwt_alg_none_rejected": "رد الگوریتم none",
    "test_protected_endpoint_rejects_bad_token": "رد توکن خراب",
    "test_refresh_token_rotation": "چرخش توکن تازه‌سازی",
    "test_refresh_token_reuse_revokes_family": "ابطال خانواده با استفادهٔ مجدد",
    "test_refresh_rejects_access_token": "رد توکن دسترسی به‌جای تازه‌سازی",
    "test_access_rejects_refresh_token": "رد توکن تازه‌سازی به‌جای دسترسی",
    "test_logout_revokes_refresh_family": "خروج و ابطال خانواده",
    "test_refresh_invalid_token": "توکن تازه‌سازی نامعتبر",
    "test_login_rate_limit": "محدودیت تلاش ورود",
    "test_rate_limit_headers_present": "سربرگ محدودیت نرخ",
    "test_model_versions_empty_by_default": "نسخه‌های مدل خالی",
    "test_model_versions_rejects_viewer": "رد بیننده از نسخه‌ها",
    "test_active_model_404_when_untrained": "مدل فعال ناموجود",
    "test_register_and_list_model_version": "ثبت و فهرست نسخه",
    "test_temperature_batch_dedups_hotspot_alerts": "یک هشدار برای دسته دما",
    "test_burners_keep_latest_row_per_burner": "آخرین ردیف هر مشعل",
    "test_temperature_map_reports_latest_timestamp": "زمان آخرین نقشه دما",
    "test_maintenance_plan_covers_every_tube": "برنامه نگهداری همه لوله‌ها",
    "test_digital_twin_line_order": "ترتیب واحدهای دوقلو",
    "test_reconciliation_flags_impossible_temperature": "دمای غیرممکن",
    "test_carbon_converts_live_burner_fuel": "تبدیل سوخت مشعل به CO2",
    "test_benchmark_flags_tube_hotter_than_100000h_reference": "لوله داغ‌تر از مرجع صد هزار ساعت",
    "test_plant_balance_scales_marjan_contract": "تراز قرارداد خوراک و محصول",
    "test_catalyst_health_separates_live_wall_from_sample_images": "سلامت کاتالیست مصرفی و پنجره خرید",
    "test_prometheus_metrics_endpoint": "خروجی پرومتئوس",
    "test_prometheus_metrics_have_request_id_header": "شناسه درخواست روی متریک",
    "test_request_id_propagated_from_header": "عبور شناسه درخواست",
    "test_deep_health_returns_checks": "سلامت عمیق",
    "test_deep_health_reports_schema_version": "نسخه طرح در سلامت عمیق",
    "test_migrations_idempotent": "مهاجرت تکرارپذیر",
    "test_migrations_apply_in_order": "ترتیب مهاجرت",
    "test_json_logger_emits_valid_json": "لاگ JSON",
    "test_rul_endpoint_handles_empty_tubes": "عمر باقیمانده بدون لوله",
    "test_temperature_map_with_no_readings": "نقشه دما بدون قرائت",
    "test_optimization_input_at_boundaries": "بهینه‌سازی روی مرز ورودی",
    "test_ingest_extreme_temperature_creates_alert": "دمای شدید و هشدار",
    "test_alerts_filter_by_status": "فیلتر وضعیت هشدار",
    "test_read_replica_helper_falls_back_when_unset": "برگشت به پایگاه اصلی",
    "test_openapi_has_tags": "برچسب‌های OpenAPI",
    "test_secret_from_env": "رمز از متغیر محیط",
    "test_secret_cached": "نهان‌سازی رمز",
    "test_secret_default_when_missing": "رمز پیش‌فرض در حالت توسعه",
    "test_require_secret_raises_when_missing": "الزام رمز در حالت سخت",
    "test_require_secret_returns_value": "خواندن رمز الزامی",
    "test_secret_from_file": "رمز از فایل",
    "test_secret_env_takes_priority_over_file": "اولویت محیط بر فایل",
    "test_secret_file_missing_falls_through": "فایل رمز ناموجود",
    "test_vault_backend_disabled_without_addr": "غیرفعال بودن Vault بدون آدرس",
    "test_sops_backend_disabled_without_sops": "غیرفعال بودن SOPS",
    "test_backup_sqlite_online": "پشتیبان آنلاین SQLite",
    "test_backup_dry_run_writes_nothing": "پشتیبان آزمایشی بدون نوشتن",
    "test_backup_retention_deletes_old_sets": "حذف پشتیبان کهنه",
    "test_restore_sqlite_with_prerestore_snapshot": "بازیابی با تصویر قبل از بازگردانی",
    "test_restore_declines_without_yes": "رد بازیابی بدون تأیید",
    "test_restore_latest_picks_newest": "انتخاب تازه‌ترین پشتیبان",
}

_MODULE_LABEL_FA = {
    "module1_temperature_map": "۱. پایش دما",
    "module2_rul": "۲. عمر باقیمانده",
    "module3_optimization": "۳. بهینه‌سازی S/C",
    "module4_burner_catalyst": "۴. مشعل و کاتالیست",
}


def _quality_paths() -> tuple[str, str]:
    backend = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(backend, "coverage.xml"), os.path.join(backend, "junit.xml")


def _algorithm_rows(metrics: dict) -> list[dict]:
    rows = []
    for spec in _MODEL_ALGORITHMS:
        data = metrics.get(spec["module"]) if isinstance(metrics, dict) else None
        n_test = None
        field = spec["n_test_field"]
        if field and isinstance(data, dict) and isinstance(data.get(field), (int, float)):
            n_test = int(data[field])
        rows.append({
            "module": spec["module"],
            "module_label": _MODULE_LABEL_FA.get(spec["module"], spec["module"]),
            "task": spec["task"],
            "algorithm": spec["algorithm"],
            "prep": spec["prep"],
            "settings": spec["settings"],
            "holdout_pct": spec["holdout_pct"],
            "n_test": n_test,
            "note": spec["note"],
        })
    return rows


def _coverage_files(path: str) -> tuple[dict | None, list[dict]]:
    if not os.path.exists(path):
        return None, []
    root = ET.parse(path).getroot()
    summary = {
        "line_rate": round(float(root.get("line-rate") or 0), 4),
        "lines_valid": int(root.get("lines-valid") or 0),
        "lines_covered": int(root.get("lines-covered") or 0),
    }
    files = []
    for node in root.iter("class"):
        filename = node.get("filename") or node.get("name") or ""
        lines = [line for line in node if line.tag == "lines"]
        line_nodes = []
        for holder in lines:
            line_nodes.extend(child for child in holder if child.tag == "line")
        if not line_nodes:
            line_nodes = [child for child in node.iter("line")]
        files.append({
            "file": filename,
            "label": _COV_FILE_FA.get(filename, filename),
            "line_rate": round(float(node.get("line-rate") or 0), 4),
            "lines_valid": len(line_nodes),
            "lines_covered": sum(1 for line in line_nodes if int(line.get("hits") or 0) > 0),
        })
    return summary, files


def _junit_tests(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    root = ET.parse(path).getroot()
    rows = []
    for case in root.iter("testcase"):
        classname = case.get("classname") or ""
        file_stem = next((part for part in classname.split(".") if part.startswith("test_")), "")
        filename = f"{file_stem}.py" if file_stem else classname
        name = case.get("name") or ""
        if case.find("failure") is not None:
            status = "failed"
        elif case.find("error") is not None:
            status = "error"
        elif case.find("skipped") is not None:
            status = "skipped"
        else:
            status = "passed"
        try:
            elapsed = round(float(case.get("time") or 0), 3)
        except ValueError:
            elapsed = 0.0
        rows.append({
            "name": name,
            "title": _TEST_TITLES.get(name, name),
            "file": filename,
            "group": _TEST_GROUP.get(filename, filename),
            "status": status,
            "time_s": elapsed,
        })
    return rows


def _suite_report() -> dict:
    """Last pytest run and line coverage of the mrto package.

    Files are written by pytest into backend/. Missing files stay empty
    instead of inventing a pass rate.
    """
    coverage_path, junit_path = _quality_paths()
    summary, files = _coverage_files(coverage_path)
    tests = _junit_tests(junit_path)
    passed = sum(1 for row in tests if row["status"] == "passed")
    failed = sum(1 for row in tests if row["status"] in ("failed", "error"))
    skipped = sum(1 for row in tests if row["status"] == "skipped")
    return {
        "available": summary is not None or bool(tests),
        "gate": 0.70,
        "line_rate": None if summary is None else summary["line_rate"],
        "lines_valid": None if summary is None else summary["lines_valid"],
        "lines_covered": None if summary is None else summary["lines_covered"],
        "tests_total": len(tests),
        "tests_passed": passed,
        "tests_failed": failed,
        "tests_skipped": skipped,
        "duration_s": round(sum(row["time_s"] for row in tests), 1),
        "files": files,
        "tests": tests,
    }


# ────────────────────────────────────────────────────────────────────
# App factory
# ────────────────────────────────────────────────────────────────────
def create_app(db_path: Optional[str] = None) -> FastAPI:
    if db_path is None:
        db_path = dbm.DEFAULT_DB_PATH
    dbm.init_db(db_path)

    app = FastAPI(
        title="MRTO — Methanol Reformer Tube & Operation Optimizer",
        version="1.0.0",
        description=(
            "Smart monitoring of reformer tube health + S/C optimization "
            "for Marjan Petrochemical (TRL 5).\n\n"
            "**Auth**: all data endpoints require a Bearer access token from "
            "`/api/v1/auth/login`. Refresh tokens rotate on every use; reuse "
            "revokes the entire family. Rate-limited at 100 req/min/IP by "
            "default, 10 logins/min/IP for brute-force protection.\n\n"
            "**Observability**: structured JSON logs (with `request_id`), "
            "Prometheus metrics at `/metrics`, deep health at "
            "`/api/v1/health/deep`."
        ),
        openapi_tags=[
            {"name": "health", "description": "Liveness/readiness probes"},
            {"name": "auth", "description": "Login, refresh, logout (JWT + rotating refresh tokens)"},
            {"name": "tubes", "description": "Reformer tube registry"},
            {"name": "temperature", "description": "Wall-temperature readings + hotspots"},
            {"name": "rul", "description": "Remaining-useful-life predictions (Module 2)"},
            {"name": "optimization", "description": "S/C ratio optimization (Module 3)"},
            {"name": "burners", "description": "Burner status + inefficiency flag (Module 4)"},
            {"name": "alerts", "description": "Hotspot & failure alerts + ack workflow"},
            {"name": "reports", "description": "Daily health report"},
            {"name": "metrics", "description": "Model quality metrics + model versioning"},
            {"name": "ingest", "description": "Write endpoints (manager/operator/it only)"},
        ],
    )
    # CORS: configurable origins via env; default to localhost only (not "*").
    cors_origins = os.environ.get(
        "MRTO_CORS_ORIGINS",
        "http://localhost:8080,http://127.0.0.1:8080",
    )
    allow_list = [o.strip() for o in cors_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )

    # Security headers middleware (defense-in-depth against XSS, clickjacking, MITM).
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response

    # Rate limiter: protects /auth/login (brute-force) and the API surface.
    limiter = RateLimiter(
        default_limit=int(os.environ.get("MRTO_RATE_LIMIT", 100)),
        default_window=int(os.environ.get("MRTO_RATE_WINDOW", 60)),
    )
    app.middleware("http")(limiter.as_middleware())

    # Phase 3: structured logging + request ID + Prometheus metrics.
    log = obs.get_logger("mrto.app")

    @app.middleware("http")
    async def observability_middleware(request: Request, call_next):
        # Assign a request ID and surface it on the response so the client
        # can correlate support tickets with server logs.
        rid = obs.set_request_id(request.headers.get("x-request-id") or None)
        obs.INFLIGHT and obs.INFLIGHT.inc()
        t0 = time.time()
        # Normalize path for metrics so /alerts/5/ack and /alerts/99/ack
        # collapse into one label (avoids high-cardinality blowup).
        raw_path = request.url.path
        metric_path = _metric_path(raw_path)
        try:
            response: Response = await call_next(request)
        except Exception:
            obs.record_request(request.method, metric_path, 500,
                               time.time() - t0)
            log.exception("unhandled error", extra={"path": raw_path})
            raise
        finally:
            if obs.INFLIGHT:
                obs.INFLIGHT.dec()
        obs.record_request(request.method, metric_path,
                           response.status_code, time.time() - t0)
        response.headers["X-Request-ID"] = rid
        # Access log line (JSON when MRTO_LOG_PRETTY != 1).
        log.info("request",
                 extra={"method": request.method, "path": raw_path,
                        "status": response.status_code,
                        "ms": int((time.time() - t0) * 1000)})
        return response

    app.state.db_path = db_path
    app.state.limiter = limiter
    app.state.log = log

    # ── Endpoints ───────────────────────────────────────────────────
    @app.get("/api/v1/health",
             tags=["health"],
             summary="Shallow health (load-balancer ping)",
             response_description="200 if the process is up and the DB answers.")
    def health() -> dict:
        with dbm.get_conn(app.state.db_path) as conn:
            n_tubes = conn.execute("SELECT COUNT(*) FROM tubes").fetchone()[0]
            n_alerts = conn.execute("SELECT COUNT(*) FROM alerts WHERE status='open'").fetchone()[0]
        return {
            "status": "ok",
            "service": "mrto",
            "version": "1.0.0",
            "ts_epoch": int(time.time()),
            "n_tubes": n_tubes,
            "open_alerts": n_alerts,
        }

    @app.post("/api/v1/auth/login",
              tags=["auth"],
              summary="Login → access + refresh token",
              responses={401: {"description": "Invalid credentials"},
                         429: {"description": "Rate-limited (>10 logins/min/IP)"}})
    def login(req: LoginReq) -> dict:
        with dbm.get_conn(app.state.db_path) as conn:
            user = dbm.verify_user(conn, req.username, req.password)
            if user is None:
                # Audit failed attempt (rate-limited above; audit gives forensics).
                dbm.audit(conn, req.username, "login_failed", "")
                obs.inc_login_failure()
                raise HTTPException(status_code=401, detail="Invalid credentials")
            dbm.audit(conn, user["username"], "login", "")
            obs.inc_login_success()
            # Update last_login_at if the column exists (migration 2).
            try:
                conn.execute(
                    "UPDATE users SET last_login_at=? WHERE id=?",
                    (int(time.time()), user["id"])
                )
            except Exception:
                pass  # column missing on unmigrated DB; harmless
            # Issue short-lived access + long-lived refresh token (DB-backed
            # so it can be rotated / revoked).
            access = make_jwt(user["username"], user["role"], kind="access")
            refresh_payload = dbm.issue_refresh_token(
                conn, user["username"], user["role"]
            )
            refresh = make_jwt(user["username"], user["role"],
                                jti=refresh_payload["jti"], kind="refresh")
        return {
            "access_token": access,
            "refresh_token": refresh,
            "token_type": "Bearer",
            "role": user["role"],
            "expires_in": JWT_TTL,
        }

    @app.post("/api/v1/auth/refresh",
              tags=["auth"],
              summary="Rotate refresh token (single-use)",
              responses={401: {"description": "Invalid/expired token OR reuse detected (family revoked)"}})
    def refresh(req: RefreshReq) -> dict:
        """Exchange a valid refresh token for a new access token + rotated
        refresh token. Detects reuse: if the presented refresh token has
        already been rotated, the entire token family is revoked and 401
        is returned (the client must re-authenticate).
        """
        payload = verify_jwt(req.refresh_token, expected_kind="refresh")
        if payload is None:
            raise HTTPException(status_code=401, detail="Invalid refresh token")
        jti = payload.get("jti")
        if not jti:
            raise HTTPException(status_code=401, detail="Malformed refresh token")
        with dbm.get_conn(app.state.db_path) as conn:
            new_payload, err = dbm.rotate_refresh_token(conn, jti)
            if err is not None:
                if err["error"] == "reuse_detected":
                    # The reuse-detection UPDATE was applied above and will be
                    # committed by the context manager. We must NOT raise inside
                    # the `with` block — doing so would rollback the revocation.
                    # Capture the signal and raise after the block exits cleanly.
                    obs.inc_refresh_reuse()
                    pass
                else:
                    raise HTTPException(status_code=401,
                                        detail=f"Refresh failed: {err['error']}")
            else:
                dbm.audit(conn, new_payload["username"], "token_refresh", "")
                access = make_jwt(new_payload["username"], new_payload["role"],
                                   kind="access")
                new_refresh = make_jwt(new_payload["username"], new_payload["role"],
                                        jti=new_payload["jti"], kind="refresh")
                return {
                    "access_token": access,
                    "refresh_token": new_refresh,
                    "token_type": "Bearer",
                    "role": new_payload["role"],
                    "expires_in": JWT_TTL,
                }
        # Reuse-detected path: connection has committed the family revocation.
        raise HTTPException(
            status_code=401,
            detail="Refresh token reuse detected — session revoked"
        )

    @app.post("/api/v1/auth/logout",
              tags=["auth"],
              summary="Revoke the refresh-token family")
    def logout(req: LogoutReq,
                _user: dict = Depends(current_user)) -> dict:
        """Revoke the refresh-token family on logout. The access token is
        stateless and remains valid until it expires (short TTL), but the
        refresh chain is dead so it cannot be renewed."""
        payload = verify_jwt(req.refresh_token, expected_kind="refresh")
        if payload is None:
            # Idempotent: an invalid/already-expired refresh token is fine here.
            return {"ok": True, "revoked": 0}
        jti = payload.get("jti")
        with dbm.get_conn(app.state.db_path) as conn:
            row = conn.execute(
                "SELECT family_id FROM refresh_tokens WHERE jti=?", (jti,)
            ).fetchone()
            if row is None:
                return {"ok": True, "revoked": 0}
            n = dbm.revoke_family(conn, row["family_id"], reason="logout")
            dbm.audit(conn, _user.get("sub", "?"), "logout",
                       f"revoked {n} tokens in family")
        return {"ok": True, "revoked": n}

    @app.get("/api/v1/tubes")
    def tubes(row: Optional[int] = None, _user: dict = Depends(current_user)) -> list[dict]:
        with dbm.get_conn(app.state.db_path) as conn:
            return dbm.get_tubes(conn, row=row)

    @app.get("/api/v1/temperature-map")
    def temperature_map(ts_minute: Optional[int] = None,
                         _user: dict = Depends(current_user)) -> dict:
        with dbm.get_conn(app.state.db_path) as conn:
            readings = dbm.get_temperature_map(conn, ts_minute=ts_minute)
        # Compute average + hotspot identification (>30 °C above mean)
        temps = [r["wall_temp_c"] for r in readings]
        if temps:
            mean_t = sum(temps) / len(temps)
            hotspots = [
                {"tube_id": r["tube_id"], "wall_temp_c": r["wall_temp_c"],
                 "delta_c": round(r["wall_temp_c"] - mean_t, 2)}
                for r in readings if (r["wall_temp_c"] - mean_t) > 30
            ]
        else:
            mean_t = 0.0
            hotspots = []
        resolved_ts = ts_minute
        if resolved_ts is None and readings:
            resolved_ts = readings[0]["ts_minute"]
        return {
            "ts_minute": resolved_ts,
            "n_tubes": len(readings),
            "mean_wall_temp_c": round(mean_t, 2),
            "hotspots": hotspots,
            "readings": readings,
        }

    @app.get("/api/v1/rul")
    def rul(_user: dict = Depends(current_user)) -> dict:
        """Return RUL per tube using the trained Module 2 model.

        Falls back to physics estimate if model unavailable.
        Each tube carries a traffic light: red, orange, or green.
        """
        return _compute_rul(app.state.db_path)

    @app.post("/api/v1/optimization")
    def optimization(req: OptimizationReq, _user: dict = Depends(current_user)) -> dict:
        model = _get_model("module3_optimization.joblib")
        if model is None:
            raise HTTPException(status_code=503, detail="Optimization model not trained")
        current = {
            "sc_ratio": req.sc_ratio,
            "t_out_c": req.t_out_c,
            "feed_flow_nm3h": req.feed_flow_nm3h,
            "cat_age": req.cat_age,
            "fuel_flow_nm3h": req.fuel_flow_nm3h,
        }
        result = model.optimize(current, alpha_eff=req.alpha_eff, alpha_fuel=req.alpha_fuel)
        return result

    @app.get("/api/v1/carbon")
    def carbon(_user: dict = Depends(current_user)) -> dict:
        """Live burner CO2 plus furnace fuel and carbon intensity at the Module 3 optimum."""
        return _carbon_view(app.state.db_path)

    @app.get("/api/v1/plant-balance")
    def plant_balance(_user: dict = Depends(current_user)) -> dict:
        """Scale reformer specific energy onto Marjan's methanol and gas contract."""
        return _plant_balance(app.state.db_path)

    @app.get("/api/v1/production-plan")
    def production_plan(_user: dict = Depends(current_user)) -> dict:
        """Production rate under live prices, with dollar, gold and demand shocks."""
        return _production_plan(app.state.db_path)

    @app.get("/api/v1/catalyst-health")
    def catalyst_health(_user: dict = Depends(current_user)) -> dict:
        """Inspection signs, remaining catalyst life, and the purchase window."""
        return _catalyst_health(app.state.db_path)

    @app.get("/api/v1/benchmark")
    def benchmark(_user: dict = Depends(current_user)) -> dict:
        """Score the live furnace against published reformer, energy, and CO2 practice."""
        return _benchmark_view(app.state.db_path)

    @app.get("/api/v1/burners")
    def burners(_user: dict = Depends(current_user)) -> dict:
        """Return burner status with inefficiency flag from Module 4 if available.

        Uses the real average tube wall temperature (from the latest
        temperature_map) instead of a hardcoded 880 °C, so the burner
        inefficiency prediction reflects actual furnace conditions.
        """
        with dbm.get_conn(app.state.db_path) as conn:
            return _furnace_burners(conn)

    @app.get("/api/v1/maintenance")
    def maintenance(_user: dict = Depends(current_user)) -> dict:
        """Preventive-maintenance plan for tubes, burners, and the catalyst bed."""
        return _maintenance_view(app.state.db_path)

    @app.get("/api/v1/digital-twin")
    def digital_twin(_user: dict = Depends(current_user)) -> dict:
        """Live digital twin of the reformer line against the nominal model."""
        return _digital_twin(app.state.db_path)

    @app.get("/api/v1/reconciliation")
    def reconciliation(_user: dict = Depends(current_user)) -> dict:
        """Validate and reconcile the latest furnace sample."""
        return _reconciliation(app.state.db_path)

    @app.get("/api/v1/alerts")
    def alerts(status_filter: str = "open",
                _user: dict = Depends(current_user)) -> dict:
        with dbm.get_conn(app.state.db_path) as conn:
            rows = dbm.get_alerts(conn, status=status_filter)
        return {"alerts": rows, "n": len(rows)}

    @app.post("/api/v1/alerts/{alert_id}/ack")
    def ack(alert_id: int, _user: dict = Depends(current_user)) -> dict:
        with dbm.get_conn(app.state.db_path) as conn:
            ok = dbm.ack_alert(conn, alert_id, username=_user.get("sub", "system"))
        if not ok:
            raise HTTPException(status_code=404, detail="Alert not found or already acked")
        return {"ok": True, "alert_id": alert_id, "status": "acked"}

    @app.get("/api/v1/reports/daily")
    def daily_report(_user: dict = Depends(current_user)) -> dict:
        """Generate daily health report."""
        with dbm.get_conn(app.state.db_path) as conn:
            readings = dbm.get_temperature_map(conn)
            alerts = dbm.get_alerts(conn, status="open")
            tubes = dbm.get_tubes(conn)
        temps = [r["wall_temp_c"] for r in readings]
        return {
            "report_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "n_tubes": len(tubes),
            "n_readings": len(readings),
            "mean_temp_c": round(sum(temps) / len(temps), 2) if temps else None,
            "max_temp_c": round(max(temps), 2) if temps else None,
            "min_temp_c": round(min(temps), 2) if temps else None,
            "open_alerts": len(alerts),
            "n_hotspots": sum(1 for r in readings if any((r["wall_temp_c"] - t) > 30 for t in [sum(temps)/max(len(temps),1)])),
            "generated_at_epoch": int(time.time()),
        }

    @app.get("/api/v1/metrics",
             tags=["metrics"],
             summary="Model quality metrics (from JSON files)")
    def metrics(_user: dict = Depends(current_user)) -> dict:
        """Return model quality metrics from JSON files."""
        out = {}
        for name, path in [
            ("module1_temperature_map", os.path.join(METRICS_DIR, "module1_metrics.json")),
            ("module2_rul", os.path.join(METRICS_DIR, "module2_metrics.json")),
            ("module3_optimization", os.path.join(METRICS_DIR, "module3_metrics.json")),
            ("module4_burner_catalyst", os.path.join(METRICS_DIR, "module4_metrics.json")),
        ]:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if name == "module1_temperature_map" and isinstance(data, dict):
                        data.setdefault("rmse_c", data.get("rmse"))
                        data.setdefault("mae_c", data.get("mae"))
                    out[name] = data
                except Exception as e:
                    out[name] = {"error": str(e)}
            else:
                out[name] = None
        out["algorithms"] = _algorithm_rows(out)
        out["suite"] = _suite_report()
        return out

    @app.post("/api/v1/ingest/temperature",
              tags=["ingest"],
              summary="Ingest one temperature reading (manager/operator/it)",
              responses={403: {"description": "Insufficient role"},
                          422: {"description": "Out-of-range input"}})
    def ingest_temp(req: TemperatureIngest,
                     _user: dict = Depends(current_user)) -> dict:
        # RBAC: only manager/operator/it roles may ingest. Anonymous viewers
        # (dev fallback) are rejected even in dev mode — this is a write.
        if _user.get("role") not in ("manager", "operator", "it"):
            raise HTTPException(status_code=403, detail="Insufficient role for ingestion")
        with dbm.get_conn(app.state.db_path) as conn:
            # Auto-create tube if missing
            existing = conn.execute("SELECT 1 FROM tubes WHERE tube_id=?", (req.tube_id,)).fetchone()
            if existing is None:
                # derive row/col from tube_id pattern T<rr>-<ccc>
                try:
                    parts = req.tube_id.split("-")
                    r = int(parts[0][1:])
                    c = int(parts[1])
                except Exception:
                    r, c = 1, 1
                conn.execute(
                    "INSERT INTO tubes (tube_id, row, col, material, installed_at_days_ago, last_replaced_days_ago, design_life_h) VALUES (?,?,?,?,?,?,?)",
                    (req.tube_id, r, c, "HP-Modified", 200, 200, 100000.0)
                )
            # Auto-generate hotspot alert if temperature > mean + 30 (computed
            # from existing readings BEFORE this insert, to avoid self-reference)
            other = conn.execute(
                "SELECT AVG(wall_temp_c) FROM temperature_readings "
                "WHERE tube_id != ?", (req.tube_id,)
            ).fetchone()[0]
            overall = conn.execute("SELECT AVG(wall_temp_c) FROM temperature_readings").fetchone()[0]
            mean_ref = other if other is not None else (overall if overall is not None else 880.0)
            rid = dbm.insert_temperature_reading(conn, req.tube_id, req.ts_minute,
                                                  req.wall_temp_c, req.source)
            obs.inc_ingest()
            if req.wall_temp_c > mean_ref + 30:
                open_same = conn.execute(
                    "SELECT 1 FROM alerts WHERE tube_id=? AND module='module1' AND status='open'",
                    (req.tube_id,),
                ).fetchone()
                if open_same is None:
                    dbm.insert_alert(conn, req.tube_id, req.ts_minute, "high", "module1",
                                      f"Hotspot on {req.tube_id}: {req.wall_temp_c:.1f}°C")
        return {"ok": True, "reading_id": rid}

    @app.post("/api/v1/ingest/temperature/batch",
              tags=["ingest"],
              summary="Ingest a batch of temperature readings (manager/operator/it)")
    def ingest_batch(req: TemperatureBatch,
                      _user: dict = Depends(current_user)) -> dict:
        if _user.get("role") not in ("manager", "operator", "it"):
            raise HTTPException(status_code=403, detail="Insufficient role for ingestion")
        last_id = None
        with dbm.get_conn(app.state.db_path) as conn:
            for item in req.readings:
                existing = conn.execute(
                    "SELECT 1 FROM tubes WHERE tube_id=?", (item.tube_id,)
                ).fetchone()
                if existing is None:
                    try:
                        parts = item.tube_id.split("-")
                        row = int(parts[0][1:])
                        col = int(parts[1])
                    except Exception:
                        row, col = 1, 1
                    conn.execute(
                        "INSERT INTO tubes (tube_id, row, col, material, installed_at_days_ago, "
                        "last_replaced_days_ago, design_life_h) VALUES (?,?,?,?,?,?,?)",
                        (item.tube_id, row, col, "HP-Modified", 200, 200, 100000.0),
                    )
                other = conn.execute(
                    "SELECT AVG(wall_temp_c) FROM temperature_readings WHERE tube_id != ?",
                    (item.tube_id,),
                ).fetchone()[0]
                overall = conn.execute(
                    "SELECT AVG(wall_temp_c) FROM temperature_readings"
                ).fetchone()[0]
                mean_ref = other if other is not None else (overall if overall is not None else 880.0)
                last_id = dbm.insert_temperature_reading(
                    conn, item.tube_id, item.ts_minute, item.wall_temp_c, item.source
                )
                obs.inc_ingest()
                if item.wall_temp_c > mean_ref + 30:
                    open_same = conn.execute(
                        "SELECT 1 FROM alerts WHERE tube_id=? AND module='module1' AND status='open'",
                        (item.tube_id,),
                    ).fetchone()
                    if open_same is None:
                        dbm.insert_alert(
                            conn, item.tube_id, item.ts_minute, "high", "module1",
                            f"Hotspot on {item.tube_id}: {item.wall_temp_c:.1f}°C",
                        )
        return {"ok": True, "n": len(req.readings), "reading_id": last_id}

    # ── Phase 3: Prometheus metrics + deep health ─────────────────────
    @app.get("/metrics")
    def prometheus_metrics() -> Response:
        """Prometheus scrape endpoint. Exposed at /metrics (outside the
        /api/v1 prefix) per Prometheus convention so existing scrape configs
        work unchanged. No auth — protect at the network/ingress layer."""
        out = obs.metrics_response()
        if out is None:
            return Response(content=b"# prometheus_client not installed\n",
                            media_type="text/plain", status_code=501)
        body, ctype = out
        return Response(content=body, media_type=ctype)

    @app.get("/api/v1/health/deep",
             tags=["health"],
             summary="Deep health (DB, schema, models, disk) — k8s readiness")
    def health_deep() -> dict:
        """Deep health check — verifies DB connectivity, schema version,
        model availability, and disk space for the models dir. Used by
        k8s liveness/readiness probes (the shallow /health is for the
        Docker HEALTHCHECK and load-balancer pings)."""
        checks = {}
        # DB
        try:
            with dbm.get_conn(app.state.db_path) as conn:
                n = conn.execute("SELECT COUNT(*) FROM tubes").fetchone()[0]
                from mrto import migrations as mig
                sv = mig.get_schema_version(conn)
                checks["db"] = {"ok": True, "n_tubes": int(n),
                                "schema_version": sv}
        except Exception as e:
            checks["db"] = {"ok": False, "error": str(e)}
        # Models present (not loaded — that's a warmup concern)
        for name, fname in [("module2_rul", "module2_rul.joblib"),
                            ("module3_optimization", "module3_optimization.joblib"),
                            ("module4_burner_catalyst", "module4_burner_catalyst.joblib")]:
            p = os.path.join(MODELS_DIR, fname)
            checks[f"model_{name}"] = {"ok": os.path.exists(p),
                                        "path": p}
        # Disk space for models dir (best-effort)
        try:
            import shutil
            total, used, free = shutil.disk_usage(MODELS_DIR)
            checks["disk"] = {"ok": free > 100 * 1024 * 1024,
                              "free_mb": int(free / 1024 / 1024)}
        except Exception as e:
            checks["disk"] = {"ok": False, "error": str(e)}
        all_ok = all(v.get("ok") if isinstance(v, dict) else True
                     for v in checks.values())
        return {
            "status": "ok" if all_ok else "degraded",
            "service": "mrto",
            "version": "1.0.0",
            "schema_version": checks["db"].get("schema_version", 0)
                              if isinstance(checks.get("db"), dict) else 0,
            "checks": checks,
            "ts_epoch": int(time.time()),
        }

    # ── Model versioning (Phase 2) ──────────────────────────────────
    @app.get("/api/v1/models/versions",
             tags=["metrics"],
             summary="List model versions (manager/operator/it)",
             responses={403: {"description": "Insufficient role"}})
    def list_model_versions_endpoint(module: str | None = None,
                                       _user: dict = Depends(current_user)) -> dict:
        """List all registered model versions (optionally filtered by module).

        Only manager/operator/it roles may read this; operators on the plant
        floor only need the active model, not the audit trail of every retrain.
        """
        if _user.get("role") not in ("manager", "operator", "it"):
            raise HTTPException(status_code=403, detail="Insufficient role")
        with dbm.get_conn(app.state.db_path) as conn:
            rows = dbm.list_model_versions(conn, module=module)
        return {"versions": rows, "n": len(rows)}

    @app.get("/api/v1/models/{module}/active")
    def get_active_model_endpoint(module: str,
                                    _user: dict = Depends(current_user)) -> dict:
        """Return metadata (version, sha256, trained_at, metrics) for the
        currently-served model of a module. Lets the dashboard show which
        model version produced the RUL / optimization numbers the operator
        is seeing — critical for traceability and incident postmortems.
        """
        with dbm.get_conn(app.state.db_path) as conn:
            row = dbm.get_active_model_version(conn, module)
        if row is None:
            raise HTTPException(status_code=404, detail="No active model for this module")
        # Parse the cached metrics JSON for convenience.
        out = dict(row)
        try:
            out["metrics"] = json.loads(row["metrics_json"]) if row["metrics_json"] else None
        except Exception:
            out["metrics"] = None
        out.pop("metrics_json", None)
        return out

    @app.get("/")
    def root() -> dict:
        return {
            "name": "MRTO Backend",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    if os.environ.get("MRTO_LIVE_SIM") in ("1", "true", "True", "yes"):
        threading.Thread(
            target=_live_loop, args=(db_path,), name="mrto-live", daemon=True
        ).start()
    return app


# Module-level instance for `uvicorn mrto.app:app`
app = create_app()
