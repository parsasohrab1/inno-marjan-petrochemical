#!/usr/bin/env python3
"""Seed burner_status records directly into the MRTO SQLite DB so the
dashboard's burner panel shows live content. The MRTO API has no public
endpoint for burner ingestion, so we write to SQLite directly.

Usage:
    python seed_burners_db.py [--db PATH] [--minutes N] [--burners N]

Env:
    MRTO_DB_PATH  (default: ./data/mrto.sqlite3)
"""
import argparse
import os
import random
import sys

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from mrto import db as dbm

DEFAULT_DB = os.environ.get(
    "MRTO_DB_PATH",
    os.path.join(BACKEND, "data", "mrto.sqlite3"),
)


def main():
    ap = argparse.ArgumentParser(description="Seed burner_status records into MRTO SQLite DB")
    ap.add_argument("--db", default=DEFAULT_DB, help="Path to SQLite DB file")
    ap.add_argument("--minutes", type=int, default=30, help="Number of minutes to seed")
    ap.add_argument("--burners", type=int, default=8, help="Number of burners (rows × 2 sides)")
    ap.add_argument("--base-ts", type=int, default=1000, help="Base ts_minute value")
    args = ap.parse_args()

    db_path = os.path.abspath(args.db)
    dbm.init_db(db_path)

    # Build burner IDs: B{r:02d}-{L|R}. Fuel and angle stay inside the
    # Module 4 training range (fuel ~120–280 Nm³/h, angle ~±15°).
    n_rows = max(1, args.burners // 2)
    burners = [f"B{r:02d}-{s}" for r in range(1, n_rows + 1) for s in ("L", "R")]
    if len(burners) > args.burners:
        burners = burners[:args.burners]

    random.seed(42)
    degraded = {"B01-L": 0.52, "B03-R": 0.40}
    rows = []
    for minute in range(args.minutes):
        for bid in burners:
            side = bid.split("-")[1]
            if bid in degraded:
                intensity = degraded[bid] + random.uniform(-0.02, 0.02)
                fuel = 250 + random.uniform(-10, 10)
            else:
                intensity = (0.96 if side == "L" else 0.93) + random.uniform(-0.02, 0.02)
                fuel = 190 + random.uniform(-8, 8)
            angle = random.uniform(-8, 8)
            rows.append((bid, args.base_ts + minute, round(fuel, 1),
                         round(intensity, 3), round(angle, 2)))

    with dbm.get_conn(db_path) as conn:
        conn.executemany(
            "INSERT INTO burner_status (burner_id, ts_minute, fuel_flow, intensity, angle) "
            "VALUES (?,?,?,?,?)",
            rows,
        )
        n = conn.execute("SELECT COUNT(*) FROM burner_status").fetchone()[0]

    print(f"✓ Inserted {len(rows)} burner readings ({args.minutes} min × {len(burners)} burners)")
    print(f"  Total rows in burner_status: {n}")
    print(f"  DB: {db_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
