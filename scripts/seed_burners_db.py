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
import sqlite3
import sys

DEFAULT_DB = os.environ.get("MRTO_DB_PATH",
                            os.path.join(os.path.dirname(__file__),
                                         "..", "backend", "data", "mrto.sqlite3"))


def main():
    ap = argparse.ArgumentParser(description="Seed burner_status records into MRTO SQLite DB")
    ap.add_argument("--db", default=DEFAULT_DB, help="Path to SQLite DB file")
    ap.add_argument("--minutes", type=int, default=30, help="Number of minutes to seed")
    ap.add_argument("--burners", type=int, default=8, help="Number of burners (rows × 2 sides)")
    ap.add_argument("--base-ts", type=int, default=1000, help="Base ts_minute value")
    args = ap.parse_args()

    db_path = os.path.abspath(args.db)
    if not os.path.exists(os.path.dirname(db_path)):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # Build burner IDs: B{r:02d}-{L|R}, r=1..N/2
    n_rows = max(1, args.burners // 2)
    burners = [f"B{r:02d}-{s}" for r in range(1, n_rows + 1) for s in ("L", "R")]
    if len(burners) > args.burners:
        burners = burners[:args.burners]

    random.seed(42)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Ensure table exists (idempotent — matches backend/mrto/db.py schema)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS burner_status (
            burner_id TEXT NOT NULL,
            ts_minute INTEGER NOT NULL,
            fuel_flow REAL,
            intensity REAL,
            angle REAL,
            PRIMARY KEY (burner_id, ts_minute)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_burner_ts ON burner_status(burner_id, ts_minute)")

    rows = []
    for minute in range(args.minutes):
        for bid in burners:
            r, s = bid.split("-")
            base_int = 0.92 if s == "L" else 0.88
            # First-row left & third-row right deliberately degraded → inefficient
            if bid == "B01-L":
                base_int = 0.55
            elif bid == "B03-R":
                base_int = 0.40
            intensity = base_int + random.uniform(-0.05, 0.05)
            fuel = (11000 + random.uniform(-200, 200)) if base_int > 0.7 else (8500 + random.uniform(-200, 200))
            angle = random.choice([0, 30, 60, 90, 120])
            rows.append((bid, args.base_ts + minute, round(fuel, 1),
                         round(intensity, 3), angle))

    conn.executemany(
        "INSERT OR REPLACE INTO burner_status (burner_id, ts_minute, fuel_flow, intensity, angle) "
        "VALUES (?,?,?,?,?)", rows
    )
    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM burner_status").fetchone()[0]
    conn.close()

    print(f"✓ Inserted {len(rows)} burner readings ({args.minutes} min × {len(burners)} burners)")
    print(f"  Total rows in burner_status: {n}")
    print(f"  DB: {db_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
