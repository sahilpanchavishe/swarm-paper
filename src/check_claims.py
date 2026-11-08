"""Number audit. paper/claims.csv columns: id, where, value, file, query, column, tol. Each claim must hit exactly one row. Run: python src/check_claims.py"""
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent; C = pd.read_csv(ROOT / "paper" / "claims.csv"); bad = 0
for _, r in C.iterrows():
    d = pd.read_csv(ROOT / r.file); x = d.query(r.query) if isinstance(r["query"], str) and r["query"] else d
    ok = len(x) == 1 and abs(float(x[r.column].iloc[0]) - float(r.value)) <= float(r.tol); bad += (not ok)
    print("OK  " if ok else "FAIL", r.id, "|", r["where"], "| text", r.value, "| table", x[r.column].iloc[0] if len(x) == 1 else f"{len(x)} rows")
print(f"\n{len(C) - bad}/{len(C)} claims verified"); raise SystemExit(1 if bad else 0)
