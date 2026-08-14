"""Build one synthetic census and write it to ``output/``.

Expects the INSEE inputs to sit in the ``data/`` layout documented in the
README (they are not distributed with this repository).

Usage
-----
    python run.py                       # YEAR / PERIMETER set below
    python run.py 2021                  # another year
    python run.py 2021 67 57            # year + department perimeter
"""

from pathlib import Path
import sys

from synthcensus import PipelineConfig, build_synthetic_census

ROOT = Path(__file__).resolve().parent  # repository root, independent of cwd

YEAR = 2021
REGIONS = None          # e.g. ["11"] for Ile-de-France
DEPTS = ["67", "57"]    # union with REGIONS; both None -> whole territory

if len(sys.argv) > 1:
    YEAR = int(sys.argv[1])
if len(sys.argv) > 2:
    DEPTS = sys.argv[2:]

year_dir = ROOT / "data" / str(YEAR)
config = PipelineConfig(year=YEAR, regions=REGIONS, depts=DEPTS)

synthetic = build_synthetic_census(
    census=year_dir / "census_individuals",
    mobpro=year_dir / "census_mobpro",
    mobsco=year_dir / "census_mobsco",
    family_margins=year_dir / "couples-familles-menages",
    iris_cantons=ROOT / "data" / "geo",
    config=config,
)

out_dir = ROOT / "output"
out_dir.mkdir(parents=True, exist_ok=True)
perimeter = "_".join(REGIONS or []) + "_".join(DEPTS or []) or "france"
out_path = out_dir / f"synthetic_census_{YEAR}_{perimeter}.parquet"
synthetic.write_parquet(out_path)
print(f"written: {out_path}")
