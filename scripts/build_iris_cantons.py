"""Build the ``data/geo/iris_cantons_<year>.parquet`` correspondence table.

The pipeline expects, for every millesime, an IRIS -> commune / canton /
department / region table. It is not distributed here; this script rebuilds it
from two INSEE reference files, which must be downloaded first into
``data/tables_correspondance_geo/``:

* ``communes/table-appartenance-geo-communes-<yy>.xlsx``
  (sheets ``COM`` and ``ARM``: commune -> canton-ville ``CANOV``),
* ``iris/reference_IRIS_geo<year>.xlsx``
  (IRIS -> commune ``DEPCOM``, department, region).

Usage
-----
    python scripts/build_iris_cantons.py 2022
    python scripts/build_iris_cantons.py 2007 2008 2009
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
GEO_SOURCES = ROOT / "data" / "tables_correspondance_geo"
GEO_OUT = ROOT / "data" / "geo"


def build_year(year: int) -> Path:
    yy = f"{year % 100:02d}"

    communes_file = GEO_SOURCES / "communes" / f"table-appartenance-geo-communes-{yy}.xlsx"
    # ARM carries the Paris/Lyon/Marseille arrondissements, absent from COM;
    # without them those IRIS would have no canton.
    corr = pl.concat(
        [
            pl.read_excel(communes_file, sheet_name="COM"),
            pl.read_excel(communes_file, sheet_name="ARM"),
        ],
        how="vertical",
    ).select(
        pl.col("CODGEO").alias("INSEE_COM"),
        pl.col("CANOV").alias("CANTON"),
    )

    iris = (
        pl.read_excel(GEO_SOURCES / "iris" / f"reference_IRIS_geo{year}.xlsx")
        .rename({"DEPCOM": "INSEE_COM"})
        .join(corr, on="INSEE_COM", how="left")
    )

    n_missing = iris.filter(pl.col("CANTON").is_null()).height
    if n_missing:
        print(f"  {year}: {n_missing} IRIS without a canton")

    out_path = GEO_OUT / f"iris_cantons_{year}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    iris.select("CODE_IRIS", "INSEE_COM", "REG", "DEP", "CANTON").write_parquet(out_path)
    print(f"  {year}: {iris.height} IRIS -> {out_path}")
    return out_path


if __name__ == "__main__":
    years = [int(a) for a in sys.argv[1:]]
    if not years:
        raise SystemExit("usage: python scripts/build_iris_cantons.py <year> [<year> ...]")
    for y in years:
        build_year(y)
