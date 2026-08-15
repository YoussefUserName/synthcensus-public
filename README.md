# synthcensus

Build a **synthetic individual census** in which every person receives:

- a **residence IRIS** (the finest INSEE geographic unit),
- a **work commune** (`DCLT`) or foreign-work code (`DCFLT`),
- a **study commune** (`DCETUF`) or foreign-study code (`DCETUE`).

It chains deterministic record linkage, controlled IRIS allocation against
household margins, and stochastic hot-deck imputation of the residual
commuting destinations. It works for **any census year** and **any
geographic perimeter**.

## Installation

```bash
git clone https://github.com/<user>/synthcensus.git
cd synthcensus
pip install -e .
```

Requires Python >= 3.10 (`polars`, `numpy`, `fastexcel`).

## Inputs

Five INSEE files (paths or in-memory `polars` frames):

| Argument | File | Role |
|---|---|---|
| `census` | RP individual file | base population (one row per person) |
| `mobpro` | MOBPRO | home-to-work flows |
| `mobsco` | MOBSCO | home-to-study flows |
| `family_margins` | BTX_IC_FAM (`.xls`) | household counts per IRIS x CSP group |
| `iris_cantons` | IRIS correspondence | IRIS -> commune / canton |

**None of these files are distributed here.** They are downloaded from INSEE
(the individual detail files and the IRIS margins carry their own conditions
of use). The `data/` tree ships empty, as the layout the scripts expect:

```
data/
  <year>/                        # 2007 ... 2022
    census_individuals/          # RP individual detail file  (.parquet)
    census_mobpro/               # MOBPRO                     (.parquet)
    census_mobsco/               # MOBSCO                     (.parquet)
    couples-familles-menages/    # BTX_IC_FAM margins         (.xls/.xlsx/.csv)
  geo/
    iris_cantons_<year>.parquet  # IRIS -> commune / canton / dept / region
```

Each input folder must contain **exactly one** file: passing the folder is
enough, the loaders resolve it (a single `.parquet`, except the margins which
may also be `.xls`/`.xlsx`/`.csv`). Passing the file path directly works too.

The `data/geo/` tables are rebuilt from two INSEE reference files with
[`scripts/build_iris_cantons.py`](scripts/build_iris_cantons.py).

## Usage

The **year is mandatory**; the perimeter is optional.

```python
import sys
from pathlib import Path

ROOT = Path("/Users/youssefelyaakoubi/Documents/URBAN_SIM/SIMULATOR/synthcensus")
sys.path.insert(0, str(ROOT))

from synthcensus import build_synthetic_census, PipelineConfig
import polars as pl

year = 2014
cfg = PipelineConfig(year=year, regions=["44"])
# cfg = PipelineConfig(year=year, depts=["67", "57"])

year_dir = ROOT / "data" / str(year)
synthetic = build_synthetic_census(census=year_dir / "census_individuals",
                                   mobpro=year_dir / "census_mobpro",
                                   mobsco=year_dir / "census_mobsco",
                                   family_margins=year_dir / "couples-familles-menages",
                                   iris

synthetic.write_parquet("synthetic_census_2008.parquet")
```

With the `data/` layout above, [`run.py`](run.py) does exactly this:

```bash
python run.py 2021 67 57     # year 2021, departments 67 and 57
```

and writes the result to `output/`. 

### Geographic perimeter (region and/or department, **union**)

`regions` and `depts` combine as a union: an individual is kept if its region
**or** its department is selected. This lets you mix a whole region with an
extra department:

```python
# All of Brittany (53) AND only department 67:
cfg = PipelineConfig(year=2008, regions=["53"], depts=["67"])
```

Leaving both `None` keeps the whole territory.

## Year handling

The year drives every INSEE name that changes across millésimes:

| Element | 2007-2021 | 2022+ |
|---|---|---|
| margin prefix | `C{yy}` (e.g. `C08`) | `C22` |
| CSP variable | `CS1` (8 posts) | `STAT_GSEC` regrouped to 8 |
| margin columns | `C{yy}_MEN_CS1..8` | `C22_MEN_STAT_GSEC11_21 .. _40` |
| reference-person CSP | `CSM` | `GSM` |
| `DIPL` | `DIPL` (2013-2016: `DIPL_15`) | `DIPL` |

For 2022, households are grouped on `STAT_GSEC` mapped to the historical
8-post split (`{11,21}->1, {12,22}->2, ..., {16,26}->6, 32->7, rest->8`), which
aligns with the `STAT_GSEC*` margin columns.

The **ARM arrondissement correction** (Paris/Lyon/Marseille) is applied to the
residence commune of MOBPRO/MOBSCO when `ARM != 'ZZZZZ'`, so arrondissements
are not collapsed onto a single commune code.

## Pipeline stages

Each stage prints progress and match/allocation rates.

1. **Workplace matching** — two-stage deterministic linkage to MOBPRO.
2. **Household variables** — reference-person CSP, student/pupil counts, age band.
3. **Study matching** — two-stage linkage to MOBSCO.
4. **IRIS allocation** — residual-weighted draw, then controlled selection.
5. **Destination imputation** — stochastic hot-deck with a 3-level fallback.

Each stage is importable on its own, so a single step can be run in isolation
(e.g. `from synthcensus.flow_matching import match_workplaces`).

## Modules

`config` (year schema + `PipelineConfig`), `utils`, `loaders`, `matching`,
`household_vars`, `flow_matching`, `iris_allocation`,
`destination_imputation`, `pipeline`.

## Reproducibility

All stochastic steps derive their seed from `PipelineConfig.root_seed`
(default 42). A fixed root seed reproduces the whole synthesis exactly.

## License

Code released under the [MIT License](LICENSE). This covers the code only:
the INSEE input files are not redistributed here and remain subject to their
own conditions of use.
