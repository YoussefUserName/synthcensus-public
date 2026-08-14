"""
synthcensus
===========

Build a synthetic individual census in which every person is assigned a
residence IRIS, a work commune and a study commune, from five INSEE 2007
inputs:

1. the individual census (RP),
2. the home-to-work file (MOBPRO),
3. the home-to-study file (MOBSCO),
4. the couples-families-households margins (BTX_IC_FAM),
5. the IRIS-to-canton correspondence table.

Typical use
-----------
>>> from synthcensus import build_synthetic_census
>>> synthetic = build_synthetic_census(
...     census="RP_2007_GRAND_EST.parquet",
...     mobpro="FD_MOBPRO_2007.parquet",
...     mobsco="FD_MOBSCO_2007.parquet",
...     family_margins="BTX_IC_FAM_2007.xls",
...     iris_cantons="iris_canton_france_2007.parquet",
... )
"""

from .config import PipelineConfig
from .pipeline import build_synthetic_census

__all__ = ["build_synthetic_census", "PipelineConfig"]
__version__ = "0.1.0"
