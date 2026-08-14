"""
End-to-end orchestration.

``build_synthetic_census`` chains every stage and returns the completed
synthetic census: one row per individual, each with a residence IRIS, a work
destination (DCLT/DCFLT) and a study destination (DCETUF/DCETUE).
"""

from __future__ import annotations

from typing import Optional

import polars as pl

from .config import COMMUNE_UNKNOWN, IRIS_UNKNOWN, PipelineConfig
from .destination_imputation import impute_destinations
from .flow_matching import match_study_places, match_workplaces
from .household_vars import add_mobsco_matching_variables
from .iris_allocation import allocate_iris
from .loaders import (
    FrameOrPath,
    build_canton_commune,
    load_census,
    load_family_margins,
    load_iris_cantons,
    load_mobpro,
    load_mobsco,
    load_regions_2016,
)
from .utils import log, section


def _log_known_commune(census_df: pl.DataFrame) -> None:
    """Log how many reference-person households have a known residence
    commune (either directly from IRIS, or back-filled from MOBPRO/MOBSCO)."""
    total = census_df.filter(pl.col("LPRM") == "1").height
    if "COMMUNE" in census_df.columns:
        cond = pl.col("COMMUNE") != IRIS_UNKNOWN[:5]
    else:
        cond = pl.col("IRIS") != IRIS_UNKNOWN
    known = census_df.filter(pl.col("LPRM") == "1").filter(cond).height
    pct = (known / total * 100) if total else 0.0
    log(f"\nhouseholds with known COMMUNE : {known} / {total} ({pct:.0f}%)\n")


def _internal_columns(config) -> list[str]:
    """Helper columns used only for matching/allocation, removed from output.
    The reference-person CSP name and the margin-group name depend on the year.
    """
    return [
        "COMMUNE_mp",
        config.schema.ref_csp_var,   # CSM (pre-2022) or GSM (2022+)
        "INPSM",
        "INEEM",
        "AGEREV10",
        config.schema.group_label_column,  # MARGIN_GROUP
    ]


_DROPPED_OUTPUT_COLUMNS = ["TRIRIS", "ARM", "CATIRIS"]
_LEADING_OUTPUT_COLUMNS = [
    "household_id", "person_id", "CANTVILLE", "COMMUNE", "IRIS",
    "DCLT", "DCFLT", "DCETUF", "DCETUE",
]
_TRAILING_OUTPUT_COLUMNS = [
    "iris_from_census", "iris_from_mobpro", "iris_from_mobsco", "iris_from_canton",
    "dclt_from_mobpro", "dclt_from_ilt",
    "dcet_from_mobsco", "dcet_from_iletud",
]


def _finalize_output(census_df: pl.DataFrame) -> pl.DataFrame:
    """Drop unused geographic codes, cast ``person_id`` to string, and order
    columns with the key identifiers/destinations first and the provenance
    flags last."""
    present_drops = [c for c in _DROPPED_OUTPUT_COLUMNS if c in census_df.columns]
    census_df = census_df.drop(present_drops)
    census_df = census_df.with_columns(pl.col("person_id").cast(pl.String))
    leading = [c for c in _LEADING_OUTPUT_COLUMNS if c in census_df.columns]
    trailing = [c for c in _TRAILING_OUTPUT_COLUMNS if c in census_df.columns]
    rest = [c for c in census_df.columns if c not in leading and c not in trailing]
    return census_df.select(leading + rest + trailing)


def build_synthetic_census(
    census: FrameOrPath,
    mobpro: FrameOrPath,
    mobsco: FrameOrPath,
    family_margins: FrameOrPath,
    iris_cantons: FrameOrPath,
    config: Optional[PipelineConfig] = None,
    family_margins_sheet: str = "IRIS",
    drop_internal_columns: bool = True,
) -> pl.DataFrame:
    """Build the synthetic census from the five INSEE inputs.

    Parameters
    ----------
    census, mobpro, mobsco, family_margins, iris_cantons : path or DataFrame
        for each of the five inputs.
    config : optional :class:`PipelineConfig`; defaults to the Grand Est
        configuration.
    family_margins_sheet : worksheet name for the Excel margins file.
    drop_internal_columns : drop the matching-only helper columns from the
        result (default True).

    Returns
    -------
    polars.DataFrame
        The synthetic census.
    """
    config = config or PipelineConfig()

    section("Loading inputs")
    regions_2016_df = load_regions_2016(iris_cantons)
    census_df = load_census(census, config, regions_2016_df)
    mobpro_df = load_mobpro(census_df, mobpro, config, regions_2016_df)
    mobsco_df = load_mobsco(census_df, mobsco, config, regions_2016_df)
    iris_cantons_df = load_iris_cantons(census_df, iris_cantons, config)
    margins_df = load_family_margins(family_margins, config, family_margins_sheet)
    canton_commune = build_canton_commune(iris_cantons_df)

    # Provenance tracking: iris_from_census fixes the baseline; _commune_source
    # records which flow file (if any) first resolved an unknown commune, for
    # _tag_iris_provenance (in allocate_iris) to turn into the IRIS flags.
    census_df = census_df.with_columns(
        iris_from_census=pl.col("IRIS") != IRIS_UNKNOWN,
        _iris_unknown_before=pl.col("IRIS") == IRIS_UNKNOWN,
        _commune_source=pl.lit(None, dtype=pl.String),
    )

    _log_known_commune(census_df)

    # 1. Workplace matching (MOBPRO) -> DCLT/DCFLT + commune back-fill.
    census_df = match_workplaces(census_df, mobpro_df, canton_commune, config)
    census_df = census_df.with_columns(
        dclt_from_mobpro=pl.when(pl.col("TACT") == "11")
        .then(pl.col("DCLT").is_not_null() | pl.col("DCFLT").is_not_null())
        .otherwise(None),
        _commune_source=pl.when(
            pl.col("_commune_source").is_null()
            & pl.col("_iris_unknown_before")
            & (pl.col("COMMUNE") != IRIS_UNKNOWN[:5])
        )
        .then(pl.lit("mobpro"))
        .otherwise(pl.col("_commune_source")),
    )

    _log_known_commune(census_df)

    # 2. Household-derived variables, then study matching (MOBSCO) -> DCETUF/DCETUE.
    census_df = add_mobsco_matching_variables(census_df, config)
    census_df = match_study_places(census_df, mobsco_df, canton_commune, config)
    census_df = census_df.with_columns(
        dcet_from_mobsco=pl.when(pl.col("ETUD") == "1")
        .then(pl.col("DCETUF").is_not_null() | pl.col("DCETUE").is_not_null())
        .otherwise(None),
        _commune_source=pl.when(
            pl.col("_commune_source").is_null()
            & pl.col("_iris_unknown_before")
            & (pl.col("COMMUNE") != IRIS_UNKNOWN[:5])
        )
        .then(pl.lit("mobsco"))
        .otherwise(pl.col("_commune_source")),
    )

    _log_known_commune(census_df)

    # 3. IRIS allocation for households still missing an IRIS (also tags
    # iris_from_mobpro/iris_from_mobsco/iris_from_canton and drops the
    # _commune_source helper).
    census_df = allocate_iris(census_df, iris_cantons_df, margins_df, config)
    census_df = census_df.drop("_iris_unknown_before")

    # 4. Stochastic imputation of the residual work/study destinations.
    census_df = impute_destinations(census_df, mobpro_df, mobsco_df, config, regions_2016_df)
    census_df = census_df.with_columns(
        dclt_from_ilt=pl.when(pl.col("TACT") == "11")
        .then((~pl.col("dclt_from_mobpro")) & (pl.col("DCLT") != COMMUNE_UNKNOWN))
        .otherwise(None),
        dcet_from_iletud=pl.when(pl.col("ETUD") == "1")
        .then((~pl.col("dcet_from_mobsco")) & (pl.col("DCETUF") != COMMUNE_UNKNOWN))
        .otherwise(None),
    )

    if drop_internal_columns:
        present = [c for c in _internal_columns(config) if c in census_df.columns]
        census_df = census_df.drop(present)

    census_df = _finalize_output(census_df)

    section("Done")
    print(f"  synthetic census: {census_df.height} individuals, {len(census_df.columns)} columns")
    return census_df
