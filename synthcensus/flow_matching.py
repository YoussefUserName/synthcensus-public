"""
Transfer of work and study communes from the flow files.

Both MOBPRO (work) and MOBSCO (study) are matched to the census in two
stages with the same logic:

* **Stage 1 - commune key.** The residence commune is part of the matching
  key, so a link is accepted only between an individual and a flow record
  sharing the same commune (and every other common variable).
* **Stage 2 - canton key.** Individuals left unmatched are retried after
  attaching their canton to the flow file, relaxing the geographic key from
  commune to canton.

Whenever a flow record carries a residence commune that the individual was
missing (the individual's IRIS was unknown), that commune is propagated to
every member of the household.
"""

from __future__ import annotations

import polars as pl

from .config import IRIS_UNKNOWN, PipelineConfig
from .matching import double_merge, get_match_cols
from .utils import log, section

# Columns that must never be used as matching keys (identifiers, outputs,
# weight handled separately, geography rebuilt downstream).
_NON_KEY_COLS = ["IPONDI", "DCLT", "DCFLT", "COMMUNE_mp"]


def _propagate_commune_to_household(
    census: pl.DataFrame, gained_commune_col: str, target_commune_col: str
) -> pl.DataFrame:
    """Propagate a newly-found residence commune to every member of the
    household, for households whose IRIS is still unknown.

    Different members of the same household can be matched to flow records
    that imply different residence communes (an ambiguous/imperfect match).
    Keeping every distinct candidate would make the join below a
    one-to-many join and duplicate the whole household, so only the most
    frequent candidate commune is kept per household (ties broken
    arbitrarily but deterministically)."""
    gained = (
        census.filter(pl.col(gained_commune_col).is_not_null(), pl.col("IRIS") == IRIS_UNKNOWN)
        .select("household_id", gained_commune_col)
        .group_by("household_id", gained_commune_col)
        .agg(pl.len().alias("_n"))
        .sort("_n", descending=True)
        .unique("household_id", keep="first", maintain_order=True)
        .drop("_n")
        .rename({gained_commune_col: "_commune_new"})
    )
    return (
        census.join(gained, on="household_id", how="left")
        .with_columns(
            **{
                target_commune_col: pl.when(
                    pl.col(gained_commune_col).is_null(), pl.col("_commune_new").is_not_null()
                )
                .then(pl.col("_commune_new"))
                .otherwise(pl.col(target_commune_col))
            }
        )
        .drop("_commune_new")
    )


def match_workplaces(
    census: pl.DataFrame,
    mobpro: pl.DataFrame,
    canton_commune: pl.DataFrame,
    config: PipelineConfig,
) -> pl.DataFrame:
    """Attach work commune (DCLT) and foreign-work flag (DCFLT) from MOBPRO,
    and back-fill residence communes gained from the match.

    Returns the census with ``DCLT``, ``DCFLT`` and a rebuilt ``COMMUNE``.
    """
    section("MOBPRO - workplace matching (DCLT)")
    extra_cols = ["DCLT", "DCFLT", "COMMUNE_mp"]

    mobpro_base = (
        mobpro.with_columns(DEPT=pl.col("COMMUNE").str.slice(0, 2))
        .rename({"COMMUNE": "COMMUNE_mp"})
    )

    # --- Stage 1: commune derived from IRIS is part of the key.
    stage1 = census.with_columns(COMMUNE=pl.col("IRIS").str.slice(0, 5))
    keys1 = get_match_cols(stage1, mobpro_base, exclude=extra_cols + ["IPONDI"])
    merged1 = double_merge(stage1, mobpro_base, keys1, extra_cols)
    first = merged1.filter(pl.col("DCLT").is_not_null())
    matched_ids = first["person_id"]
    log(f"  stage 1 (commune key)  - matched: {first.height:>8}")

    # --- Stage 2: canton key for the still-unmatched individuals.
    stage2 = census.filter(~pl.col("person_id").is_in(matched_ids))
    mobpro_canton = (
        mobpro_base.rename({"COMMUNE_mp": "COMMUNE"})
        .join(canton_commune, on="COMMUNE", how="left")
        .rename({"COMMUNE": "COMMUNE_mp"})
    )
    keys2 = get_match_cols(stage2, mobpro_canton, exclude=extra_cols + ["IPONDI"])
    merged2 = double_merge(stage2, mobpro_canton, keys2, extra_cols)
    second = merged2.filter(pl.col("DCLT").is_not_null())
    log(f"  stage 2 (canton key)   - matched: {second.height:>8}")

    assignment = (
        pl.concat(
            [first.select(["person_id", *extra_cols]), second.select(["person_id", *extra_cols])],
            how="vertical",
        ).unique("person_id")
    )
    census = census.join(assignment, on="person_id", how="left")

    # Back-fill residence commune gained from MOBPRO to whole households.
    census = _propagate_commune_to_household(census, "COMMUNE_mp", "COMMUNE_mp")

    # Rebuild COMMUNE: 5-digit prefix of IRIS, or the MOBPRO commune when the
    # IRIS was unknown.
    census = census.with_columns(COMMUNE=pl.col("IRIS").str.slice(0, 5)).with_columns(
        COMMUNE=pl.when(pl.col("IRIS") == IRIS_UNKNOWN, pl.col("COMMUNE_mp").is_not_null())
        .then(pl.col("COMMUNE_mp"))
        .otherwise(pl.col("COMMUNE"))
    )

    workers = census.filter(pl.col("TACT") == "11")
    n_workers = workers.height
    n_workers_matched = workers.filter(pl.col("DCLT").is_not_null()).height
    log(f"  workers (TACT=11)      : {n_workers:>8}")
    pct_w = n_workers_matched / n_workers * 100 if n_workers else 0.0
    log(f"  workers matched        : {n_workers_matched:>8}  ({pct_w:.1f}%)")
    return census


def match_study_places(
    census: pl.DataFrame,
    mobsco: pl.DataFrame,
    canton_commune: pl.DataFrame,
    config: PipelineConfig,
) -> pl.DataFrame:
    """Attach study commune (DCETUF) and foreign-study flag (DCETUE) from
    MOBSCO, using the same two-stage logic, then back-fill communes.

    Requires the household-derived matching variables (CSM/GSM, INPSM, AGEREV10)
    to already be present on ``census``.
    """
    section("MOBSCO - study-place matching (DCETUF)")
    extra_cols = ["DCETUF", "DCETUE", "COMMUNE_ms"]
    exclude = extra_cols + ["IPONDI", "DCLT", "DCFLT", "COMMUNE_mp", "AGEREV10"]

    mobsco_base = mobsco.with_columns(
        DEPT=pl.col("COMMUNE").str.slice(0, 2), COMMUNE_ms=pl.col("COMMUNE")
    )

    # --- Stage 1: commune key (census already carries a real COMMUNE here).
    keys1 = get_match_cols(census, mobsco_base, exclude=exclude)
    merged1 = double_merge(census, mobsco_base, keys1, extra_cols)
    first = merged1.filter(pl.col("DCETUF").is_not_null())
    matched_ids = first["person_id"]
    log(f"  stage 1 (commune key)  - matched: {first.height:>8}")

    # --- Stage 2: canton key for the unmatched.
    stage2 = census.filter(~pl.col("person_id").is_in(matched_ids))
    mobsco_canton = (
        mobsco_base.drop("COMMUNE_ms")
        .join(canton_commune, on="COMMUNE", how="left")
        .rename({"COMMUNE": "COMMUNE_ms"})
    )
    keys2 = get_match_cols(stage2, mobsco_canton, exclude=exclude)
    merged2 = double_merge(stage2, mobsco_canton, keys2, extra_cols)
    second = merged2.filter(pl.col("DCETUF").is_not_null())
    log(f"  stage 2 (canton key)   - matched: {second.height:>8}")

    assignment = (
        pl.concat(
            [first.select(["person_id", *extra_cols]), second.select(["person_id", *extra_cols])],
            how="vertical",
        ).unique("person_id")
    )
    census = census.join(assignment, on="person_id", how="left")

    # Back-fill residence commune gained from MOBSCO, then drop the helper.
    census = _propagate_commune_to_household(census, "COMMUNE_ms", "COMMUNE")
    census = census.drop("COMMUNE_ms")

    students = census.filter(pl.col("ETUD") == "1")
    n_students = students.height
    n_students_matched = students.filter(pl.col("DCETUF").is_not_null()).height
    log(f"  students (ETUD=1)      : {n_students:>8}")
    pct_s = n_students_matched / n_students * 100 if n_students else 0.0
    log(f"  students matched       : {n_students_matched:>8}  ({pct_s:.1f}%)")
    return census
