"""
Household-level derived variables used as extra MOBSCO matching keys.

The reference-person socio-professional category is named CSM before 2022 and
GSM from 2022 on; the count variables are built by conditional aggregation
inside the group_by so that households with a zero count keep a row (a 0,
never a null).
"""

from __future__ import annotations

import polars as pl

from .config import PipelineConfig
from .utils import log


def add_reference_person_csp(census: pl.DataFrame, config: PipelineConfig) -> pl.DataFrame:
    """Add the reference person's CSP (CSM pre-2022, GSM from 2022), taken from
    the household head (LPRM == '1') and broadcast to every member.

    The source value is the year's individual CSP variable (CS1 or GS), read on
    the reference person.
    """
    schema = config.schema
    ref_name = schema.ref_csp_var
    # Source variable on the reference person: GS in 2022, CS1 before.
    source_var = "GS" if schema.year >= 2022 else "CS1"
    if source_var not in census.columns:
        # Fall back to the configured csp variable if GS/CS1 is absent.
        source_var = schema.csp_var
    ref = (
        census.filter(pl.col("LPRM") == "1")
        .select("household_id", source_var)
        .rename({source_var: ref_name})
    )
    return census.join(ref, on="household_id", how="left")


def _household_count(census: pl.DataFrame, name: str, predicate: pl.Expr) -> pl.DataFrame:
    """Per-household count of members satisfying ``predicate`` (0 kept, not null)."""
    return (
        census.group_by("household_id")
        .agg(predicate.sum().cast(pl.Int64).alias(name))
        .with_columns(pl.col(name).cast(pl.Utf8))
    )


def add_student_count(census: pl.DataFrame) -> pl.DataFrame:
    """Add INPSM: number of enrolled members (ETUD == '1') per household."""
    counts = _household_count(census, "INPSM", pl.col("ETUD") == "1")
    return census.join(counts, on="household_id", how="left").with_columns(
        pl.col("INPSM").fill_null("0")
    )


def add_pupil_count(census: pl.DataFrame) -> pl.DataFrame:
    """Add INEEM: number of pupils/students aged 14+ (TACT == '22') per
    household. TACT == '22' is valid across all years, including 2022."""
    counts = _household_count(census, "INEEM", pl.col("TACT") == "22")
    return census.join(counts, on="household_id", how="left").with_columns(
        pl.col("INEEM").fill_null("0")
    )


def add_age_band(census: pl.DataFrame, column: str = "AGEREV10") -> pl.DataFrame:
    """Add the 10-class MOBSCO age band derived from AGED. Idempotent."""
    if column in census.columns:
        return census
    return (
        census.with_columns(_age=pl.col("AGED").cast(pl.Int64))
        .with_columns(
            **{
                column: pl.when(pl.col("_age").is_between(0, 2)).then(pl.lit("00"))
                .when(pl.col("_age") == 3).then(pl.lit("03"))
                .when(pl.col("_age") == 4).then(pl.lit("04"))
                .when(pl.col("_age") == 5).then(pl.lit("05"))
                .when(pl.col("_age").is_between(6, 10)).then(pl.lit("06"))
                .when(pl.col("_age").is_between(11, 14)).then(pl.lit("11"))
                .when(pl.col("_age").is_between(15, 17)).then(pl.lit("15"))
                .when(pl.col("_age").is_between(18, 24)).then(pl.lit("18"))
                .when(pl.col("_age").is_between(25, 29)).then(pl.lit("25"))
                .when(pl.col("_age") >= 30).then(pl.lit("30"))
                .otherwise(pl.lit("Z"))
            }
        )
        .drop("_age")
    )


def add_mobsco_matching_variables(census: pl.DataFrame, config: PipelineConfig) -> pl.DataFrame:
    """Add every household-derived MOBSCO matching key: reference-person CSP
    (CSM/GSM), student and pupil counts, age band."""
    census = add_reference_person_csp(census, config)
    census = add_student_count(census)
    census = add_pupil_count(census)
    census = add_age_band(census)
    log(f"  household matching variables added ({config.schema.ref_csp_var}, INPSM, INEEM, AGEREV10)")
    return census
