"""
Allocation of households to IRIS units.

Households whose commune is known are drawn into one of that commune's IRIS
(probability proportional to residual capacity); households known only at the
canton level are placed by controlled selection within their canton, per
margin group, on the residual group margins.

The grouping variable is the year-independent ``MARGIN_GROUP`` (labels
'1'..'8') derived in the loaders, and the margin columns are taken from the
year's schema, so this stage is identical across millésimes.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .config import IRIS_UNKNOWN, MARGIN_GROUPS, PipelineConfig
from .utils import log, section, weight_to_float


def _current_load_by_iris(census: pl.DataFrame) -> pl.DataFrame:
    """Located weight per IRIS, counted once per household (reference person)."""
    return (
        census.filter(pl.col("IRIS") != IRIS_UNKNOWN, pl.col("LPRM") == "1")
        .group_by("IRIS")
        .agg(pl.col("_w").sum().alias("load"))
    )


def _allocate_stage1(census, iris_cantons, margins, config, rng):
    """Place commune-known households into a commune IRIS by residual-weighted
    random draw."""
    schema = config.schema
    iris_commune = iris_cantons.select("IRIS", "COMMUNE").unique()
    total_margin = margins.select(
        "IRIS", pl.col(schema.total_margin_column).cast(pl.Float64).alias("men_total")
    )
    load = _current_load_by_iris(census)

    candidates = (
        iris_commune.join(total_margin, on="IRIS", how="left")
        .join(load, on="IRIS", how="left")
        .with_columns(pl.col("men_total").fill_null(0.0), pl.col("load").fill_null(0.0))
        .with_columns(
            residual=(pl.col("men_total") * config.margin_tolerance - pl.col("load")).clip(lower_bound=0.0)
        )
    )
    n_iris = candidates.group_by("COMMUNE").agg(pl.len().alias("n_iris"))
    candidates = candidates.join(n_iris, on="COMMUNE", how="left")
    candidates = candidates.filter((pl.col("n_iris") == 1) | (pl.col("residual") > 0)).drop("n_iris")

    targets = census.filter(
        pl.col("LPRM") == "1",
        pl.col("IRIS") == IRIS_UNKNOWN,
        pl.col("COMMUNE").is_not_null(),
        pl.col("COMMUNE") != IRIS_UNKNOWN[:5],
    ).select("household_id", "COMMUNE", "_w")

    assignments: list[dict] = []
    n_no_candidate = 0
    n_communes = n_communes_single = 0
    n_households_single = 0
    for commune in targets["COMMUNE"].unique().to_list():
        households = targets.filter(pl.col("COMMUNE") == commune)
        cand = candidates.filter(pl.col("COMMUNE") == commune)
        if len(cand) == 0:
            n_no_candidate += len(households)
            continue
        n_communes += 1
        iris_codes = cand["IRIS"].to_list()
        if len(iris_codes) == 1:
            n_communes_single += 1
            n_households_single += len(households)
            chosen = [iris_codes[0]] * len(households)
        else:
            weights = cand["residual"].to_numpy().astype(float)
            probs = weights / weights.sum() if weights.sum() > 0 else np.ones(len(iris_codes)) / len(iris_codes)
            chosen = rng.choice(iris_codes, size=len(households), p=probs).tolist()
        for hid, iris in zip(households["household_id"].to_list(), chosen):
            assignments.append({"household_id": hid, "IRIS_NEW": iris})

    if n_no_candidate:
        log(f"  stage 1: commune known but no IRIS candidate - households: {n_no_candidate}")
    if n_communes:
        pct_communes = 100 * n_communes_single / n_communes
        pct_households = 100 * n_households_single / len(assignments) if assignments else 0.0
        log(
            f"  stage 1: single-IRIS communes (deterministic 1-to-1): "
            f"{n_communes_single}/{n_communes} communes ({pct_communes:.1f}%) \n"
            f"{n_households_single}/{len(assignments)} households ({pct_households:.1f}%)"
        )

    return _apply_assignments(census, assignments)


def _apply_assignments(census, assignments):
    """Join a household->IRIS assignment list, overwrite IRIS where unknown.
    Returns (census, assignment_df)."""
    df = (
        pl.DataFrame(assignments)
        if assignments
        else pl.DataFrame(schema={"household_id": pl.Utf8, "IRIS_NEW": pl.Utf8})
    )
    census = (
        census.join(df, on="household_id", how="left")
        .with_columns(
            IRIS=pl.when((pl.col("IRIS") == IRIS_UNKNOWN) & pl.col("IRIS_NEW").is_not_null())
            .then(pl.col("IRIS_NEW"))
            .otherwise(pl.col("IRIS"))
        )
        .drop("IRIS_NEW")
    )
    return census, df


def _residual_margins(census, margins, iris_cantons, config):
    """Residual group margin per IRIS = full group margin minus located weight,
    floored at zero, with canton attached. Group labels are '1'..'8'."""
    schema = config.schema
    group_col = schema.group_label_column
    group_to_col = schema.group_to_margin_column()  # '1'->col, ..., '8'->col

    load_grp = (
        census.filter(pl.col("IRIS") != IRIS_UNKNOWN, pl.col("LPRM") == "1")
        .group_by(["IRIS", group_col])
        .agg(pl.col("_w").sum().alias("load"))
    )

    # Long margins: one row per (IRIS, group label, margin value).
    margins_long = (
        margins.select(["IRIS", *schema.margin_columns])
        .unpivot(index="IRIS", variable_name="margin_col", value_name="margin")
        .with_columns(margin=pl.col("margin").cast(pl.Float64))
    )
    col_to_group = {col: grp for grp, col in group_to_col.items()}
    margins_long = margins_long.with_columns(
        pl.col("margin_col").replace_strict(col_to_group, default=None).alias(group_col)
    ).drop("margin_col")

    return (
        margins_long.join(load_grp, on=["IRIS", group_col], how="left")
        .with_columns(pl.col("load").fill_null(0.0), pl.col("margin").fill_null(0.0))
        .with_columns(residual=(pl.col("margin") - pl.col("load")).clip(lower_bound=0.0))
        .join(iris_cantons.select("IRIS", "CANTVILLE").unique(), on="IRIS", how="left")
    )


def _allocate_stage2(census, iris_cantons, margins, config, rng):
    """Controlled selection within each canton x margin group, on residual
    capacity."""
    group_col = config.schema.group_label_column
    residual = _residual_margins(census, margins, iris_cantons, config)
    households = census.filter(pl.col("IRIS") == IRIS_UNKNOWN, pl.col("LPRM") == "1").select(
        "household_id", "CANTVILLE", group_col, "_w"
    )
    canton_iris = (
        iris_cantons.select("CANTVILLE", "IRIS").unique().group_by("CANTVILLE").agg(pl.col("IRIS"))
    )

    assignments: list[dict] = []
    for cantville in households["CANTVILLE"].unique().to_list():
        canton_hh = households.filter(pl.col("CANTVILLE") == cantville)
        canton_res = residual.filter(pl.col("CANTVILLE") == cantville)
        fb = canton_iris.filter(pl.col("CANTVILLE") == cantville)
        fallback_iris = fb["IRIS"].to_list()[0] if len(fb) else []

        for group in MARGIN_GROUPS:
            grp_hh = canton_hh.filter(pl.col(group_col) == group)
            if len(grp_hh) == 0:
                continue

            rc = canton_res.filter(pl.col(group_col) == group).sort("IRIS")
            iris_codes = rc["IRIS"].to_list()
            raw = rc["residual"].to_numpy().astype(float)
            total_raw = raw.sum()
            grp_weight = grp_hh["_w"].sum()

            if total_raw > 0:
                target = raw * (grp_weight / total_raw)
            else:
                iris_codes = fallback_iris if fallback_iris else iris_codes
                if not iris_codes:
                    continue
                target = np.ones(len(iris_codes)) * (grp_weight / len(iris_codes))

            shuffled = grp_hh.select(["household_id", "_w"]).sample(
                fraction=1.0, shuffle=True, seed=int(rng.integers(1_000_000_000))
            )
            ids = shuffled["household_id"].to_numpy()
            cum = np.cumsum(shuffled["_w"].to_numpy())
            bounds = np.concatenate([[0], np.cumsum(target)])
            idx = np.searchsorted(bounds[1:], cum, side="left")

            oob = idx >= len(iris_codes)
            if oob.any():
                idx[oob] = rng.choice(len(iris_codes), size=int(oob.sum()), p=target / target.sum())
            for hid, j in zip(ids, idx):
                assignments.append({"household_id": hid, "IRIS_NEW": iris_codes[j]})

    return _apply_assignments(census, assignments)


def _tag_iris_provenance(census, stage1_ids, stage2_ids):
    """Add the iris_from_mobpro/iris_from_mobsco/iris_from_canton flags.

    Stage-1 households were placed within a known commune; whether that
    commune came from MOBPRO or MOBSCO is recorded in ``_commune_source``
    (set upstream in the pipeline). Stage-2 households had no known commune
    and were placed within their canton.
    """
    return census.with_columns(
        iris_from_mobpro=pl.col("household_id").is_in(stage1_ids)
        & (pl.col("_commune_source") == "mobpro"),
        iris_from_mobsco=pl.col("household_id").is_in(stage1_ids)
        & (pl.col("_commune_source") == "mobsco"),
        iris_from_canton=pl.col("household_id").is_in(stage2_ids),
    )


def allocate_iris(census, iris_cantons, margins, config):
    """Run both IRIS-allocation stages and rebuild COMMUNE from the final IRIS.
    Asserts no household is left without an IRIS.

    Expects an upstream ``_commune_source`` column (values 'mobpro'/'mobsco'/
    null) to derive the iris_from_mobpro/iris_from_mobsco/iris_from_canton
    flags; that helper column is dropped before returning.
    """
    section("IRIS allocation")
    rng = np.random.default_rng(config.root_seed)
    census = census.with_columns(_w=weight_to_float("IPONDI"))

    census, stage1_df = _allocate_stage1(census, iris_cantons, margins, config, rng)
    log(f"  stage 1 (commune known) - households placed: {stage1_df.height:>8}")

    census, stage2_df = _allocate_stage2(census, iris_cantons, margins, config, rng)
    log(f"  stage 2 (canton)        - households placed: {stage2_df.height:>8}")

    census = _tag_iris_provenance(
        census, stage1_df["household_id"].unique(), stage2_df["household_id"].unique()
    )
    census = census.drop("_commune_source")

    census = census.with_columns(COMMUNE=pl.col("IRIS").str.slice(0, 5))

    n_unknown = census.filter(pl.col("IRIS") == IRIS_UNKNOWN).height
    log(f"  IRIS still unknown      : {n_unknown}")
    census = census.drop("_w")
    assert n_unknown == 0, "Some individuals still have no IRIS assigned"
    return census
