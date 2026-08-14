"""
Stochastic imputation of work and study communes for the residual
individuals the deterministic match could not resolve.

For these individuals the census still gives a relative-location indicator
(ILT for work, ILETUD for study). When the indicator is "same commune" the
destination is the residence commune itself. For the other in-country
modalities (2, 3, 4) and "abroad" (7), the precise destination is drawn from
the empirical flow distribution observed in MOBPRO/MOBSCO, conditional on the
residence commune and a stratum, with a three-level fallback when the fine
stratum is empty:

  1. residence commune x stratum,
  2. residence commune (all strata),
  3. whole geographic perimeter, weighted by destination size.

See the methodology note for the formal description.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .config import COMMUNE_UNKNOWN, FOREIGN_FRANCE_CODE, PipelineConfig
from .loaders import add_region_2016
from .utils import log, section


def _build_index(df: pl.DataFrame, key_cols: list[str]) -> dict:
    """Map each key tuple to ([destinations], [weights]) for fast sampling."""
    index: dict = {}
    for row in df.iter_rows(named=True):
        key = tuple(row[c] for c in key_cols)
        dests, weights = index.setdefault(key, ([], []))
        dests.append(row["__dest__"])
        weights.append(row["w"])
    return index


def impute_destination(
    targets: pl.DataFrame,
    flow: pl.DataFrame,
    origin_col: str,
    dest_col: str,
    stratum_cols: list[str],
    perimeter_mask: pl.Expr,
    seed: int,
) -> pl.DataFrame:
    """Draw a destination for each target individual.

    Parameters
    ----------
    targets : individuals to impute; must carry ``COMMUNE``, ``person_id`` and
        every column in ``stratum_cols``.
    flow : flow file restricted by ``perimeter_mask`` to the geographic scope
        compatible with the individuals' location modality.
    origin_col, dest_col : residence and destination commune columns in ``flow``.
    stratum_cols : fine-stratum columns (e.g. ['SEXE', 'CS1']).
    seed : per-call seed for reproducibility.

    Returns a frame with ``person_id`` and ``__dest__`` (the drawn commune).
    """
    rng = np.random.default_rng(seed)
    flow_p = flow.filter(perimeter_mask)

    keys_l1 = ["__orig__"] + stratum_cols
    level1 = (
        flow_p.rename({origin_col: "__orig__", dest_col: "__dest__"})
        .group_by(keys_l1 + ["__dest__"])
        .agg(w=pl.col("IPONDI").cast(pl.Float64).sum())
    )
    level2 = (
        flow_p.rename({origin_col: "__orig__", dest_col: "__dest__"})
        .group_by(["__orig__", "__dest__"])
        .agg(w=pl.col("IPONDI").cast(pl.Float64).sum())
    )
    level3 = (
        flow_p.rename({dest_col: "__dest__"})
        .group_by("__dest__")
        .agg(w=pl.col("IPONDI").cast(pl.Float64).sum())
    )
    dest_l3 = level3["__dest__"].to_numpy()
    w_l3 = level3["w"].to_numpy()
    p_l3 = w_l3 / w_l3.sum() if w_l3.sum() > 0 else None

    idx_l1 = _build_index(level1, keys_l1)
    idx_l2 = _build_index(level2, ["__orig__"])

    assignments: list[dict] = []
    n1 = n2 = n3 = n_fail = 0

    for row in targets.iter_rows(named=True):
        commune = row["COMMUNE"]
        k1 = tuple([commune] + [row[c] for c in stratum_cols])
        k2 = (commune,)

        if k1 in idx_l1:
            dests, weights = idx_l1[k1]
            level = 1
        elif k2 in idx_l2:
            dests, weights = idx_l2[k2]
            level = 2
        elif p_l3 is not None:
            assignments.append({"person_id": row["person_id"], "__dest__": rng.choice(dest_l3, p=p_l3)})
            n3 += 1
            continue
        else:
            n_fail += 1
            continue

        weights = np.asarray(weights, float)
        assignments.append(
            {"person_id": row["person_id"], "__dest__": rng.choice(dests, p=weights / weights.sum())}
        )
        if level == 1:
            n1 += 1
        else:
            n2 += 1

    log(f"    level 1 (commune x stratum): {n1}")
    log(f"    level 2 (commune only)     : {n2}")
    log(f"    level 3 (perimeter)        : {n3}")
    log(f"    failed (no flow)           : {n_fail}")

    schema = {"person_id": pl.Int64, "__dest__": pl.Utf8}
    return pl.DataFrame(assignments, schema=schema) if assignments else pl.DataFrame(schema=schema)


def _coalesce_destination(census: pl.DataFrame, result: pl.DataFrame, dest_col: str) -> pl.DataFrame:
    """Fill the still-null destination column with the imputed values."""
    return (
        census.join(result.rename({"__dest__": "_imp"}), on="person_id", how="left")
        .with_columns(**{dest_col: pl.coalesce([pl.col(dest_col), pl.col("_imp")])})
        .drop("_imp")
    )


def _add_flow_geography(
    flow: pl.DataFrame,
    regions: pl.DataFrame,
    regions_2016: pl.DataFrame,
    dest_col: str,
    dest_region_cols: tuple[str, ...],
    harmonized_dest_region_col: str,
    config: PipelineConfig,
) -> pl.DataFrame:
    """Attach residence/destination department and region to a flow file, and
    restrict it to flows originating in the study perimeter.

    REGION_R/REGION_D use the year's own region nomenclature -- required for
    the same-region/same-department modality masks below to match how INSEE
    coded ILT/ILETUD for that millésime. REGION_D is read directly from the
    flow's own native destination-region variable (REGLT/REGIONLT for
    MOBPRO, REGETUD/REGIONETUD for MOBSCO) when present: that variable covers
    the whole country, unlike a DEPT lookup restricted to the study
    perimeter, so genuine cross-region commutes ending outside the perimeter
    aren't silently lost.

    2007 has no such native column at all. For that millésime only, both
    sides fall back to the harmonised 2016 nomenclature (``REGION2016`` for
    the residence, ``REGLT2016``/``REGETUD2016`` for the destination --
    already computed nationwide, department-based, in
    ``load_mobpro``/``load_mobsco``), since that's the only country-wide
    correspondence available; comparing a native (old) REGION_R to a
    harmonised REGION_D would silently misclassify every cross-nomenclature
    pair as "different region".

    The perimeter restriction itself always applies
    ``config.regions``/``config.depts`` in the harmonised post-2016
    nomenclature, consistent with ``loaders._apply_perimeter``.
    """
    flow = flow.with_columns(
        DEP_R=pl.col("COMMUNE").str.slice(0, 2), DEP_D=pl.col(dest_col).str.slice(0, 2)
    ).join(regions.rename({"REGION": "REGION_R"}), left_on="DEP_R", right_on="DEPT", how="left")

    if config.year < 2014:
        flow = add_region_2016(flow, config, regions_2016, dept_col="DEP_R")
        perimeter_region_col = "REGION2016"
    else:
        perimeter_region_col = "REGION_R"

    native_dest_region = next((c for c in dest_region_cols if c in flow.columns), None)
    if native_dest_region:
        flow = flow.with_columns(REGION_D=pl.col(native_dest_region))
    else:
        flow = flow.with_columns(
            REGION_R=pl.col(perimeter_region_col),
            REGION_D=pl.col(harmonized_dest_region_col),
        )

    if config.keeps_whole_territory():
        return flow
    cond = pl.lit(False)
    if config.regions:
        cond = cond | pl.col(perimeter_region_col).is_in(config.regions)
    if config.depts:
        cond = cond | pl.col("DEP_R").is_in(config.depts)
    return flow.filter(cond)


def _set_same_commune_destinations(census: pl.DataFrame) -> pl.DataFrame:
    """Modality 1 (same commune as residence): destination is the residence
    commune; the foreign code marks "metropolitan France"."""
    return census.with_columns(
        DCETUF=pl.when(pl.col("ILETUD") == "1", pl.col("DCETUF").is_null())
        .then(pl.col("COMMUNE"))
        .otherwise(pl.col("DCETUF")),
        DCETUE=pl.when(pl.col("ILETUD") == "1", pl.col("DCETUE").is_null())
        .then(pl.lit(FOREIGN_FRANCE_CODE))
        .otherwise(pl.col("DCETUE")),
        DCLT=pl.when(pl.col("ILT") == "1", pl.col("DCLT").is_null())
        .then(pl.col("COMMUNE"))
        .otherwise(pl.col("DCLT")),
        DCFLT=pl.when(pl.col("ILT") == "1", pl.col("DCFLT").is_null())
        .then(pl.lit(FOREIGN_FRANCE_CODE))
        .otherwise(pl.col("DCFLT")),
    )


def _finalise_foreign_codes(census: pl.DataFrame) -> pl.DataFrame:
    """Ensure the France/abroad code pairs are mutually consistent: a person
    with a French work commune gets the France code in DCFLT and vice versa
    (same for study)."""
    return (
        census.with_columns(
            DCLT=pl.when(pl.col("DCLT").is_null(), pl.col("DCFLT").is_not_null())
            .then(pl.lit(FOREIGN_FRANCE_CODE))
            .otherwise(pl.col("DCLT"))
        )
        .with_columns(
            DCFLT=pl.when(pl.col("DCLT").is_not_null(), pl.col("DCFLT").is_null())
            .then(pl.lit(FOREIGN_FRANCE_CODE))
            .otherwise(pl.col("DCFLT"))
        )
        .with_columns(
            DCETUF=pl.when(pl.col("DCETUF").is_null(), pl.col("DCETUE").is_not_null())
            .then(pl.lit(FOREIGN_FRANCE_CODE))
            .otherwise(pl.col("DCETUF"))
        )
        .with_columns(
            DCETUE=pl.when(pl.col("DCETUE").is_not_null(), pl.col("DCETUF").is_null())
            .then(pl.lit(FOREIGN_FRANCE_CODE))
            .otherwise(pl.col("DCETUE"))
        )
    )


def _log_destination_coverage(census: pl.DataFrame) -> None:
    """Log the share of workers/students with a known work/study commune."""
    workers = census.filter(pl.col("TACT") == "11")
    n_workers = workers.height
    n_workers_known = workers.filter(pl.col("DCLT").is_not_null()).height
    pct_w = (n_workers_known / n_workers * 100) if n_workers else 0.0

    students = census.filter(pl.col("ETUD") == "1")
    n_students = students.height
    n_students_known = students.filter(pl.col("DCETUF").is_not_null()).height
    pct_s = (n_students_known / n_students * 100) if n_students else 0.0

    log(f"  workers with known DCLT    : {n_workers_known} / {n_workers} ({pct_w:.0f}%)")
    log(f"  students with known DCETUF : {n_students_known} / {n_students} ({pct_s:.0f}%)\n")


def impute_destinations(
    census: pl.DataFrame,
    mobpro: pl.DataFrame,
    mobsco: pl.DataFrame,
    config: PipelineConfig,
    regions_2016: pl.DataFrame,
) -> pl.DataFrame:
    """Impute every residual work/study destination, in place on the census.

    Work strata: SEXE x <year CSP var>. Study strata: SEXE x AGEREV10. The "abroad"
    modality draws a foreign country code from the corresponding flow file.
    """
    section("Destination imputation (work & study)")
    rng = np.random.default_rng(config.root_seed)
    csp = config.schema.csp_var  # CS1 before 2022, STAT_GSEC from 2022

    census = _set_same_commune_destinations(census)
    _log_destination_coverage(census)

    regions = census.select(["REGION", "DEPT"]).unique(["REGION", "DEPT"])
    mobpro_geo = _add_flow_geography(
        mobpro, regions, regions_2016, "DCLT", ("REGLT", "REGIONLT"), "REGLT2016", config
    )
    mobsco_geo = _add_flow_geography(
        mobsco, regions, regions_2016, "DCETUF", ("REGETUD", "REGIONETUD"), "REGETUD2016", config
    )

    # The study stratum is SEXE x AGEREV10, so the MOBSCO flow needs the same
    # age band. Derive it from AGED when MOBSCO does not already carry it;
    # without an age column the study stratum silently falls back to commune.
    if "AGEREV10" not in mobsco_geo.columns:
        from .household_vars import add_age_band

        if "AGED" in mobsco_geo.columns:
            mobsco_geo = add_age_band(mobsco_geo)
        else:
            log("  note: MOBSCO has no AGED/AGEREV10; study stratum falls back to commune level")

    # --- WORK (DCLT) for in-country modalities 2/3/4.
    log("work (DCLT):")
    work_stratum = ["SEXE", csp] if csp in mobpro_geo.columns else ["SEXE"]
    work_modalities = [
        ("2", (pl.col("REGION_D") == pl.col("REGION_R")) & (pl.col("DEP_D") == pl.col("DEP_R")) & (pl.col("COMMUNE") != pl.col("DCLT"))),
        ("3", (pl.col("REGION_D") == pl.col("REGION_R")) & (pl.col("DEP_D") != pl.col("DEP_R"))),
        ("4", pl.col("REGION_D").is_null() | (pl.col("REGION_D") != pl.col("REGION_R"))),
    ]
    for ilt, mask in work_modalities:
        targets = census.filter(pl.col("DCLT").is_null(), pl.col("TACT") == "11", pl.col("ILT") == ilt)
        if targets.height == 0:
            continue
        log(f"  ILT={ilt} - to impute: {targets.height}")
        res = impute_destination(targets, mobpro_geo, "COMMUNE", "DCLT", work_stratum, mask, int(rng.integers(1_000_000_000)))
        census = _coalesce_destination(census, res, "DCLT")

    # --- WORK abroad (DCFLT), stratum commune x <year CSP var>.
    log("work abroad (DCFLT):")
    targets = census.filter(pl.col("DCFLT").is_null(), pl.col("TACT") == "11", pl.col("ILT") == "7")
    if targets.height > 0:
        mask = pl.col("DCFLT").is_not_null() & (pl.col("DCFLT") != FOREIGN_FRANCE_CODE)
        res = impute_destination(targets, mobpro_geo, "COMMUNE", "DCFLT", [csp] if csp in mobpro_geo.columns else [], mask, int(rng.integers(1_000_000_000)))
        census = _coalesce_destination(census, res, "DCFLT")

    # --- STUDY (DCETUF) for in-country modalities 2/3/4.
    log("study (DCETUF):")
    study_stratum = ["SEXE", "AGEREV10"] if "AGEREV10" in mobsco_geo.columns else ["SEXE"]
    study_modalities = [
        ("2", (pl.col("REGION_D") == pl.col("REGION_R")) & (pl.col("DEP_D") == pl.col("DEP_R")) & (pl.col("COMMUNE") != pl.col("DCETUF"))),
        ("3", (pl.col("REGION_D") == pl.col("REGION_R")) & (pl.col("DEP_D") != pl.col("DEP_R"))),
        ("4", pl.col("REGION_D").is_null() | (pl.col("REGION_D") != pl.col("REGION_R"))),
    ]
    for iletud, mask in study_modalities:
        targets = census.filter(pl.col("DCETUF").is_null(), pl.col("ETUD") == "1", pl.col("ILETUD") == iletud)
        if targets.height == 0:
            continue
        log(f"  ILETUD={iletud} - to impute: {targets.height}")
        res = impute_destination(targets, mobsco_geo, "COMMUNE", "DCETUF", study_stratum, mask, int(rng.integers(1_000_000_000)))
        census = _coalesce_destination(census, res, "DCETUF")

    # --- STUDY abroad (DCETUE), no stratum (global weighted draw).
    log("study abroad (DCETUE):")
    targets = census.filter(pl.col("DCETUE").is_null(), pl.col("ETUD") == "1", pl.col("ILETUD") == "7")
    if targets.height > 0:
        mask = pl.col("DCETUE").is_not_null() & (pl.col("DCETUE") != FOREIGN_FRANCE_CODE)
        res = impute_destination(targets, mobsco_geo, "COMMUNE", "DCETUE", [], mask, int(rng.integers(1_000_000_000)))
        census = _coalesce_destination(census, res, "DCETUE")

    census = _finalise_foreign_codes(census)
    census = census.with_columns(
        pl.col(["DCLT", "DCFLT", "DCETUF", "DCETUE"]).fill_null(COMMUNE_UNKNOWN)
    )
    return census
