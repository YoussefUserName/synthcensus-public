"""
Input loading and preparation.

Five user inputs are consumed: the individual census (RP), MOBPRO (work),
MOBSCO (study), the couples-families-households margins (BTX_IC_FAM) and the
IRIS-to-canton correspondence. Each loader accepts a path or an in-memory
``polars.DataFrame``.

Year- and perimeter-dependent preparation happens here: the geographic filter
(region/department union), the ARM arrondissement correction on the flow
files, the derived 8-post margin group, and the selection of the correct
margin columns for the requested year.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import polars as pl

from .config import ARM_NONE, PipelineConfig
from .utils import log, normalize_weight

FrameOrPath = Union[str, Path, pl.DataFrame]


def _resolve_single_file(src: FrameOrPath) -> FrameOrPath:
    """If ``src`` is a directory, resolve it to the single file it contains.

    Lets callers pass a folder (e.g. ``data/2022/census_individuals/``)
    instead of the exact file path. Raises if the folder doesn't contain
    exactly one file, since there would be no unambiguous choice.
    """
    if isinstance(src, pl.DataFrame):
        return src
    path = Path(src)
    if not path.is_dir():
        return path
    files = [f for f in path.iterdir() if f.is_file() and not f.name.startswith(".")]
    if len(files) != 1:
        raise ValueError(f"Expected exactly one file in {path}, found {len(files)}")
    return files[0]


def _read_parquet(src: FrameOrPath) -> pl.DataFrame:
    """Read a parquet file, decoding categorical columns to plain strings.

    Some millésimes (e.g. 2022 MOBPRO/MOBSCO) are exported with Categorical
    dtype on code columns (COMMUNE, ARM, DCLT, DCFLT...); the rest of the
    pipeline does string operations (``str.slice``) on these columns, which
    polars refuses on Categorical without an explicit cast.
    """
    src = _resolve_single_file(src)
    if isinstance(src, pl.DataFrame):
        df = src
    else:
        if Path(src).suffix.lower() != ".parquet":
            raise ValueError(f"Expected a .parquet file, got {src}")
        df = pl.read_parquet(src)
    cat_cols = [name for name, dtype in df.schema.items() if dtype == pl.Categorical]
    if cat_cols:
        df = df.with_columns(pl.col(cat_cols).cast(pl.Utf8))
    return df


def _apply_perimeter(df: pl.DataFrame, config: PipelineConfig, region_col: str, dept_col: str) -> pl.DataFrame:
    """Keep rows whose region OR department is selected (union).

    If neither regions nor departments are configured, the frame is returned
    unchanged (whole territory).
    """
    if config.keeps_whole_territory():
        return df
    cond = pl.lit(False)
    if config.regions:
        cond = cond | pl.col(region_col).is_in(config.regions)
    if config.depts:
        cond = cond | pl.col(dept_col).is_in(config.depts)
    return df.filter(cond)


def add_margin_group(df: pl.DataFrame, config: PipelineConfig) -> pl.DataFrame:
    """Add the 8-post margin group column derived from the year's CSP variable.

    For 2007-2021 the CSP variable (CS1) already holds labels '1'..'8', so the
    group is a copy. For 2022 STAT_GSEC is regrouped via the schema mapping,
    every unlisted value falling into group '8' (other inactive).
    """
    schema = config.schema
    target = schema.group_label_column
    if not schema.csp_to_group:
        return df.with_columns(pl.col(schema.csp_var).alias(target))
    expr = pl.col(schema.csp_var)
    mapped = pl.lit("8")
    for value, group in schema.csp_to_group.items():
        mapped = pl.when(expr == value).then(pl.lit(group)).otherwise(mapped)
    return df.with_columns(mapped.alias(target))


def load_regions_2016(src: FrameOrPath) -> pl.DataFrame:
    """DEPT -> REGION correspondence using the post-reform (2016) nomenclature.

    Used to filter years whose own REGION codes predate the 2016 regions
    reform. The 2016 vintage specifically is required: the 2015 IRIS-cantons
    file still carries pre-reform codes even though the 2015
    census/MOBPRO/MOBSCO releases already use the new ones. When ``src`` is a
    directory (the usual ``iris_cantons`` pipeline input), the
    ``iris_cantons_2016.parquet`` file within it is picked.
    """
    if isinstance(src, pl.DataFrame):
        df = src
    else:
        path = Path(src)
        if path.is_dir():
            path = path / "iris_cantons_2016.parquet"
        df = pl.read_parquet(path)
    return (
        df.select(pl.col("REG").alias("REGION2016"), pl.col("DEP").alias("DEPT"))
        .unique(["REGION2016", "DEPT"])
    )


def add_region_2016(
    df: pl.DataFrame, config: PipelineConfig, regions_2016: pl.DataFrame, dept_col: str = "DEPT"
) -> pl.DataFrame:
    """Add ``REGION2016`` for millésimes preceding the 2016 regions reform.

    The 2013 transitional files already ship a native ``REGION2016`` column
    (INSEE's own precomputed harmonization); the ``DEPT -> REGION2016`` join
    is only needed when it's absent, otherwise it collides with the native
    column. From 2014 the year's own REGION is already post-reform, so
    nothing is added.
    """
    if config.year >= 2014 or "REGION2016" in df.columns:
        return df
    return df.join(regions_2016, left_on=dept_col, right_on="DEPT", how="left")


def load_census(src: FrameOrPath, config: PipelineConfig, regions_2016: pl.DataFrame) -> pl.DataFrame:
    """Load the census, restrict to ordinary households in metropolitan France
    and to the configured perimeter, build ids, canonicalise the weight, and
    add the margin group."""
    census = (
        _read_parquet(src)
        .lazy()
        .with_columns(household_id=pl.col("NUMMI") + pl.col("CANTVILLE"))
        .filter(pl.col("NUMMI").is_not_null(), pl.col("NUMMI") != "Z")
        .filter(~pl.col("REGION").is_in(config.overseas_regions))
        .collect()
    )
    census = add_region_2016(census, config, regions_2016)
    if config.year < 2014:
        census = _apply_perimeter(census, config, "REGION2016", "DEPT")
    else:
        census = _apply_perimeter(census, config, "REGION", "DEPT")
    census = normalize_weight(census, "IPONDI", config.ipondi_round_decimals)
    census = census.sort("household_id").with_columns(
        pl.int_range(1, pl.len() + 1).alias("person_id")
    )
    census = add_margin_group(census, config)
    log(f"  census loaded: {census.height:>9} individuals")
    return census


def _correct_arm(df: pl.DataFrame) -> pl.DataFrame:
    """Replace COMMUNE by ARM when ARM designates a real arrondissement.

    In Paris/Lyon/Marseille the flow files carry the global commune code in
    COMMUNE (e.g. 75056 for Paris) and the arrondissement in ARM; without this
    correction every arrondissement collapses onto a single commune code.
    Applied only to the residence commune of the flow files (not destinations).
    """
    if "ARM" not in df.columns:
        return df
    return df.with_columns(
        COMMUNE=pl.when(pl.col("ARM") != ARM_NONE).then(pl.col("ARM")).otherwise(pl.col("COMMUNE"))
    )


def _harmonize_dest_region(
    df: pl.DataFrame,
    dest_col: str,
    native_col: str,
    harmonized_native_col: str,
    config: PipelineConfig,
    regions_2016: pl.DataFrame,
    out_col: str,
) -> pl.DataFrame:
    """Add ``out_col``: the destination (work/study) region in 2016 nomenclature.

    INSEE ships a precomputed harmonized column only for the 2013 millésime
    (``REGLT16``/``REGETUD16``); from 2014 the native column (``REGLT``/
    ``REGETUD``) is already post-reform. Earlier millésimes (native
    ``REGIONLT``/``REGIONETUD``, or no destination-region column at all in
    2007) are derived from the destination commune's department via
    ``regions_2016``.
    """
    if harmonized_native_col in df.columns:
        return df.with_columns(pl.col(harmonized_native_col).alias(out_col))
    if native_col in df.columns and config.year >= 2014:
        return df.with_columns(pl.col(native_col).alias(out_col))
    dep_col = f"_DEP_{out_col}"
    return (
        df.with_columns(**{dep_col: pl.col(dest_col).str.slice(0, 2)})
        .join(regions_2016.rename({"REGION2016": out_col}), left_on=dep_col, right_on="DEPT", how="left")
        .drop(dep_col)
    )


def load_mobpro(
    census: pl.DataFrame, src: FrameOrPath, config: PipelineConfig, regions_2016: pl.DataFrame
) -> pl.DataFrame:
    mobpro = _correct_arm(_read_parquet(src))
    mobpro = normalize_weight(mobpro, "IPONDI", config.ipondi_round_decimals)
    if 'DEPT' not in mobpro.columns:
        mobpro = mobpro.with_columns(DEPT=pl.col('COMMUNE').str.slice(0, 2))
    if 'REGION' not in mobpro.columns:
        regions = census.select(['REGION', 'DEPT']).unique(['REGION', 'DEPT'])
        mobpro = mobpro.join(regions, on='DEPT', how='left')
    mobpro = add_region_2016(mobpro, config, regions_2016)
    if config.year < 2014:
        mobpro = _apply_perimeter(mobpro, config, "REGION2016", "DEPT")
    else:
        mobpro = _apply_perimeter(mobpro, config, "REGION", "DEPT")
    mobpro = _harmonize_dest_region(mobpro, "DCLT", "REGLT", "REGLT16", config, regions_2016, "REGLT2016")
    log(f"  mobpro loaded: {mobpro.height:>9} commuting records")
    return mobpro


def load_mobsco(
    census: pl.DataFrame, src: FrameOrPath, config: PipelineConfig, regions_2016: pl.DataFrame
) -> pl.DataFrame:
    mobsco = _correct_arm(_read_parquet(src))
    mobsco = normalize_weight(mobsco, "IPONDI", config.ipondi_round_decimals)
    if 'DEPT' not in mobsco.columns:
        mobsco = mobsco.with_columns(DEPT=pl.col('COMMUNE').str.slice(0, 2))
    if 'REGION' not in mobsco.columns:
        regions = census.select(['REGION', 'DEPT']).unique(['REGION', 'DEPT'])
        mobsco = mobsco.join(regions, on='DEPT', how='left')
    mobsco = add_region_2016(mobsco, config, regions_2016)
    if config.year < 2014:
        mobsco = _apply_perimeter(mobsco, config, "REGION2016", "DEPT")
    else:
        mobsco = _apply_perimeter(mobsco, config, "REGION", "DEPT")
    mobsco = _harmonize_dest_region(mobsco, "DCETUF", "REGETUD", "REGETUD16", config, regions_2016, "REGETUD2016")
    log(f"  mobsco loaded: {mobsco.height:>9} schooling records")
    return mobsco


def load_iris_cantons(census: pl.DataFrame, src: FrameOrPath, config: PipelineConfig) -> pl.DataFrame:
    """Load the IRIS-to-canton correspondence for the configured year.

    INSEE's census being a rolling-year survey, a canton present in the
    census is not always present in that same year's IRIS-cantons file. When
    ``src`` is a directory, every ``iris_cantons_*.parquet`` file in it is
    loaded and tagged with its year; the configured year's rows are used,
    and for any CANTVILLE found in the census but missing from that year,
    the corresponding rows from any other year are added as a fallback.
    """
    if isinstance(src, pl.DataFrame):
        all_years = src
    elif Path(src).is_dir():
        dfs = []
        for fichier in sorted(Path(src).glob("iris_cantons_*.parquet")):
            annee = fichier.stem.split("_")[-1]
            dfs.append(pl.read_parquet(fichier).with_columns(year=pl.lit(annee)))
        all_years = pl.concat(dfs, how="vertical_relaxed")
    else:
        all_years = pl.read_parquet(src)

    all_years = all_years.rename({"CODE_IRIS": "IRIS", "INSEE_COM": "COMMUNE", "CANTON": "CANTVILLE"})

    if "year" not in all_years.columns:
        log(f"  iris_cantons loaded: {all_years.height:>9} IRIS rows")
        return all_years

    iris_cantons = all_years.filter(pl.col("year") == str(config.year))
    missing = (
        census.select("CANTVILLE")
        .unique()
        .join(iris_cantons.select("CANTVILLE").unique(), on="CANTVILLE", how="anti")
    )
    if missing.height:
        fallback = (
            all_years.join(missing, on="CANTVILLE", how="inner")
            .unique(subset=["IRIS"], keep="first")
        )
        iris_cantons = pl.concat([iris_cantons, fallback], how="vertical_relaxed")
        log(f"  iris_cantons: {missing.height} CANTVILLE missing from {config.year}, filled from other years")

    log(f"  iris_cantons loaded: {iris_cantons.height:>9} IRIS rows")
    return iris_cantons


def load_family_margins(src: FrameOrPath, config: PipelineConfig, sheet_name: str = "IRIS") -> pl.DataFrame:
    """Load the couples-families-households margins for the requested year.

    Selects the total household column and the eight group columns named after
    the year's nomenclature, casting them to float.
    """
    schema = config.schema
    margin_cols = [schema.total_margin_column, *schema.margin_columns]
    src = _resolve_single_file(src)
    if isinstance(src, pl.DataFrame):
        margins = src
    elif Path(src).suffix.lower() == ".csv":
        margins = pl.read_csv(src, separator=';', infer_schema_length=0)
    else:
        margins = pl.read_excel(src, sheet_name=sheet_name)
    margins = margins.select(["IRIS", *margin_cols]).with_columns(pl.col(margin_cols).cast(pl.Float64))
    log(f"  family_margins loaded: {margins.height:>9} IRIS margins ({schema.margin_prefix})")
    return margins


def build_canton_commune(iris_cantons: pl.DataFrame) -> pl.DataFrame:
    return iris_cantons.select(["COMMUNE", "CANTVILLE"]).unique(["COMMUNE", "CANTVILLE"])
