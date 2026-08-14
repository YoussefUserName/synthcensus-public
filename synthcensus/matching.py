"""
Deterministic probabilistic record linkage.

The matching strategy links census individuals to a flow file (MOBPRO,
MOBSCO or the dwelling file) on the set of variables the two share. A link
is accepted only when a combination of matching values is *unique on both
sides* (one-to-one), which guarantees an unambiguous transfer of the extra
columns (work commune, study commune, ...).

Each link is attempted twice -- once including the individual weight
``IPONDI`` in the key, once excluding it -- and the weight-included result is
preferred. Including the weight sharpens the match (the weight is an almost
unique fingerprint within a stratum); excluding it recovers the individuals
whose weight differs slightly between the two files.
"""

from __future__ import annotations

import polars as pl


def get_match_cols(left: pl.DataFrame, right: pl.DataFrame, exclude: list[str]) -> list[str]:
    """Return the columns common to both frames, minus the excluded ones.

    Column order follows ``left`` so the key is deterministic.
    """
    return [c for c in left.columns if c in right.columns and c not in exclude]


def _cast_keys_to_str(df: pl.DataFrame, match_cols: list[str]) -> pl.DataFrame:
    """Cast every key column not already Utf8 to Utf8 (matching is textual).
    Columns already Utf8 are left untouched to avoid redundant work.

    Some millésimes (e.g. 2022 MOBPRO) store integer-coded variables as
    Float64 while the census has them as String; casting Float64 straight to
    Utf8 yields "1.0" instead of "1", silently breaking every match. These
    code columns are always integer-valued, so floats are routed through
    Int64 first.
    """
    exprs = []
    for c in match_cols:
        dtype = df[c].dtype
        if dtype == pl.Utf8:
            continue
        if dtype in (pl.Float32, pl.Float64):
            exprs.append(pl.col(c).cast(pl.Int64).cast(pl.Utf8))
        else:
            exprs.append(pl.col(c).cast(pl.Utf8))
    return df.with_columns(exprs) if exprs else df


def unique_match(
    left: pl.DataFrame,
    right: pl.DataFrame,
    match_cols: list[str],
    extra_cols: list[str],
) -> pl.DataFrame:
    """Return the rows of ``right`` (with ``extra_cols``) whose combination of
    ``match_cols`` occurs exactly once in *both* frames.

    Only bilaterally unique combinations are kept, so the returned table maps
    each retained key to a single donor row.
    """
    left_str = _cast_keys_to_str(left, match_cols)
    right_str = _cast_keys_to_str(right, match_cols)

    left_unique = (
        left_str.group_by(match_cols).agg(pl.len().alias("_n"))
        .filter(pl.col("_n") == 1).drop("_n")
    )
    right_unique = (
        right_str.group_by(match_cols).agg(pl.len().alias("_n"))
        .filter(pl.col("_n") == 1).drop("_n")
    )

    keys = left_unique.join(right_unique, on=match_cols, how="inner")
    return right_str.join(keys, on=match_cols, how="inner").select(match_cols + extra_cols)


def _run_single_pass(
    left: pl.DataFrame,
    right: pl.DataFrame,
    match_cols: list[str],
    extra_cols: list[str],
) -> pl.DataFrame:
    """Run one unique match and left-join the donor columns back onto ``left``
    (null where no unique donor was found)."""
    left_str = _cast_keys_to_str(left, match_cols)
    donors = unique_match(left, right, match_cols, extra_cols)
    return left_str.join(donors, on=match_cols, how="left")


def double_merge(
    left: pl.DataFrame,
    right: pl.DataFrame,
    match_cols_base: list[str],
    extra_cols: list[str],
    weight_col: str = "IPONDI",
    id_col: str = "person_id",
) -> pl.DataFrame:
    """Two-pass unique match: with then without the weight in the key.

    The weight-included match is preferred; the weight-excluded match is used
    as a fallback for individuals it could not resolve. Returns ``left`` with
    ``extra_cols`` appended (null where neither pass found a unique donor).
    """
    with_weight = _run_single_pass(left, right, match_cols_base + [weight_col], extra_cols)
    without_weight = _run_single_pass(left, right, match_cols_base, extra_cols)

    res_with = with_weight.select([id_col, *extra_cols]).rename(
        {c: f"{c}_with" for c in extra_cols}
    )
    res_without = without_weight.select([id_col, *extra_cols]).rename(
        {c: f"{c}_without" for c in extra_cols}
    )

    combined = (
        res_with.join(res_without, on=id_col, how="left")
        .with_columns(
            [pl.coalesce([pl.col(f"{c}_with"), pl.col(f"{c}_without")]).alias(c) for c in extra_cols]
        )
        .select([id_col, *extra_cols])
    )
    return left.join(combined, on=id_col, how="left")
