"""
Low-level helpers shared across the pipeline: weight canonicalisation and a
small logging helper for progress messages.
"""

from __future__ import annotations

import polars as pl


def log(message: str) -> None:
    """Print a progress message. Centralised so the output style is uniform
    and could later be redirected to a real logger without touching callers."""
    print(message)


def section(title: str) -> None:
    """Print a section header to visually separate pipeline stages."""
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def normalize_weight(df: pl.DataFrame, col: str = "IPONDI", decimals: int = 10) -> pl.DataFrame:
    """Canonicalise the individual weight column for reliable text matching.

    The INSEE detail files store the weight either as a float or as a string
    using a comma or a dot decimal separator, and integer-valued weights may
    appear as ``"4"`` in one file and ``"4,0"`` in another. Both differences
    break a naive text join even though the numbers are identical.

    This function casts the column to ``Float64`` (handling either separator),
    rounds to ``decimals`` places to drop last-bit floating noise, then
    re-formats it to a single canonical ``Utf8`` representation so that equal
    weights compare equal as strings.

    The result is written back into ``col`` (typically ``"IPONDI"``); the
    column name is preserved so downstream matching code is unchanged.
    """
    dtype = df[col].dtype
    if dtype == pl.Utf8:
        as_float = pl.col(col).str.replace(",", ".", literal=True).cast(pl.Float64)
    else:
        as_float = pl.col(col).cast(pl.Float64)
    return df.with_columns(as_float.round(decimals).cast(pl.Utf8).alias(col))


def weight_to_float(col: str = "IPONDI") -> pl.Expr:
    """Return an expression converting the canonical weight string back to a
    float, for use in weighted sums and probability computations."""
    return pl.col(col).str.replace(",", ".", literal=True).cast(pl.Float64)
