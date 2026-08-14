"""
Configuration for the synthetic census pipeline.

The pipeline is parameterised by **census year** and by an optional
**geographic perimeter** (regions and/or departments). The year is mandatory
because several INSEE variable and column names depend on it; the perimeter
is optional and, when both regions and departments are given, they are
combined as a **union** (an individual is kept if its region is selected *or*
its department is selected).

``CensusSchema`` resolves, for a given year, every year-dependent name the
rest of the package needs: the household-margin column prefix, the
socio-professional variable, the mapping from that variable to the eight
margin groups, the reference-person CSP variable used for MOBSCO matching,
and the ``DIPL`` variable name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Sentinels and global constants
# ---------------------------------------------------------------------------
IRIS_UNKNOWN: str = "ZZZZZZZZZ"        # "commune not split into IRIS"
ARM_NONE: str = "ZZZZZ"                 # ARM sentinel meaning "not applicable"
COMMUNE_UNKNOWN: str = "ZZZZZ"          # INSEE sentinel for a missing commune (DCLT/DCFLT/DCETUF/DCETUE)
FOREIGN_FRANCE_CODE: str = "99999"      # in DCFLT/DCETUE: "metropolitan France"
OVERSEAS_REGIONS: list[str] = ["01", "02", "03", "04"]
ROOT_SEED: int = 42

# Eight-group labels "1".."8" in canonical INSEE order:
# 1 farmers, 2 craftsmen, 3 managers, 4 intermediate, 5 employees, 6 workers,
# 7 retired, 8 other inactive.
MARGIN_GROUPS: list[str] = ["1", "2", "3", "4", "5", "6", "7", "8"]


def _yy(year: int) -> str:
    """Two-digit year used in INSEE margin column names (2007 -> '07')."""
    return f"{year % 100:02d}"


@dataclass(frozen=True)
class CensusSchema:
    """Year-dependent variable and column names, resolved from the year."""

    year: int
    csp_var: str = field(init=False)
    ref_csp_var: str = field(init=False)
    dipl_var: str = field(init=False)
    margin_prefix: str = field(init=False)
    margin_columns: list[str] = field(init=False)
    csp_to_group: dict[str, str] = field(init=False)

    def __post_init__(self) -> None:
        yy = _yy(self.year)
        object.__setattr__(self, "margin_prefix", f"C{yy}")

        # Diploma renamed DIPL_15 for the 2013-2016 millésimes.
        dipl = "DIPL_15" if self.year in (2013, 2014, 2015, 2016) else "DIPL"
        object.__setattr__(self, "dipl_var", dipl)

        if self.year >= 2022:
            # From 2022 CS1 is gone: the individual file uses GS (6 posts) but
            # the household margins are built on STAT_GSEC. Households are
            # grouped on STAT_GSEC, mapped back to the historical 8-post split.
            object.__setattr__(self, "csp_var", "STAT_GSEC")
            object.__setattr__(self, "ref_csp_var", "GSM")
            object.__setattr__(
                self,
                "margin_columns",
                [
                    f"C{yy}_MEN_STAT_GSEC11_21",
                    f"C{yy}_MEN_STAT_GSEC12_22",
                    f"C{yy}_MEN_STAT_GSEC13_23",
                    f"C{yy}_MEN_STAT_GSEC14_24",
                    f"C{yy}_MEN_STAT_GSEC15_25",
                    f"C{yy}_MEN_STAT_GSEC16_26",
                    f"C{yy}_MEN_STAT_GSEC32",
                    f"C{yy}_MEN_STAT_GSEC40",
                ],
            )
            mapping = {
                "11": "1", "21": "1",
                "12": "2", "22": "2",
                "13": "3", "23": "3",
                "14": "4", "24": "4",
                "15": "5", "25": "5",
                "16": "6", "26": "6",
                "32": "7",
            }
            # Everything else (19, 20, 29, 31, 33, 40, 99, ZZ, ...) -> group 8.
            object.__setattr__(self, "csp_to_group", mapping)
        else:
            # 2007-2021: CS1 in 8 posts already matches margin groups 1..8.
            object.__setattr__(self, "csp_var", "CS1")
            object.__setattr__(self, "ref_csp_var", "CSM")
            object.__setattr__(
                self, "margin_columns", [f"C{yy}_MEN_CS{i}" for i in range(1, 9)]
            )
            object.__setattr__(self, "csp_to_group", {})

    @property
    def total_margin_column(self) -> str:
        """Total household count column, e.g. 'C07_MEN'."""
        return f"{self.margin_prefix}_MEN"

    @property
    def group_label_column(self) -> str:
        """Name of the derived 8-post group column added to the census."""
        return "MARGIN_GROUP"

    def group_to_margin_column(self) -> dict[str, str]:
        """Map each group label '1'..'8' to its margin column for this year."""
        return dict(zip(MARGIN_GROUPS, self.margin_columns))


@dataclass
class PipelineConfig:
    """Runtime configuration for a full synthetic-census build.

    Parameters
    ----------
    year : census millésime (mandatory). Drives every year-dependent name.
    regions, depts : optional perimeter, combined as a **union** (kept if
        REGION in regions OR DEPT in depts). Both None -> whole territory.
    root_seed : root random seed (stochastic stages derive from it).
    margin_tolerance : multiplicative slack on IRIS household margins.
    ipondi_round_decimals : decimals kept when canonicalising the weight.
    """

    year: int
    regions: Optional[list[str]] = None
    depts: Optional[list[str]] = None
    overseas_regions: list[str] = field(default_factory=lambda: list(OVERSEAS_REGIONS))
    root_seed: int = ROOT_SEED
    margin_tolerance: float = 1.05
    ipondi_round_decimals: int = 10

    def __post_init__(self) -> None:
        if self.year is None:
            raise ValueError("`year` is mandatory (e.g. year=2008).")
        self.schema = CensusSchema(self.year)

    def keeps_whole_territory(self) -> bool:
        return not self.regions and not self.depts
