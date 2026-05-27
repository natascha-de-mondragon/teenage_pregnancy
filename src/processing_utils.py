"""
Shared pure-transform helpers for the AGYW teenage pregnancy data pipeline.

Functions here are intentionally free of side-effects so they can be unit-tested
without touching any data files.
"""
import re
import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

ASSET_VARS = [
    "radio", "tv_set", "bicycle", "motorcycle", "own_home",
    "cell_phone", "reg_phone", "computer", "income_busin",
    "bath_room", "run_water", "electricity", "car", "generator", "solar",
]

EDU_BUCKET_CATS = ["None", "less_than_UPE", "UPE", "USE", "higher_than_USE", "Unknown"]


# ---------------------------------------------------------------------------
# Yes / no recoding
# ---------------------------------------------------------------------------

def recode_yes_no(series: pd.Series) -> pd.Series:
    """Recode a yes/no column to 1/0 regardless of string or 1/2 numeric encoding."""
    s = series.copy()
    if (
        pd.api.types.is_object_dtype(s)
        or isinstance(s.dtype, pd.CategoricalDtype)
        or pd.api.types.is_string_dtype(s)
    ):
        s2 = (
            s.astype("string").str.strip().str.lower()
            .replace({"nan": pd.NA, "": pd.NA})
        )
        return s2.map({"yes": 1, "y": 1, "no": 0, "n": 0}).astype("Int64")
    s_num = pd.to_numeric(s, errors="coerce")
    return s_num.map({1: 1, 2: 0}).astype("Int64")


# ---------------------------------------------------------------------------
# Schooling level normalisation and recoding
# ---------------------------------------------------------------------------

def norm_level_str(x) -> str:
    """Normalise a raw schooling-level string to a canonical form for exact_map lookup."""
    if pd.isna(x):
        return np.nan
    s = str(x).upper()
    s = s.replace("‘", "'").replace("’", "'").replace("–", "-").replace("—", "-")
    s = s.replace(" ", " ").replace(" ", " ").replace(" ", " ")
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*-\s*", " - ", s)
    s = re.sub(r"\bO\s*LEVEL\b", "O' LEVEL", s)
    s = re.sub(r"\bA\s*LEVEL\b", "A' LEVEL", s)
    s = re.sub(r"\bP\s*\.?\s*([1-7])\b", r"P.\1", s)
    s = re.sub(r"\bS\s*\.?\s*([1-6])\b", r"S.\1", s)
    s = re.sub(r"(P\.[1-7])\s*-\s*(P\.[1-7])", r"\1 - \2", s)
    s = re.sub(r"(S\.[1-6])\s*-\s*(S\.[1-6])", r"\1 - \2", s)
    if s == "OTHER":
        s = "OTHER (SPECIFY)"
    return s


_EXACT_LEVEL_MAP = {
    "PRIMARY (P.1 - P.7)": 2,
    "PRIMARY PROFESSIONAL": 3,
    "O' LEVEL (S.1 - S.4)": 4,
    "O' LEVEL PROFESSIONAL": 5,
    "A' LEVEL (S.5 - S.6)": 6,
    "UNIVERSITY": 7,
    "OTHER TERTIARY (AFTER S.6)": 8,
    "OTHER (SPECIFY)": 9,
}


def recode_level_scol(series: pd.Series) -> pd.Series:
    """Map a schooling-level string series to a numeric code (2–9, Int64)."""
    lvl_norm = series.apply(norm_level_str)
    codes = pd.Series(pd.NA, index=series.index, dtype="Int64")

    for key, val in _EXACT_LEVEL_MAP.items():
        codes[lvl_norm == key] = val

    need = codes.isna()
    # Cast the full series to string so .str accessor works even on all-NaN input
    # and index alignment with `need` is preserved.
    ln = lvl_norm.astype("string")
    codes.loc[need & ln.str.contains(r"\bUNIVERSITY\b", na=False)] = 7
    codes.loc[need & ln.str.contains(r"\b(?:COLLEGE|TERTIARY|INSTITUTE|POLYTECH|DIPLOMA|DEGREE)\b", na=False)] = 8
    codes.loc[need & ln.str.contains(r"\bA' LEVEL\b|\bS\.[5-6]\b|\bFORM\s*[5-6]\b", na=False)] = 6
    codes.loc[need & ln.str.contains(r"\bO' LEVEL\b|\bS\.[1-4]\b|\bFORM\s*[1-4]\b", na=False)] = 4
    codes.loc[need & ln.str.contains(r"O' LEVEL.*PROF|PROFESSIONAL", na=False)] = 5
    codes.loc[need & ln.str.contains(r"\bPRIMARY\b|\bPRI\b|\bP\.[1-7]\b", na=False)] = 2
    codes.loc[need & ln.str.contains(r"PRIMARY.*PROF|PROFESSIONAL", na=False)] = 3

    n_coded = codes.notna().sum()
    log.debug("recode_level_scol: %d / %d values coded", n_coded, len(series))
    return codes


# ---------------------------------------------------------------------------
# Education bucket → school completion outcome
# ---------------------------------------------------------------------------

def make_school_complete3(df: pd.DataFrame) -> pd.Series:
    """
    Derive a 3-way schooling outcome from scol_status and edu_bucket_highest.

    Returns Int64 series: 0=in school, 1=completed ≥O-level, 2=dropped out.
    """
    status = (
        df["scol_status"].astype("string").str.strip().str.lower()
        .where(lambda x: x.isin(["in school", "out of school"]), other=pd.NA)
    )
    ebh = df["edu_bucket_highest"].astype("string")
    out = pd.Series(pd.NA, index=df.index, dtype="Int64")
    out[status.eq("in school")] = 0
    out[status.eq("out of school") & ebh.isin(["USE", "higher_than_USE"])] = 1
    out[status.eq("out of school") & ebh.isin(["less_than_UPE", "UPE"])] = 2
    out[status.isna() & ebh.isin(["USE", "higher_than_USE"])] = 1
    out[status.isna() & ebh.isin(["less_than_UPE", "UPE"])] = 2
    return out


# ---------------------------------------------------------------------------
# Marriage / pregnancy timing
# ---------------------------------------------------------------------------

def marriage_any_vs_cases(row) -> str:
    """Classify marriage/pregnancy timing into one of five mutually exclusive categories."""
    ap = pd.to_numeric(row["age_preg"], errors="coerce")
    am = pd.to_numeric(row["age_marry"], errors="coerce")
    if pd.isna(ap) and pd.isna(am):
        return "never_marry_or_preg"
    if pd.notna(ap) and pd.isna(am):
        return "preg_never_marry"
    if pd.isna(ap) and pd.notna(am):
        return "marry_never_preg"
    return "marry_after_preg" if am > ap else "marry_before_preg"


# ---------------------------------------------------------------------------
# Wealth index (PCA) + tertiles
# ---------------------------------------------------------------------------

def compute_wealth_tertile(df: pd.DataFrame, asset_vars: list) -> pd.DataFrame:
    """
    Add wealth_index (PCA score) and wealth_tertile (Low/Medium/High) columns.

    Modifies df in-place and returns it.
    """
    df_assets = df[asset_vars].replace(98, np.nan).copy()
    df_assets = df_assets.applymap(lambda x: 1 if x == 1 else 0).fillna(0)

    scaler = StandardScaler()
    assets_scaled = scaler.fit_transform(df_assets)
    pca = PCA(n_components=1)
    df["wealth_index"] = pca.fit_transform(assets_scaled)[:, 0]

    w = df["wealth_index"]
    mask = w.notna()
    ranks = w[mask].rank(method="first")
    labels = ["Low", "Medium", "High"]
    tertiles = pd.qcut(ranks, 3, labels=labels)
    df["wealth_tertile"] = pd.NA
    df.loc[mask, "wealth_tertile"] = tertiles.astype(str)
    df["wealth_tertile"] = pd.Categorical(df["wealth_tertile"], categories=labels, ordered=True)

    log.debug(
        "wealth_tertile distribution: %s",
        df["wealth_tertile"].value_counts().to_dict(),
    )
    return df
