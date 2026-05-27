"""
Process the 2018 AGYW survey data and write data/processed_df_2018_aligned.csv.

Run from the repo root:
    python src/process_2018.py
"""
import logging
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from processing_utils import (
    ASSET_VARS,
    EDU_BUCKET_CATS,
    compute_wealth_tertile,
    make_school_complete3,
    marriage_any_vs_cases,
    recode_level_scol,
    recode_yes_no,
)

warnings.filterwarnings("ignore")

SURVEY_YEAR = 2018
REPO_ROOT = Path(__file__).parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

FEATURES = [
    "ado_preg", "been_preg", "age_completed", "age_cohort", "age_preg", "sex_age",
    "will_sex_binary", "do_anything_binary", "age_marry", "married_by19", "marriage_timing",
    "school_complete3_lbl", "years_school", "level_scol_recode", "edu_bucket_highest",
    "person_sex_group", "male_condom", "female_condom", "iud_coil", "avoid_other",
    "pill", "withdrawal", "implant", "often_usecondom", "sex_active_12m",
    "condom_use_ord", "condom_use_ord_active", "wealth_tertile",
]

RENAME_MAP = {
    "Scol_status": "scol_status", "Scol_level": "scol_level",
    "Born_month": "born_month", "Born_year": "born_year",
    "Age_completed": "age_completed", "Attend_scol": "attend_scol",
    "Level_scol": "level_scol", "Left_scol": "left_scol",
    "Lack_fees": "lack_fees", "Got_preg": "got_preg", "Got_married": "got_married",
    "Got_sick": "got_sick", "Need_money": "need_money", "Good_std": "good_std",
    "Int_scol": "int_scol", "Other_reas": "other_reas",
    "Read_write": "read_write", "Current_married": "current_married",
    "More_wife": "more_wife", "Part_age": "part_age",
    "Radio": "radio", "Tv_set": "tv_set", "Bicycle": "bicycle",
    "Motorcycle": "motorcycle", "Own_home": "own_home", "Cell_phone": "cell_phone",
    "Reg_phone": "reg_phone", "Computer": "computer", "Income_busin": "income_busin",
    "Bath_room": "bath_room", "Run_water": "run_water", "electricity": "electricity",
    "car": "car", "generator": "generator", "solar": "solar",
    "Life_sex": "life_sex", "Sex_age": "sex_age", "Person_sex": "person_sex",
    "Will_sex": "will_sex", "Do_anything": "do_anything",
    "Male_condom": "male_condom", "pill": "pill", "Injection": "injection",
    "Female_condom": "female_condom", "withdrawal": "withdrawal",
    "Emergency": "emergency", "Iud_coil": "iud_coil", "implant": "implant",
    "Avoid_other": "avoid_other",
    "Last_sex": "last_sex", "Often_usecondom": "often_usecondom",
    "Relate_sex": "relate_sex", "Old_parner": "old_parner",
    "Partuse_condom": "partuse_condom", "Some_times": "some_times",
    "Worry_preg": "worry_preg", "Under_influe": "under_influe",
    "Been_preg": "been_preg", "Age_preg": "age_preg", "Preg_end": "preg_end",
    "Current_use": "current_use",
    "educattained": "educattained", "married_relshp": "married_relshp",
    "Exper_sexualint": "exper_sexualint",
}

METHOD_COLS = [
    "male_condom", "female_condom", "iud_coil", "avoid_other",
    "pill", "withdrawal", "implant", "injection", "emergency",
]

CLASS_CATS = [*(f"P{i}" for i in range(1, 8)), *(f"S{i}" for i in range(1, 7)), "88", "89"]


def load_and_rename(path: Path) -> pd.DataFrame:
    log.info("Loading %s", path)
    df = pd.read_stata(path)
    raw_cols = set(df.columns)
    df = df.rename(columns=RENAME_MAP)
    present = [k for k in RENAME_MAP if k in raw_cols]
    missing = [k for k in RENAME_MAP if k not in raw_cols]
    log.info("Rename map: %d matched, %d not found in raw data", len(present), len(missing))
    if missing:
        log.debug("Missing rename keys: %s", missing[:20])
    return df


def basic_filter(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.lower()
    df = df.drop(columns=[
        c for c in ["qn_111_do_you_or_do_ld_own_the_f", "sex_age_raw"]
        if c in df.columns
    ])
    df = df.dropna(axis=1, how="all")
    before = len(df)
    df = df.dropna(subset=["been_preg"])
    log.info("Rows after filtering to been_preg asked: %d → %d", before, len(df))
    return df


def recode_sex_basics(df: pd.DataFrame) -> pd.DataFrame:
    df["been_preg"] = recode_yes_no(df["been_preg"])
    log.info("been_preg distribution:\n%s", df["been_preg"].value_counts(dropna=False).to_string())

    life = df["life_sex"]
    life_bin = pd.to_numeric(life, errors="coerce").map({1: 1, 2: 0}).astype("Int64")
    if life_bin.isna().all():
        life_bin = (
            life.astype("string").str.strip().str.lower()
            .map({"yes": 1, "no": 0}).astype("Int64")
        )
    df["life_sex"] = life_bin

    df["age_completed"] = pd.to_numeric(df["age_completed"], errors="coerce")
    df["will_sex_binary"] = df["will_sex"].map({1.0: 1, 2.0: 1, 3.0: 0, 4.0: 0}).astype("Int64")

    s = df["do_anything"]
    if pd.api.types.is_numeric_dtype(s):
        df["do_anything_binary"] = (
            pd.to_numeric(s, errors="coerce")
            .map({1: 1.0, 2: 0.0, 3: np.nan})
            .astype("Float64")
        )
    else:
        ss = s.astype("string").str.strip().str.lower()
        df["do_anything_binary"] = (
            ss.map({"yes": 1.0, "no": 0.0, "don't remember": np.nan, "dont remember": np.nan})
            .astype("Float64")
        )
    df.loc[df["life_sex"].eq(0), "do_anything_binary"] = pd.NA
    return df


def recode_person_sex(df: pd.DataFrame) -> pd.DataFrame:
    num = pd.to_numeric(df["person_sex"], errors="coerce")
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    if num.notna().sum() > 0:
        out.loc[num.eq(1)] = "boyfriend"
        out.loc[num.eq(2)] = "husband"
        out.loc[num.notna() & ~num.isin([1, 2])] = "other"
    else:
        s = df["person_sex"].astype("string").str.strip().str.lower()
        out.loc[s.str.contains("boyfriend", na=False)] = "boyfriend"
        out.loc[s.str.contains("husband", na=False)] = "husband"
        out.loc[s.notna() & out.isna()] = "other"
    df["person_sex_group"] = pd.Categorical(
        out, categories=["boyfriend", "husband", "other"], ordered=False
    )
    return df


def recode_methods(df: pd.DataFrame) -> pd.DataFrame:
    sex_mask = pd.to_numeric(df["life_sex"], errors="coerce").eq(1)
    for col in METHOD_COLS:
        raw = df[col]
        v = pd.Series(pd.NA, index=df.index, dtype="Float64")
        raw_num = pd.to_numeric(raw, errors="coerce")
        v.loc[sex_mask & raw_num.eq(1)] = 1.0
        v.loc[sex_mask & raw_num.eq(2)] = 2.0
        raw_str = raw.astype("string").str.strip().str.lower()
        miss = v.isna()
        v.loc[miss & sex_mask & raw_str.eq("selected")] = 1.0
        v.loc[miss & sex_mask & raw_str.eq("not selected")] = 2.0
        df[col] = v
    log.debug(
        "male_condom after recode: %s",
        df["male_condom"].value_counts(dropna=False).to_dict(),
    )
    return df


def recode_condom_use(df: pd.DataFrame) -> pd.DataFrame:
    ocon = pd.to_numeric(df["often_usecondom"], errors="coerce")
    df["sex_active_12m"] = ocon.isin([1, 2, 3]).astype(int)
    df["condom_use_ord"] = ocon.map({1: 0, 2: 1, 3: 2})
    df["condom_use_ord_active"] = df["condom_use_ord"].fillna(0) * df["sex_active_12m"]
    df["often_usecondom"] = ocon
    return df


def recode_schooling(df: pd.DataFrame) -> pd.DataFrame:
    df["scol_status"] = df["scol_status"].astype("string").str.strip().str.upper()
    df.loc[
        (df["scol_status"] == "OUT OF SCHOOL") & df["attend_scol"].isna(),
        "attend_scol",
    ] = "No"
    att = df["attend_scol"].astype("string").str.strip().str.title()
    df["attend_scol_binary"] = att.map({"Yes": 1, "No": 0}).astype("Int64")

    drop_reason_cols = ["lack_fees", "got_preg", "got_married", "got_sick", "need_money", "good_std", "int_scol"]
    for col in drop_reason_cols:
        s = df[col].astype("string").str.strip().str.lower()
        df[col] = s.eq("selected").fillna(False).astype("int8")

    df["level_scol_recode"] = recode_level_scol(df["level_scol"])

    # class_clean (2018 has this variable; 2023 does not)
    if "class" in df.columns:
        s = df["class"].astype(str).str.strip().str.upper()
        s = s.replace({"": np.nan, "NAN": np.nan, "NA": np.nan, "NONE": np.nan})
        norm = s.str.replace(r"[^A-Z0-9]", "", regex=True)
        norm = norm.str.replace(r"^UNIVERSITY(?:88)?$", "88", regex=True)
        norm = norm.str.replace(r"^OTHERTERTIARY(?:89)?$", "89", regex=True)
        norm = norm.str.replace(r"^Y\d+$", "88", regex=True)
        norm = norm.str.replace(r"^J([1-3])$", r"S\1", regex=True)
        norm = norm.str.replace(r"^5([1-6])$", r"S\1", regex=True)
        norm = norm.mask(norm.eq("S"))
        norm = norm.str.replace(r"^98$", "89", regex=True)
        m_bare = norm.str.fullmatch(r"[1-7]").fillna(False)
        norm.loc[m_bare] = "P" + norm.loc[m_bare]
        norm = norm.mask(norm.str.fullmatch(r"\d+") & ~norm.isin({"88", "89"}))
        df["class_clean"] = pd.Categorical(norm.where(norm.isin(set(CLASS_CATS))), categories=CLASS_CATS)
    else:
        df["class_clean"] = pd.Categorical([pd.NA] * len(df), categories=CLASS_CATS)

    df["current_married_binary"] = df["current_married"].map({
        "MARRIED/UNION": 1, "DIVORCED/SEPARATED": 0,
        "WIDOWED": 0, "NEVER MARRIED": 0, "IN RELATIONSHIP BUT NOT MARRIED": 0,
    })
    df["been_married_binary"] = df["current_married"].map({
        "MARRIED/UNION": 1, "DIVORCED/SEPARATED": 1, "WIDOWED": 1,
        "NEVER MARRIED": 0, "IN RELATIONSHIP BUT NOT MARRIED": 0,
    })

    df["age_marry"] = pd.to_numeric(df.get("age_marry"), errors="coerce")
    q1, q3 = df["age_marry"].quantile(0.25), df["age_marry"].quantile(0.75)
    df["age_marry"] = df["age_marry"].clip(lower=(q1 - 1.5 * (q3 - q1)))
    df["married_by19"] = (df["age_marry"] <= 19).astype(float)
    return df


def derive_edu_buckets(df: pd.DataFrame) -> pd.DataFrame:
    years_from_class = {
        **{f"P{i}": i - 1 for i in range(1, 8)},
        "S1": 7, "S2": 8, "S3": 9, "S4": 10, "S5": 11, "S6": 12, "88": 13, "89": 13,
    }
    cls = df["class_clean"].astype("string").str.strip()
    years_cls = cls.map(years_from_class).astype("float")

    level = pd.to_numeric(df["level_scol_recode"], errors="coerce")
    years_from_level = pd.Series(np.nan, index=df.index, dtype="float")
    years_from_level[level.isin([2, 3])] = 7
    years_from_level[level.isin([4, 5])] = 11
    years_from_level[level.eq(6)] = 13
    years_from_level[level.isin([7, 8, 9])] = 14
    df["years_school"] = pd.concat([years_cls, years_from_level], axis=1).max(axis=1).fillna(0).clip(lower=0)

    class_to_band = {
        **{f"P{i}": "less_than_UPE" for i in range(1, 8)},
        **{f"S{i}": "UPE" for i in range(1, 5)},
        **{f"S{i}": "USE" for i in range(5, 7)},
        "88": "higher_than_USE", "89": "higher_than_USE",
    }
    from_class_band = df["class_clean"].map(class_to_band)
    level_band_map = {2: "UPE", 3: "UPE", 4: "USE", 5: "USE", 6: "higher_than_USE", 7: "higher_than_USE", 8: "higher_than_USE", 9: "higher_than_USE"}
    from_level_band = level.map(level_band_map)

    attend_norm = df["attend_scol"].astype("string").str.strip().str.lower() if "attend_scol" in df.columns else pd.Series(False, index=df.index)
    never_attended = attend_norm.isin(["no", "0", "false"])

    status_norm = (
        df["scol_status"].astype("string").str.strip().str.lower()
        .where(lambda x: x.isin(["in school", "out of school"]), other=pd.NA)
    )
    is_in = status_norm.eq("in school")
    is_out = status_norm.eq("out of school")

    edu_high = pd.Series(pd.NA, index=df.index, dtype="object")
    idx = is_in.fillna(False)
    edu_high.loc[idx] = from_class_band.loc[idx]
    edu_high.loc[idx & edu_high.isna() & from_level_band.notna()] = from_level_band.loc[idx & edu_high.isna() & from_level_band.notna()]
    idx = is_out.fillna(False)
    edu_high.loc[idx] = from_level_band.loc[idx]
    edu_high.loc[idx & edu_high.isna() & from_class_band.notna()] = from_class_band.loc[idx & edu_high.isna() & from_class_band.notna()]
    idx = ~(is_in.fillna(False) | is_out.fillna(False))
    edu_high.loc[idx] = from_level_band.loc[idx]
    edu_high.loc[idx & edu_high.isna() & from_class_band.notna()] = from_class_band.loc[idx & edu_high.isna() & from_class_band.notna()]
    edu_high.loc[never_attended] = "None"

    df["edu_bucket_highest"] = pd.Categorical(
        pd.Series(edu_high).fillna("Unknown"), categories=EDU_BUCKET_CATS, ordered=True
    )
    df["school_complete3"] = make_school_complete3(df)
    df["school_complete3_lbl"] = df["school_complete3"].map(
        {0: "in_school", 1: "completed_lower_secondary", 2: "dropped_out"}
    ).astype("string")
    return df


def derive_age_and_timing(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[df["born_year"].isna(), "born_year"] = SURVEY_YEAR - df["age_completed"]
    df.loc[df["born_month"].isna(), "born_month"] = 7
    df["born_month"] = pd.to_numeric(df["born_month"], errors="coerce")
    df["born_year"] = pd.to_numeric(df["born_year"], errors="coerce")
    df.loc[~df["born_month"].between(1, 12), "born_month"] = np.nan
    df.loc[~df["born_year"].between(SURVEY_YEAR - 24, SURVEY_YEAR - 10), "born_year"] = np.nan
    df["born_month"] = df["born_month"].fillna(7)
    df["age_months"] = (SURVEY_YEAR - df["born_year"]) * 12 + (7 - df["born_month"])

    df["age_preg"] = pd.to_numeric(df["age_preg"], errors="coerce")
    df.loc[~df["age_preg"].between(10, 24), "age_preg"] = np.nan
    df.loc[(df["been_preg"] == 1) & df["age_preg"].isna(), "age_preg"] = df["age_completed"] - 1
    return df


def harmonise_ages(df: pd.DataFrame) -> pd.DataFrame:
    df["sex_age"] = pd.to_numeric(df["sex_age"], errors="coerce")
    df.loc[(df["been_preg"] == 0) & df["age_preg"].notna(), "been_preg"] = 1
    df.loc[~df["sex_age"].between(5, 24), "sex_age"] = np.nan
    df.loc[(df["been_preg"] == 1) & df["sex_age"].isna(), "sex_age"] = df["age_preg"]
    mask = (df["been_preg"] == 1) & df["sex_age"].notna() & df["age_preg"].notna()
    df.loc[mask & (df["sex_age"] > df["age_preg"]), "sex_age"] = df["age_preg"]
    df.loc[(df["been_preg"] == 1) & (df["age_preg"] > df["age_completed"]), "age_preg"] = df["age_completed"]
    df.loc[df["age_marry"] > df["age_completed"], "age_marry"] = df["age_completed"]
    df.loc[(df["been_married_binary"] == 0) & df["age_marry"].notna(), "age_marry"] = np.nan
    df = df[(df["age_preg"].isna()) | (df["age_preg"] <= 25)].copy()

    df.loc[df["been_preg"] == 1, "diff"] = df["age_months"] / 12 - df["age_preg"]
    df.loc[df["diff"] < -0.5, "diff"] = np.nan
    df["ado_preg"] = ((df["been_preg"] == 1) & (df["age_preg"] <= 19)).fillna(False).astype("int8")
    df["age_cohort"] = np.where(df["age_completed"] <= 14, "10–14", "15–19")
    df["marriage_timing"] = df.apply(marriage_any_vs_cases, axis=1)
    log.info(
        "ado_preg: %d cases out of %d rows (%.1f%%)",
        df["ado_preg"].sum(), len(df), 100 * df["ado_preg"].mean(),
    )
    return df


def main():
    df = load_and_rename(REPO_ROOT / "data" / "AGYW_dataset_2018.dta")
    df = basic_filter(df)
    df = recode_sex_basics(df)
    df = recode_person_sex(df)
    df = recode_methods(df)
    df = recode_condom_use(df)
    df = recode_schooling(df)
    df = derive_edu_buckets(df)
    df = derive_age_and_timing(df)
    df = harmonise_ages(df)
    df = compute_wealth_tertile(df, ASSET_VARS)

    cols = [c for c in FEATURES if c in df.columns]
    out = df[cols].copy()
    out["year"] = SURVEY_YEAR

    out_path = REPO_ROOT / "data" / "processed_df_2018_aligned.csv"
    out.to_csv(out_path, index=False)
    log.info("Saved %s (%d rows × %d cols)", out_path, *out.shape)


if __name__ == "__main__":
    main()
