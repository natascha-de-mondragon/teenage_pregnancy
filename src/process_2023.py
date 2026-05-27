"""
Process the 2023 AGYW survey data and write data/processed_df_2023_aligned.csv.

Run from the repo root:
    python src/process_2023.py
"""
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from processing_utils import (
    ASSET_VARS,
    EDU_BUCKET_CATS,
    compute_wealth_tertile,
    make_school_complete3,
    recode_level_scol,
)

warnings.filterwarnings("ignore")

SURVEY_YEAR = 2023
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
    "SCHOOLING_STATUS": "scol_status", "SCHOOL_LEVEL": "scol_level",
    "QN_101_In_what_month_were_you": "born_month",
    "QN_101_In_what_year_were_you_": "born_year",
    "QN_102_How_old_were_EAR_OF_BIRTH": "age_completed",
    "education": "educattained", "marital": "married_relshp",
    "eversex": "exper_sexualint", "agefirstsex": "sex_age",
    "_103a_Have_you_ever_attended_s": "attend_scol",
    "q104a": "level_scol",
    "QN_104b_How_long_ha_ince_you_lef": "left_scol",
    "QN_104c_What_are_some_of_the_1": "lack_fees",
    "QN_104c_What_are_some_of_the_2": "got_preg",
    "QN_104c_What_are_some_of_the_3": "got_married",
    "QN_104c_What_are_some_of_the_4": "got_sick",
    "QN_104c_What_are_some_of_the_5": "need_money",
    "QN_104c_What_are_some_of_the_6": "good_std",
    "QN_104c_What_are_some_of_the_7": "int_scol",
    "QN_104c_What_are_some_of_the_8": "other_reas",
    "QN105a_Are_you_able_to_read_a": "read_write",
    "QN_107a_Are_you_currently_mar": "current_married",
    "QN_107b_Does_your_s_he_considers": "more_wife",
    "QN_107c_How_old_is_your_partner": "part_age",
    "A_radio": "radio", "A_television_set": "tv_set", "A_bicycle": "bicycle",
    "A_motor_cycle": "motorcycle", "Your_own_family_home": "own_home",
    "a_cell_phone": "cell_phone", "a_regular_land_line_phone": "reg_phone",
    "a_computer": "computer", "An_income_generating_business": "income_busin",
    "An_indoor_bathroom": "bath_room", "Running_water_either_mpound_of_y": "run_water",
    "Electricity": "electricity", "Car": "car", "Generator": "generator", "Solar": "solar",
    "QN_401_Have_you_ever_had_any_": "life_sex",
    "QN_202_How_old_were_the_very_fir": "sex_age_raw",
    "QN_403_Which_person_did_you_h": "person_sex",
    "QN_205_The_first_ti_r_not_willin": "will_sex",
    "QN_406_The_first_time_you_had": "do_anything",
    "QN_407_The_first_time_you_had": "method_raw",
    "QN_407_The_first_time_you_had1": "male_condom",
    "QN_407_The_first_time_you_had2": "pill",
    "QN_407_The_first_time_you_had3": "injection",
    "QN_407_The_first_time_you_had4": "female_condom",
    "QN_407_The_first_time_you_had5": "withdrawal",
    "QN_407_The_first_time_you_had6": "emergency",
    "QN_407_The_first_time_you_had7": "iud_coil",
    "QN_407_The_first_time_you_had8": "implant",
    "QN_407_The_first_time_you_had9": "avoid_other",
    "QN_409_When_was_the_last_time": "last_sex",
    "QN_211_In_the_past_h_all_these_p": "often_usecondom",
    "QN_412_What_was_your_relation": "relate_sex",
    "QN_213_How_old_was_th_for_the_la": "old_parner",
    "QN_414_Thinking_of_THE_LAST_T": "partuse_condom",
    "QN_215_Thinking_of_sometimes_or_": "some_times",
    "QN_417_Are_you_currently_usin": "current_use",
    "QN_1001_Have_you_ever_been_pr": "been_preg",
    "QN_702_At_what_age_t_for_the_fir": "age_preg",
    "QN_706_Are_you_currently_preg": "current_preg",
    "QN_103_How_did_the_first_preg": "preg_end",
}

METHOD_COLS = [
    "male_condom", "female_condom", "iud_coil", "avoid_other",
    "pill", "withdrawal", "implant", "injection", "emergency",
]

CLASS_CATS = [*(f"P{i}" for i in range(1, 8)), *(f"S{i}" for i in range(1, 7)), "88", "89"]


def load_and_rename(path: Path) -> pd.DataFrame:
    log.info("Loading %s", path)
    df = pd.read_stata(path)
    df = df.rename(columns=RENAME_MAP)
    df = df.drop(columns=[c for c in ["QN_111_Do_you_or_do_ld_own_the_f", "sex_age_raw"] if c in df.columns])
    df.columns = df.columns.str.lower()
    return df


def basic_filter(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(axis=1, how="all")
    before = len(df)
    df = df.dropna(subset=["been_preg"])
    log.info("Rows after filtering to been_preg asked: %d → %d", before, len(df))
    return df


def recode_sex_basics(df: pd.DataFrame) -> pd.DataFrame:
    df["been_preg"] = (
        df["been_preg"].astype("string").str.strip().str.title()
        .map({"Yes": 1, "No": 0})
        .astype("Int64")
    )
    log.info("been_preg distribution:\n%s", df["been_preg"].value_counts(dropna=False).to_string())

    df["life_sex"] = (
        df["life_sex"].astype("string").str.strip().str.lower()
        .map({"yes": 1, "no": 0})
        .astype("Int64")
    )
    log.info("life_sex distribution:\n%s", df["life_sex"].value_counts(dropna=False).to_string())

    df["age_completed"] = pd.to_numeric(df["age_completed"], errors="coerce")
    df["will_sex_binary"] = df["will_sex"].map({1.0: 1, 2.0: 1, 3.0: 0, 4.0: 0}).astype("Int64")
    df["do_anything_binary"] = (
        df["do_anything"].astype("string").str.strip().str.lower()
        .replace({"dont remember": "don't remember"})
        .map({"yes": 1.0, "no": 0.0, "don't remember": np.nan})
        .astype("Float64")
    )
    df.loc[df["life_sex"].eq(0), "do_anything_binary"] = pd.NA
    return df


def recode_person_sex(df: pd.DataFrame) -> pd.DataFrame:
    def classify(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip().lower()
        if "boyfriend" in s:
            return "boyfriend"
        if "husband" in s:
            return "husband"
        return "other"

    df["person_sex_group"] = pd.Categorical(
        df["person_sex"].apply(classify),
        categories=["boyfriend", "husband", "other"],
        ordered=False,
    )
    return df


def recode_methods(df: pd.DataFrame) -> pd.DataFrame:
    sex_mask = df["life_sex"].eq(1)
    for col in METHOD_COLS:
        s = df[col].astype("string").str.strip().str.lower()
        v = pd.Series(pd.NA, index=df.index, dtype="Float64")
        v.loc[sex_mask & s.eq("selected")] = 1.0
        v.loc[sex_mask & s.eq("not selected")] = 2.0
        df[col] = v
    return df


def recode_condom_use(df: pd.DataFrame) -> pd.DataFrame:
    ocon = pd.to_numeric(df["often_usecondom"], errors="coerce")
    df["sex_active_12m"] = ocon.isin([1, 2, 3]).astype(int)
    df["condom_use_ord"] = ocon.map({1: 0, 2: 1, 3: 2})
    df["condom_use_ord_active"] = df["condom_use_ord"].fillna(0) * df["sex_active_12m"]
    df["often_usecondom"] = ocon
    return df


def recode_schooling(df: pd.DataFrame) -> pd.DataFrame:
    status = df["scol_status"].astype("string").str.strip().str.upper()
    status = status.replace({"IN-SCHOOL": "IN SCHOOL", "OUT-OF-SCHOOL": "OUT OF SCHOOL"})
    df["scol_status"] = status
    log.info("scol_status distribution:\n%s", df["scol_status"].value_counts(dropna=False).to_string())

    df.loc[(df["scol_status"] == "OUT OF SCHOOL") & df["attend_scol"].isna(), "attend_scol"] = "No"
    att = df["attend_scol"].astype("string").str.strip().str.title()
    df["attend_scol_binary"] = att.map({"Yes": 1, "No": 0}).astype("Int64")

    drop_reason_cols = ["lack_fees", "got_preg", "got_married", "got_sick", "need_money", "good_std", "int_scol"]
    for col in drop_reason_cols:
        s = df[col].astype("string").str.strip().str.lower()
        df[col] = s.eq("selected").fillna(False).astype("int8")

    df["level_scol_recode"] = recode_level_scol(df["level_scol"])
    # 2023 has no class variable
    df["class_clean"] = pd.Categorical([pd.NA] * len(df), categories=CLASS_CATS)
    return df


def derive_edu_buckets(df: pd.DataFrame) -> pd.DataFrame:
    level = pd.to_numeric(df["level_scol_recode"], errors="coerce")

    years_from_level = pd.Series(np.nan, index=df.index, dtype="float")
    years_from_level[level.isin([2, 3])] = 7
    years_from_level[level.isin([4, 5])] = 11
    years_from_level[level.eq(6)] = 13
    years_from_level[level.isin([7, 8, 9])] = 14
    df["years_school"] = years_from_level.fillna(0).clip(lower=0)

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
    edu_high.loc[is_in.fillna(False)] = from_level_band.loc[is_in.fillna(False)]
    edu_high.loc[is_out.fillna(False)] = from_level_band.loc[is_out.fillna(False)]
    edu_high.loc[edu_high.isna()] = from_level_band.loc[edu_high.isna()]
    edu_high.loc[never_attended] = "None"
    edu_high = edu_high.replace({pd.NA: "Unknown"}).fillna("Unknown")

    df["edu_bucket_highest"] = pd.Categorical(edu_high, categories=EDU_BUCKET_CATS, ordered=True)
    df["school_complete3"] = make_school_complete3(df)
    df["school_complete3_lbl"] = df["school_complete3"].map(
        {0: "in_school", 1: "completed_lower_secondary", 2: "dropped_out"}
    ).astype("string")
    return df


def derive_marriage(df: pd.DataFrame) -> pd.DataFrame:
    # 2023 has no age-at-first-marriage question → force NA for alignment with 2018
    df["age_marry"] = pd.Series(pd.NA, index=df.index, dtype="Float64")
    df["married_by19"] = pd.Series(pd.NA, index=df.index, dtype="Float64")
    df["marriage_timing"] = pd.Series(pd.NA, index=df.index, dtype="object")
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
    df.loc[(df["been_preg"] == 1) & (df["sex_age"] > df["age_preg"]), "sex_age"] = df["age_preg"]
    df.loc[(df["been_preg"] == 1) & (df["age_preg"] > df["age_completed"]), "age_preg"] = df["age_completed"]
    df = df[(df["age_preg"].isna()) | (df["age_preg"] <= 25)].copy()

    df.loc[df["been_preg"] == 1, "diff"] = df["age_months"] / 12 - df["age_preg"]
    df.loc[df["diff"] < -0.5, "diff"] = np.nan
    df["ado_preg"] = ((df["been_preg"] == 1) & (df["age_preg"] <= 19)).fillna(False).astype("int8")
    df["age_cohort"] = np.where(df["age_completed"] <= 14, "10–14", "15–19")
    log.info(
        "ado_preg: %d cases out of %d rows (%.1f%%)",
        df["ado_preg"].sum(), len(df), 100 * df["ado_preg"].mean(),
    )
    return df


def main():
    df = load_and_rename(REPO_ROOT / "data" / "AGYW_dataset_2023.dta")
    df = basic_filter(df)
    df = recode_sex_basics(df)
    df = recode_person_sex(df)
    df = recode_methods(df)
    df = recode_condom_use(df)
    df = recode_schooling(df)
    df = derive_edu_buckets(df)
    df = derive_marriage(df)
    df = derive_age_and_timing(df)
    df = harmonise_ages(df)
    df = compute_wealth_tertile(df, ASSET_VARS)

    cols = [c for c in FEATURES if c in df.columns]
    out = df[cols].copy()
    out["year"] = SURVEY_YEAR

    out_path = REPO_ROOT / "data" / "processed_df_2023_aligned.csv"
    out.to_csv(out_path, index=False)
    log.info("Saved %s (%d rows × %d cols)", out_path, *out.shape)


if __name__ == "__main__":
    main()
