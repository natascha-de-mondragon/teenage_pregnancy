"""
Unit tests for src/processing_utils.py.

Run from the repo root:
    pytest tests/
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from processing_utils import (
    make_school_complete3,
    marriage_any_vs_cases,
    norm_level_str,
    recode_level_scol,
    recode_yes_no,
)


# ---------------------------------------------------------------------------
# recode_yes_no
# ---------------------------------------------------------------------------

class TestRecodeYesNo:
    def test_numeric_1_2_coding(self):
        result = recode_yes_no(pd.Series([1, 2, 1, 2]))
        assert list(result) == [1, 0, 1, 0]

    def test_string_yes_no(self):
        result = recode_yes_no(pd.Series(["Yes", "No", "yes", "NO", "YES"]))
        assert list(result) == [1, 0, 1, 0, 1]

    def test_short_y_n(self):
        result = recode_yes_no(pd.Series(["y", "n"]))
        assert list(result) == [1, 0]

    def test_numeric_nan_becomes_na(self):
        result = recode_yes_no(pd.Series([1, np.nan, 2]))
        assert result.iloc[0] == 1
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == 0

    def test_string_nan_becomes_na(self):
        result = recode_yes_no(pd.Series(["Yes", None, "No"]))
        assert result.iloc[0] == 1
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == 0

    def test_unexpected_values_become_na(self):
        result = recode_yes_no(pd.Series(["maybe", "3", ""]))
        assert result.isna().all()

    def test_returns_int64_dtype(self):
        result = recode_yes_no(pd.Series([1, 2]))
        assert result.dtype == "Int64"


# ---------------------------------------------------------------------------
# norm_level_str
# ---------------------------------------------------------------------------

class TestNormLevelStr:
    def test_nan_returns_nan(self):
        assert pd.isna(norm_level_str(np.nan))
        assert pd.isna(norm_level_str(None))

    def test_primary_range(self):
        assert norm_level_str("Primary (P.1 - P.7)") == "PRIMARY (P.1 - P.7)"

    def test_o_level_apostrophe_fix(self):
        assert norm_level_str("O Level (S.1 - S.4)") == "O' LEVEL (S.1 - S.4)"

    def test_a_level_apostrophe_fix(self):
        assert norm_level_str("A Level (S.5 - S.6)") == "A' LEVEL (S.5 - S.6)"

    def test_other_becomes_other_specify(self):
        assert norm_level_str("other") == "OTHER (SPECIFY)"

    def test_university_exact(self):
        assert norm_level_str("university") == "UNIVERSITY"

    def test_whitespace_collapsed(self):
        assert norm_level_str("  primary   (P.1  -  P.7)  ") == "PRIMARY (P.1 - P.7)"

    def test_s_level_normalisation(self):
        # "S 1" → "S.1"
        result = norm_level_str("O Level (S 1 - S 4)")
        assert "S.1" in result and "S.4" in result

    def test_p_level_normalisation(self):
        result = norm_level_str("Primary P 3 - P 7")
        assert "P.3" in result and "P.7" in result


# ---------------------------------------------------------------------------
# recode_level_scol
# ---------------------------------------------------------------------------

class TestRecodeLevelScol:
    def test_primary_exact_match(self):
        s = pd.Series(["Primary (P.1 - P.7)"])
        assert recode_level_scol(s).iloc[0] == 2

    def test_o_level_exact_match(self):
        s = pd.Series(["O Level (S.1 - S.4)"])
        assert recode_level_scol(s).iloc[0] == 4

    def test_university_exact(self):
        s = pd.Series(["UNIVERSITY"])
        assert recode_level_scol(s).iloc[0] == 7

    def test_university_fuzzy(self):
        s = pd.Series(["University of Kampala"])
        assert recode_level_scol(s).iloc[0] == 7

    def test_college_maps_to_other_tertiary(self):
        s = pd.Series(["Makerere College"])
        assert recode_level_scol(s).iloc[0] == 8

    def test_nan_returns_na(self):
        s = pd.Series([np.nan])
        assert pd.isna(recode_level_scol(s).iloc[0])

    def test_returns_int64_dtype(self):
        s = pd.Series(["UNIVERSITY", np.nan])
        result = recode_level_scol(s)
        assert result.dtype == "Int64"

    def test_multiple_values(self):
        s = pd.Series(["Primary (P.1 - P.7)", "O Level (S.1 - S.4)", "UNIVERSITY", np.nan])
        result = recode_level_scol(s)
        assert list(result[:3]) == [2, 4, 7]
        assert pd.isna(result.iloc[3])


# ---------------------------------------------------------------------------
# make_school_complete3
# ---------------------------------------------------------------------------

def _school_df(status, edu_bucket):
    return pd.DataFrame({
        "scol_status": [status],
        "edu_bucket_highest": pd.Categorical(
            [edu_bucket],
            categories=["None", "less_than_UPE", "UPE", "USE", "higher_than_USE", "Unknown"],
            ordered=True,
        ),
    })


class TestMakeSchoolComplete3:
    def test_in_school_returns_0(self):
        df = _school_df("in school", "UPE")
        assert make_school_complete3(df).iloc[0] == 0

    def test_in_school_case_insensitive(self):
        df = _school_df("IN SCHOOL", "USE")
        assert make_school_complete3(df).iloc[0] == 0

    def test_out_completed_use_returns_1(self):
        df = _school_df("out of school", "USE")
        assert make_school_complete3(df).iloc[0] == 1

    def test_out_completed_higher_returns_1(self):
        df = _school_df("out of school", "higher_than_USE")
        assert make_school_complete3(df).iloc[0] == 1

    def test_out_dropout_upe_returns_2(self):
        df = _school_df("out of school", "UPE")
        assert make_school_complete3(df).iloc[0] == 2

    def test_out_dropout_less_than_upe_returns_2(self):
        df = _school_df("out of school", "less_than_UPE")
        assert make_school_complete3(df).iloc[0] == 2

    def test_unknown_status_with_high_edu_returns_1(self):
        df = _school_df("unknown_value", "USE")
        assert make_school_complete3(df).iloc[0] == 1

    def test_unknown_status_unknown_edu_returns_na(self):
        df = _school_df("unknown_value", "Unknown")
        assert pd.isna(make_school_complete3(df).iloc[0])


# ---------------------------------------------------------------------------
# marriage_any_vs_cases
# ---------------------------------------------------------------------------

class TestMarriageAnyVsCases:
    def test_neither_preg_nor_marry(self):
        row = {"age_preg": np.nan, "age_marry": np.nan}
        assert marriage_any_vs_cases(row) == "never_marry_or_preg"

    def test_preg_never_married(self):
        row = {"age_preg": 17, "age_marry": np.nan}
        assert marriage_any_vs_cases(row) == "preg_never_marry"

    def test_married_never_preg(self):
        row = {"age_preg": np.nan, "age_marry": 18}
        assert marriage_any_vs_cases(row) == "marry_never_preg"

    def test_married_before_preg(self):
        row = {"age_preg": 18, "age_marry": 16}
        assert marriage_any_vs_cases(row) == "marry_before_preg"

    def test_married_after_preg(self):
        row = {"age_preg": 16, "age_marry": 18}
        assert marriage_any_vs_cases(row) == "marry_after_preg"

    def test_same_age_is_marry_before(self):
        # am == ap → not (am > ap) → "marry_before_preg"
        row = {"age_preg": 17, "age_marry": 17}
        assert marriage_any_vs_cases(row) == "marry_before_preg"

    def test_string_numbers_are_parsed(self):
        row = {"age_preg": "16", "age_marry": "18"}
        assert marriage_any_vs_cases(row) == "marry_after_preg"

    def test_covers_all_five_categories(self):
        cases = [
            {"age_preg": np.nan, "age_marry": np.nan},
            {"age_preg": 17, "age_marry": np.nan},
            {"age_preg": np.nan, "age_marry": 18},
            {"age_preg": 18, "age_marry": 16},
            {"age_preg": 16, "age_marry": 18},
        ]
        expected = [
            "never_marry_or_preg", "preg_never_marry", "marry_never_preg",
            "marry_before_preg", "marry_after_preg",
        ]
        for row, exp in zip(cases, expected):
            assert marriage_any_vs_cases(row) == exp
