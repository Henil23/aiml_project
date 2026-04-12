# the notebook’s imputation stage included a target-conditioned step (median / group-median / mode) on injected missing values for Age at enrollment, 
# Curricular units 1st sem (grade), and Tuition fees up to date.
#  That exact target-based part cannot be used at prediction time, because the target is unknown for new rows.\
#  So in a deployable predictor.py, that part should be replaced with training-based median/mode imputation.

# What does it do 
# - takes in a CSV file path (from the Flask route) and returns a summary dict with total students, counts per predicted class, and output CSV path
# - the output CSV is saved to the data/ directory and includes the original input columns plus the new prediction/probability columns
# - the predictor handles the same 3 input formats as the notebook (raw Kaggle-style, preprocessed snapshot, or fully encoded), and applies the necessary transformations to get to the final 21 encoded features before prediction
# - it uses the same training-time statistics for imputation and scaling to ensure consistency with the notebook preprocessing
# - it raises informative errors if the input CSV is missing required columns or if the format is not recognized, to guide users in preparing their data correctly
# - the predictor is designed to be called from the Flask route after a user uploads a CSV file, and the returned summary dict is used to display results on the results page and provide a download link for the output CSV with predictions.
import os
import uuid
import joblib
import pandas as pd
import csv

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest_at_risk.pkl")

RAW_REFERENCE_PATH = os.path.join(DATA_DIR, "dataset.csv")
PRE_REFERENCE_PATH = os.path.join(DATA_DIR, "df_preprocessed_before_encoding.csv")

label_map = {
    0: "At-risk",
    1: "Regular",
    2: "Exceptional"
}

# -------------------------------------------------------------------
# Manual-entry dropdown mappings
# Add these near the top so predict_single_student can use them
# -------------------------------------------------------------------

MARITAL_STATUS_OPTIONS = {
    1: "Single",
    2: "Married",
    3: "Widower",
    4: "Divorced",
    5: "Facto union",
    6: "Legally separated",
}

NATIONALITY_OPTIONS = {
    1: "Portuguese",
    2: "German",
    3: "Spanish",
    4: "Italian",
    5: "Dutch",
    6: "English",
    7: "Lithuanian",
    8: "Angolan",
    9: "Cape Verdean",
    10: "Guinean",
    11: "Mozambican",
    12: "Santomean",
    13: "Turkish",
    14: "Brazilian",
    15: "Romanian",
    16: "Moldova (Republic of)",
    17: "Mexican",
    18: "Ukrainian",
    19: "Russian",
    20: "Cuban",
    21: "Colombian",
}

COURSE_OPTIONS = {
    1: "Biofuel Production Technologies",
    2: "Animation and Multimedia Design",
    3: "Social Service (evening attendance)",
    4: "Agronomy",
    5: "Communication Design",
    6: "Veterinary Nursing",
    7: "Informatics Engineering",
    8: "Equinculture",
    9: "Management",
    10: "Social Service",
    11: "Tourism",
    12: "Nursing",
    13: "Oral Hygiene",
    14: "Advertising and Marketing Management",
    15: "Journalism and Communication",
    16: "Basic Education",
    17: "Management (evening attendance)",
}

PARENT_QUALIFICATION_OPTIONS = {
    1: "Secondary Education—12th Year of Schooling or Equivalent",
    2: "Higher Education—bachelor’s degree",
    3: "Higher Education—degree",
    4: "Higher Education—master’s degree",
    5: "Higher Education—doctorate",
    6: "Frequency of Higher Education",
    7: "12th Year of Schooling—not completed",
    8: "11th Year of Schooling—not completed",
    9: "7th Year (Old)",
    10: "Other—11th Year of Schooling",
    11: "2nd year complementary high school course",
    12: "10th Year of Schooling",
    13: "General commerce course",
    14: "Basic Education 3rd Cycle (9th/10th/11th Year) or Equivalent",
    15: "Complementary High School Course",
    16: "Technical-professional course",
    17: "Complementary High School Course—not concluded",
    18: "7th year of schooling",
    19: "2nd cycle of the general high school course",
    20: "9th Year of Schooling—not completed",
    21: "8th year of schooling",
    22: "General Course of Administration and Commerce",
    23: "Supplementary Accounting and Administration",
    24: "Unknown",
    25: "Cannot read or write",
    26: "Can read without having a 4th year of schooling",
    27: "Basic education 1st cycle (4th/5th year) or equivalent",
    28: "Basic Education 2nd Cycle (6th/7th/8th Year) or equivalent",
    29: "Technological specialization course",
    30: "Higher education—degree (1st cycle)",
    31: "Specialized higher studies course",
    32: "Professional higher technical course",
    33: "Higher education—master’s degree (2nd cycle)",
    34: "Higher education—doctorate (3rd cycle)",
}

PARENT_OCCUPATION_OPTIONS = {
    1: "Student",
    2: "Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers",
    3: "Specialists in Intellectual and Scientific Activities",
    4: "Intermediate Level Technicians and Professions",
    5: "Administrative staff",
    6: "Personal Services, Security and Safety Workers, and Sellers",
    7: "Farmers and Skilled Workers in Agriculture, Fisheries, and Forestry",
    8: "Skilled Workers in Industry, Construction, and Craftsmen",
    9: "Installation and Machine Operators and Assembly Workers",
    10: "Unskilled Workers",
    11: "Armed Forces Professions",
    12: "Other Situation",
    13: "(blank)",
    14: "Armed Forces Officers",
    15: "Armed Forces Sergeants",
    16: "Other Armed Forces personnel",
    17: "Directors of administrative and commercial services",
    18: "Hotel, catering, trade, and other services directors",
    19: "Specialists in the physical sciences, mathematics, engineering, and related techniques",
    20: "Health professionals",
    21: "Teachers",
    22: "Specialists in finance, accounting, administrative organization, and public and commercial relations",
    23: "Intermediate level science and engineering technicians and professions",
    24: "Technicians and professionals of intermediate level of health",
    25: "Intermediate level technicians from legal, social, sports, cultural, and similar services",
    26: "Information and communication technology technicians",
    27: "Office workers, secretaries in general, and data processing operators",
    28: "Data, accounting, statistical, financial services, and registry-related operators",
    29: "Other administrative support staff",
    30: "Personal service workers",
    31: "Sellers",
    32: "Personal care workers and the like",
    33: "Protection and security services personnel",
    34: "Market-oriented farmers and skilled agricultural and animal production workers",
    35: "Farmers, livestock keepers, fishermen, hunters and gatherers, and subsistence",
    36: "Skilled construction workers and the like, except electricians",
    37: "Skilled workers in metallurgy, metalworking, and similar",
    38: "Skilled workers in electricity and electronics",
    39: "Workers in food processing, woodworking, and clothing and other industries and crafts",
    40: "Fixed plant and machine operators",
    41: "Assembly workers",
    42: "Vehicle drivers and mobile equipment operators",
    43: "Unskilled workers in agriculture, animal production, and fisheries and forestry",
    44: "Unskilled workers in extractive industry, construction, manufacturing, and transport",
    45: "Meal preparation assistants",
    46: "Street vendors (except food) and street service providers",
}

# final 21 columns expected by the trained model
ENCODED_COLUMNS = [
    "Course",
    "Morning_Attend",
    "Displaced",
    "SpecialNeeds",
    "Debtor",
    "Fees_Paid",
    "Gender",
    "Scholarship",
    "Age",
    "S1_Evaluations",
    "S1_Grade",
    "S2_Evaluations",
    "S2_Grade",
    "Unemployment_Rate",
    "Inflation_Rate",
    "GDP",
    "Parent_Income_Proxy",
    "Has_Spouse_has_spouse",
    "Has_Spouse_single_or_no_spouse",
    "Developed_Nation_developed",
    "Developed_Nation_not_developed"
]

# pre-encoding 19-feature schema used right before one-hot encoding
PRE_COLUMNS = [
    "Course",
    "Morning_Attend",
    "Displaced",
    "SpecialNeeds",
    "Debtor",
    "Fees_Paid",
    "Gender",
    "Scholarship",
    "Age",
    "S1_Evaluations",
    "S1_Grade",
    "S2_Evaluations",
    "S2_Grade",
    "Unemployment_Rate",
    "Inflation_Rate",
    "GDP",
    "Has_Spouse",
    "Developed_Nation",
    "Parent_Income_Proxy"
]

# raw columns needed to reproduce the notebook feature engineering
RAW_REQUIRED_COLUMNS = [
    "Course",
    "Daytime/evening attendance",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "Age at enrollment",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (grade)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
    "Marital status",
    "Nationality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation"
]

# the notebook scaled 10 columns and left binary columns unscaled
SCALED_COLUMNS = [
    "Course",
    "Age",
    "S1_Evaluations",
    "S1_Grade",
    "S2_Evaluations",
    "S2_Grade",
    "Unemployment_Rate",
    "Inflation_Rate",
    "GDP",
    "Parent_Income_Proxy"
]

model = joblib.load(MODEL_PATH)


def _read_csv_like_notebook(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8-sig") as csv_file:
        sample = csv_file.read(4096)
        detected_delimiter = csv.Sniffer().sniff(sample, delimiters=",;").delimiter

    df = pd.read_csv(path, sep=detected_delimiter)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"Nacionality": "Nationality"})
    return df


def _mode(series: pd.Series):
    mode_vals = series.mode(dropna=True)
    if len(mode_vals) == 0:
        return None
    return mode_vals.iloc[0]


def _build_reverse_lookup(options_dict):
    return {str(v).strip().lower(): k for k, v in options_dict.items()}


MARITAL_STATUS_LOOKUP = _build_reverse_lookup(MARITAL_STATUS_OPTIONS)
NATIONALITY_LOOKUP_UI = _build_reverse_lookup(NATIONALITY_OPTIONS)
COURSE_LOOKUP = _build_reverse_lookup(COURSE_OPTIONS)
PARENT_QUALIFICATION_LOOKUP = _build_reverse_lookup(PARENT_QUALIFICATION_OPTIONS)
PARENT_OCCUPATION_LOOKUP = _build_reverse_lookup(PARENT_OCCUPATION_OPTIONS)


def _normalize_pre_reference(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Daytime/evening attendance": "Morning_Attend",
        "Educational special needs": "SpecialNeeds",
        "Tuition fees up to date": "Fees_Paid",
        "Scholarship holder": "Scholarship",
        "Age at enrollment": "Age",
        "Curricular units 1st sem (evaluations)": "S1_Evaluations",
        "Curricular units 1st sem (grade)": "S1_Grade",
        "Curricular units 2nd sem (evaluations)": "S2_Evaluations",
        "Curricular units 2nd sem (grade)": "S2_Grade",
        "Unemployment rate": "Unemployment_Rate",
        "Inflation rate": "Inflation_Rate",
        "Marital status (binary)": "Has_Spouse",
        "Nationality (dev_status)": "Developed_Nation",
        "Parent_avg_income_proxy": "Parent_Income_Proxy",
    }
    df = df.rename(columns=rename_map).copy()

    for col in ["Target", "Target_3level", "Target_3level_num"]:
        if col in df.columns:
            df = df.drop(columns=col)

    missing = [c for c in PRE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "The reference file df_preprocessed_before_encoding.csv is missing columns: "
            f"{missing}"
        )
    return df[PRE_COLUMNS].copy()


RAW_REFERENCE = _read_csv_like_notebook(RAW_REFERENCE_PATH)
PRE_REFERENCE = _normalize_pre_reference(_read_csv_like_notebook(PRE_REFERENCE_PATH))

NUMERIC_IMPUTE_VALUES = {
    "Age": PRE_REFERENCE["Age"].median(),
    "S1_Grade": PRE_REFERENCE["S1_Grade"].median(),
    "Fees_Paid": _mode(PRE_REFERENCE["Fees_Paid"]),
}

SCALE_MEANS = PRE_REFERENCE[SCALED_COLUMNS].mean()
SCALE_STDS = PRE_REFERENCE[SCALED_COLUMNS].std(ddof=0).replace(0, 1)

RAW_REFERENCE = RAW_REFERENCE.reset_index(drop=True)
PRE_REFERENCE = PRE_REFERENCE.reset_index(drop=True)

MARITAL_LOOKUP = (
    pd.concat([RAW_REFERENCE[["Marital status"]], PRE_REFERENCE[["Has_Spouse"]]], axis=1)
    .groupby("Marital status")["Has_Spouse"]
    .agg(_mode)
    .to_dict()
)

NATIONALITY_LOOKUP = (
    pd.concat([RAW_REFERENCE[["Nationality"]], PRE_REFERENCE[["Developed_Nation"]]], axis=1)
    .groupby("Nationality")["Developed_Nation"]
    .agg(_mode)
    .to_dict()
)

_parent_key_ref = RAW_REFERENCE[
    ["Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation"]
].astype(str).agg("||".join, axis=1)

PARENT_PROXY_LOOKUP = (
    pd.DataFrame({
        "parent_key": _parent_key_ref,
        "Parent_Income_Proxy": PRE_REFERENCE["Parent_Income_Proxy"]
    })
    .groupby("parent_key")["Parent_Income_Proxy"]
    .median()
    .to_dict()
)

PARENT_PROXY_DEFAULT = PRE_REFERENCE["Parent_Income_Proxy"].median()


def _missing_columns(df: pd.DataFrame, required_cols: list[str]) -> list[str]:
    return [c for c in required_cols if c not in df.columns]


def _normalize_actual_labels(label_series: pd.Series) -> pd.Series:
    mapping = {
        "0": "At-risk",
        "1": "Regular",
        "2": "Exceptional",
        0: "At-risk",
        1: "Regular",
        2: "Exceptional",
        "At-risk": "At-risk",
        "Regular": "Regular",
        "Exceptional": "Exceptional",
    }
    return label_series.map(mapping).fillna(label_series.astype(str).str.strip())


def _extract_actual_labels(df_input: pd.DataFrame):
    if "Target_3level" in df_input.columns:
        return _normalize_actual_labels(df_input["Target_3level"])
    if "Actual_Class" in df_input.columns:
        return _normalize_actual_labels(df_input["Actual_Class"])
    if "Actual_Label" in df_input.columns:
        return _normalize_actual_labels(df_input["Actual_Label"])
    return None


def _extract_reason_frame(df_input: pd.DataFrame) -> pd.DataFrame | None:
    if all(col in df_input.columns for col in RAW_REQUIRED_COLUMNS):
        df = df_input.rename(columns={
            "Tuition fees up to date": "Fees_Paid",
            "Scholarship holder": "Scholarship",
            "Age at enrollment": "Age",
            "Curricular units 1st sem (evaluations)": "S1_Evaluations",
            "Curricular units 1st sem (grade)": "S1_Grade",
            "Curricular units 2nd sem (evaluations)": "S2_Evaluations",
            "Curricular units 2nd sem (grade)": "S2_Grade",
        }).copy()

        return df[[
            "Fees_Paid",
            "Debtor",
            "Scholarship",
            "Age",
            "S1_Evaluations",
            "S1_Grade",
            "S2_Evaluations",
            "S2_Grade"
        ]].copy()

    try:
        df = _normalize_pre_reference(df_input)
        return df[[
            "Fees_Paid",
            "Debtor",
            "Scholarship",
            "Age",
            "S1_Evaluations",
            "S1_Grade",
            "S2_Evaluations",
            "S2_Grade"
        ]].copy()
    except Exception:
        return None


def _build_reason_summaries(reason_df: pd.DataFrame | None, pred_labels: list[str]) -> list[str]:
    if reason_df is None:
        return [
            "Risk summary unavailable for this input format."
            if label == "At-risk" else ""
            for label in pred_labels
        ]

    s1_grade_q1 = PRE_REFERENCE["S1_Grade"].quantile(0.25)
    s2_grade_q1 = PRE_REFERENCE["S2_Grade"].quantile(0.25)
    s1_eval_q1 = PRE_REFERENCE["S1_Evaluations"].quantile(0.25)
    s2_eval_q1 = PRE_REFERENCE["S2_Evaluations"].quantile(0.25)

    summaries = []

    for i, label in enumerate(pred_labels):
        if label != "At-risk":
            summaries.append("")
            continue

        row = reason_df.iloc[i]
        reasons = []

        if pd.notna(row.get("Fees_Paid")) and float(row["Fees_Paid"]) == 0:
            reasons.append("tuition fees are not up to date")

        if pd.notna(row.get("Debtor")) and float(row["Debtor"]) == 1:
            reasons.append("student is marked as debtor")

        if pd.notna(row.get("S1_Grade")) and float(row["S1_Grade"]) <= s1_grade_q1:
            reasons.append("first-semester grade is low")

        if pd.notna(row.get("S2_Grade")) and float(row["S2_Grade"]) <= s2_grade_q1:
            reasons.append("second-semester grade is low")

        if pd.notna(row.get("S1_Evaluations")) and float(row["S1_Evaluations"]) <= s1_eval_q1:
            reasons.append("first-semester evaluation activity is low")

        if pd.notna(row.get("S2_Evaluations")) and float(row["S2_Evaluations"]) <= s2_eval_q1:
            reasons.append("second-semester evaluation activity is low")

        if len(reasons) == 0:
            reasons.append("combined academic and financial pattern resembles past at-risk students")

        summaries.append("; ".join(reasons[:3]))

    return summaries


def _preprocess_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    missing = _missing_columns(df_raw, RAW_REQUIRED_COLUMNS)
    if missing:
        raise ValueError(
            "Raw CSV is missing required columns needed for prediction: "
            + ", ".join(missing)
        )

    df = df_raw.copy()

    df = df.rename(columns={
        "Daytime/evening attendance": "Morning_Attend",
        "Educational special needs": "SpecialNeeds",
        "Tuition fees up to date": "Fees_Paid",
        "Scholarship holder": "Scholarship",
        "Age at enrollment": "Age",
        "Curricular units 1st sem (evaluations)": "S1_Evaluations",
        "Curricular units 1st sem (grade)": "S1_Grade",
        "Curricular units 2nd sem (evaluations)": "S2_Evaluations",
        "Curricular units 2nd sem (grade)": "S2_Grade",
        "Unemployment rate": "Unemployment_Rate",
        "Inflation rate": "Inflation_Rate",
    })

    df["Has_Spouse"] = df_raw["Marital status"].map(MARITAL_LOOKUP).fillna("single_or_no_spouse")
    df["Developed_Nation"] = df_raw["Nationality"].map(NATIONALITY_LOOKUP).fillna("not_developed")

    parent_key = df_raw[
        ["Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation"]
    ].astype(str).agg("||".join, axis=1)
    df["Parent_Income_Proxy"] = parent_key.map(PARENT_PROXY_LOOKUP).fillna(PARENT_PROXY_DEFAULT)

    df = df[
        [
            "Course",
            "Morning_Attend",
            "Displaced",
            "SpecialNeeds",
            "Debtor",
            "Fees_Paid",
            "Gender",
            "Scholarship",
            "Age",
            "S1_Evaluations",
            "S1_Grade",
            "S2_Evaluations",
            "S2_Grade",
            "Unemployment_Rate",
            "Inflation_Rate",
            "GDP",
            "Has_Spouse",
            "Developed_Nation",
            "Parent_Income_Proxy",
        ]
    ].copy()

    return _preprocess_preencoded_snapshot(df)


def _preprocess_preencoded_snapshot(df_pre: pd.DataFrame) -> pd.DataFrame:
    df_pre = _normalize_pre_reference(df_pre)

    df_pre["Age"] = pd.to_numeric(df_pre["Age"], errors="coerce").fillna(NUMERIC_IMPUTE_VALUES["Age"])
    df_pre["S1_Grade"] = pd.to_numeric(df_pre["S1_Grade"], errors="coerce").fillna(NUMERIC_IMPUTE_VALUES["S1_Grade"])
    df_pre["Fees_Paid"] = pd.to_numeric(df_pre["Fees_Paid"], errors="coerce").fillna(NUMERIC_IMPUTE_VALUES["Fees_Paid"])

    numeric_cols = [
        "Course", "Morning_Attend", "Displaced", "SpecialNeeds", "Debtor", "Fees_Paid",
        "Gender", "Scholarship", "Age", "S1_Evaluations", "S1_Grade", "S2_Evaluations",
        "S2_Grade", "Unemployment_Rate", "Inflation_Rate", "GDP", "Parent_Income_Proxy"
    ]
    for col in numeric_cols:
        df_pre[col] = pd.to_numeric(df_pre[col], errors="coerce")

    for col in numeric_cols:
        if df_pre[col].isna().any():
            fallback = PRE_REFERENCE[col].median()
            df_pre[col] = df_pre[col].fillna(fallback)

    df_pre["Has_Spouse"] = df_pre["Has_Spouse"].fillna("single_or_no_spouse").astype(str)
    df_pre["Developed_Nation"] = df_pre["Developed_Nation"].fillna("not_developed").astype(str)

    encoded = df_pre.copy()
    encoded["Has_Spouse_has_spouse"] = (encoded["Has_Spouse"] == "has_spouse").astype(int)
    encoded["Has_Spouse_single_or_no_spouse"] = (encoded["Has_Spouse"] == "single_or_no_spouse").astype(int)
    encoded["Developed_Nation_developed"] = (encoded["Developed_Nation"] == "developed").astype(int)
    encoded["Developed_Nation_not_developed"] = (encoded["Developed_Nation"] == "not_developed").astype(int)

    encoded = encoded.drop(columns=["Has_Spouse", "Developed_Nation"])

    for col in SCALED_COLUMNS:
        encoded[col] = (encoded[col] - SCALE_MEANS[col]) / SCALE_STDS[col]

    missing_final = _missing_columns(encoded, ENCODED_COLUMNS)
    if missing_final:
        raise ValueError(
            "Preprocessing failed because these final model columns were not created: "
            + ", ".join(missing_final)
        )

    return encoded[ENCODED_COLUMNS].copy()


def _prepare_features(df_input: pd.DataFrame) -> pd.DataFrame:
    if all(col in df_input.columns for col in ENCODED_COLUMNS):
        extra_cols = [c for c in df_input.columns if c not in ENCODED_COLUMNS]
        df = df_input.drop(columns=extra_cols, errors="ignore").copy()
        return df[ENCODED_COLUMNS]

    if all(col in df_input.columns for col in PRE_COLUMNS) or (
        "Daytime/evening attendance" in df_input.columns and "Parent_avg_income_proxy" in df_input.columns
    ):
        return _preprocess_preencoded_snapshot(df_input)

    if all(col in df_input.columns for col in RAW_REQUIRED_COLUMNS):
        return _preprocess_raw(df_input)

    raw_missing = _missing_columns(df_input, RAW_REQUIRED_COLUMNS)
    pre_missing = _missing_columns(df_input, PRE_COLUMNS)
    enc_missing = _missing_columns(df_input, ENCODED_COLUMNS)

    raise ValueError(
        "Input CSV format not recognized.\n"
        f"- Missing for raw Kaggle-style input: {raw_missing}\n"
        f"- Missing for preprocessed 19-column input: {pre_missing}\n"
        f"- Missing for encoded 21-column input: {enc_missing}"
    )


def predict_dataframe(df_input: pd.DataFrame, loaded_model):
    actual_labels = _extract_actual_labels(df_input)
    reason_df = _extract_reason_frame(df_input)

    X_new = _prepare_features(df_input.copy())

    pred_encoded = loaded_model.predict(X_new)
    pred_probs = loaded_model.predict_proba(X_new)

    pred_labels = [label_map[int(p)] for p in pred_encoded]
    reason_summaries = _build_reason_summaries(reason_df, pred_labels)

    result_df = df_input.copy()
    result_df["Predicted_Class"] = pred_labels
    result_df["Prob_At_Risk"] = pred_probs[:, 0]
    result_df["Prob_Regular"] = pred_probs[:, 1]
    result_df["Prob_Exceptional"] = pred_probs[:, 2]
    result_df["Alert_Flag"] = (result_df["Predicted_Class"] == "At-risk").astype(int)
    result_df["Risk_Reason_Summary"] = reason_summaries

    if actual_labels is not None and len(actual_labels) == len(result_df):
        result_df["Actual_Class"] = actual_labels

    return result_df


def predict_csv(input_csv_path: str):
    df_input = _read_csv_like_notebook(input_csv_path)
    result_df = predict_dataframe(df_input, model)

    output_filename = f"predictions_{uuid.uuid4().hex}.csv"
    output_path = os.path.join(DATA_DIR, output_filename)
    result_df.to_csv(output_path, index=False)

    summary = {
        "total_students": int(len(result_df)),
        "at_risk_count": int((result_df["Predicted_Class"] == "At-risk").sum()),
        "regular_count": int((result_df["Predicted_Class"] == "Regular").sum()),
        "exceptional_count": int((result_df["Predicted_Class"] == "Exceptional").sum()),
        "actual_at_risk_available": "Actual_Class" in result_df.columns,
        "correctly_flagged_at_risk": int(
            ((result_df["Predicted_Class"] == "At-risk") & (result_df.get("Actual_Class") == "At-risk")).sum()
        ) if "Actual_Class" in result_df.columns else None,
        "output_path": output_path,
    }

    return summary


def predict_single_student(form_data: dict):
    def _clean(v):
        if v is None:
            return None
        if isinstance(v, str):
            v = v.strip()
            return v if v != "" else None
        return v

    def _map_binary(value, mapping, field_name):
        value = _clean(value)
        if value is None:
            return None

        try:
            num = int(float(value))
            if num in [0, 1]:
                return num
        except Exception:
            pass

        key = str(value).strip().lower()
        if key in mapping:
            return mapping[key]

        raise ValueError(
            f"Invalid value for '{field_name}'. "
            f"Use dropdown values like {list(mapping.keys())} or numeric code 0/1."
        )

    def _map_coded_field(value, reverse_lookup, options_dict, field_name):
        value = _clean(value)
        if value is None:
            return None

        try:
            num = int(float(value))
            if num in options_dict:
                return num
        except Exception:
            pass

        key = str(value).strip().lower()
        if key in reverse_lookup:
            return reverse_lookup[key]

        raise ValueError(
            f"Invalid value for '{field_name}': '{value}'. "
            f"Use one of the dropdown labels or the dataset numeric code."
        )

    yes_no_map = {
        "yes": 1,
        "no": 0,
        "y": 1,
        "n": 0,
        "true": 1,
        "false": 0
    }

    gender_map = {
        "male": 1,
        "female": 0,
        "m": 1,
        "f": 0
    }

    attendance_map = {
        "daytime": 1,
        "evening": 0,
        "day": 1,
        "eve": 0
    }

    mapped_data = {k: _clean(v) for k, v in form_data.items()}

    mapped_data["Daytime/evening attendance"] = _map_binary(
        mapped_data.get("Daytime/evening attendance"),
        attendance_map,
        "Daytime/evening attendance"
    )

    mapped_data["Displaced"] = _map_binary(
        mapped_data.get("Displaced"),
        yes_no_map,
        "Displaced"
    )

    mapped_data["Educational special needs"] = _map_binary(
        mapped_data.get("Educational special needs"),
        yes_no_map,
        "Educational special needs"
    )

    mapped_data["Debtor"] = _map_binary(
        mapped_data.get("Debtor"),
        yes_no_map,
        "Debtor"
    )

    mapped_data["Tuition fees up to date"] = _map_binary(
        mapped_data.get("Tuition fees up to date"),
        yes_no_map,
        "Tuition fees up to date"
    )

    mapped_data["Gender"] = _map_binary(
        mapped_data.get("Gender"),
        gender_map,
        "Gender"
    )

    mapped_data["Scholarship holder"] = _map_binary(
        mapped_data.get("Scholarship holder"),
        yes_no_map,
        "Scholarship holder"
    )

    mapped_data["Course"] = _map_coded_field(
        mapped_data.get("Course"),
        COURSE_LOOKUP,
        COURSE_OPTIONS,
        "Course"
    )

    mapped_data["Marital status"] = _map_coded_field(
        mapped_data.get("Marital status"),
        MARITAL_STATUS_LOOKUP,
        MARITAL_STATUS_OPTIONS,
        "Marital status"
    )

    mapped_data["Nationality"] = _map_coded_field(
        mapped_data.get("Nationality"),
        NATIONALITY_LOOKUP_UI,
        NATIONALITY_OPTIONS,
        "Nationality"
    )

    mapped_data["Mother's qualification"] = _map_coded_field(
        mapped_data.get("Mother's qualification"),
        PARENT_QUALIFICATION_LOOKUP,
        PARENT_QUALIFICATION_OPTIONS,
        "Mother's qualification"
    )

    mapped_data["Father's qualification"] = _map_coded_field(
        mapped_data.get("Father's qualification"),
        PARENT_QUALIFICATION_LOOKUP,
        PARENT_QUALIFICATION_OPTIONS,
        "Father's qualification"
    )

    mapped_data["Mother's occupation"] = _map_coded_field(
        mapped_data.get("Mother's occupation"),
        PARENT_OCCUPATION_LOOKUP,
        PARENT_OCCUPATION_OPTIONS,
        "Mother's occupation"
    )

    mapped_data["Father's occupation"] = _map_coded_field(
        mapped_data.get("Father's occupation"),
        PARENT_OCCUPATION_LOOKUP,
        PARENT_OCCUPATION_OPTIONS,
        "Father's occupation"
    )

    df_input = pd.DataFrame([mapped_data]).copy()
    df_input = df_input.replace("", pd.NA)

    missing = [col for col in RAW_REQUIRED_COLUMNS if col not in df_input.columns]
    if missing:
        raise ValueError(
            "Manual entry is missing required fields: " + ", ".join(missing)
        )

    for col in RAW_REQUIRED_COLUMNS:
        df_input[col] = pd.to_numeric(df_input[col], errors="coerce")

    missing_values = [col for col in RAW_REQUIRED_COLUMNS if df_input[col].isna().iloc[0]]
    if missing_values:
        raise ValueError(
            "Please fill all required manual-entry fields. "
            "Missing/invalid values: " + ", ".join(missing_values)
        )

    reason_df = _extract_reason_frame(df_input)
    X_new = _prepare_features(df_input)

    pred_encoded = model.predict(X_new)
    pred_probs = model.predict_proba(X_new)

    pred_label = label_map[int(pred_encoded[0])]
    reason_summary = _build_reason_summaries(reason_df, [pred_label])[0]

    result = {
        "predicted_class": pred_label,
        "prob_at_risk": float(pred_probs[0][0]),
        "prob_regular": float(pred_probs[0][1]),
        "prob_exceptional": float(pred_probs[0][2]),
        "alert_flag": int(pred_label == "At-risk"),
        "risk_reason_summary": reason_summary,
    }

    return result