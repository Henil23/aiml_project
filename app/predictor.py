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
# helper - functions to read CSVs, preprocess the same way as the notebook, and prepare features for prediction
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

    # drop target-ish columns if present
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

# training-time stats for deployable imputation / scaling
NUMERIC_IMPUTE_VALUES = {
    "Age": PRE_REFERENCE["Age"].median(),
    "S1_Grade": PRE_REFERENCE["S1_Grade"].median(),
    "Fees_Paid": _mode(PRE_REFERENCE["Fees_Paid"]),
}

SCALE_MEANS = PRE_REFERENCE[SCALED_COLUMNS].mean()
SCALE_STDS = PRE_REFERENCE[SCALED_COLUMNS].std(ddof=0).replace(0, 1)

# exact lookup tables learned from the aligned training files
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


def _preprocess_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    missing = _missing_columns(df_raw, RAW_REQUIRED_COLUMNS)
    if missing:
        raise ValueError(
            "Raw CSV is missing required columns needed for prediction: "
            + ", ".join(missing)
        )

    df = df_raw.copy()

    # rename raw -> pre-encoding features
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

    # derive notebook-engineered categorical features from training reference mappings
    df["Has_Spouse"] = df_raw["Marital status"].map(MARITAL_LOOKUP).fillna("single_or_no_spouse")
    df["Developed_Nation"] = df_raw["Nationality"].map(NATIONALITY_LOOKUP).fillna("not_developed")

    parent_key = df_raw[
        ["Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation"]
    ].astype(str).agg("||".join, axis=1)
    df["Parent_Income_Proxy"] = parent_key.map(PARENT_PROXY_LOOKUP).fillna(PARENT_PROXY_DEFAULT)

    # keep only the model-relevant pre-encoding features
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
    # allow either exact 19-column schema or the human-readable snapshot names
    df_pre = _normalize_pre_reference(df_pre)

    # deployable imputation using training statistics
    df_pre["Age"] = pd.to_numeric(df_pre["Age"], errors="coerce").fillna(NUMERIC_IMPUTE_VALUES["Age"])
    df_pre["S1_Grade"] = pd.to_numeric(df_pre["S1_Grade"], errors="coerce").fillna(NUMERIC_IMPUTE_VALUES["S1_Grade"])
    df_pre["Fees_Paid"] = pd.to_numeric(df_pre["Fees_Paid"], errors="coerce").fillna(NUMERIC_IMPUTE_VALUES["Fees_Paid"])

    # make sure the rest are numeric where expected
    numeric_cols = [
        "Course", "Morning_Attend", "Displaced", "SpecialNeeds", "Debtor", "Fees_Paid",
        "Gender", "Scholarship", "Age", "S1_Evaluations", "S1_Grade", "S2_Evaluations",
        "S2_Grade", "Unemployment_Rate", "Inflation_Rate", "GDP", "Parent_Income_Proxy"
    ]
    for col in numeric_cols:
        df_pre[col] = pd.to_numeric(df_pre[col], errors="coerce")

    # generic fallback imputation for any remaining numeric NaNs
    for col in numeric_cols:
        if df_pre[col].isna().any():
            fallback = PRE_REFERENCE[col].median()
            df_pre[col] = df_pre[col].fillna(fallback)

    # categorical fallback
    df_pre["Has_Spouse"] = df_pre["Has_Spouse"].fillna("single_or_no_spouse").astype(str)
    df_pre["Developed_Nation"] = df_pre["Developed_Nation"].fillna("not_developed").astype(str)

    # one-hot encode exactly as the notebook did
    encoded = df_pre.copy()
    encoded["Has_Spouse_has_spouse"] = (encoded["Has_Spouse"] == "has_spouse").astype(int)
    encoded["Has_Spouse_single_or_no_spouse"] = (encoded["Has_Spouse"] == "single_or_no_spouse").astype(int)
    encoded["Developed_Nation_developed"] = (encoded["Developed_Nation"] == "developed").astype(int)
    encoded["Developed_Nation_not_developed"] = (encoded["Developed_Nation"] == "not_developed").astype(int)

    encoded = encoded.drop(columns=["Has_Spouse", "Developed_Nation"])

    # standardize the same 10 columns using training reference stats
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
    # case 1: already encoded / model-ready
    if all(col in df_input.columns for col in ENCODED_COLUMNS):
        extra_cols = [c for c in df_input.columns if c not in ENCODED_COLUMNS]
        df = df_input.drop(columns=extra_cols, errors="ignore").copy()
        return df[ENCODED_COLUMNS]

    # case 2: pre-encoding snapshot
    if all(col in df_input.columns for col in PRE_COLUMNS) or (
        "Daytime/evening attendance" in df_input.columns and "Parent_avg_income_proxy" in df_input.columns
    ):
        return _preprocess_preencoded_snapshot(df_input)

    # case 3: raw Kaggle-style dataset
    if all(col in df_input.columns for col in RAW_REQUIRED_COLUMNS):
        return _preprocess_raw(df_input)

    # otherwise explain what is missing
    raw_missing = _missing_columns(df_input, RAW_REQUIRED_COLUMNS)
    pre_missing = _missing_columns(df_input, PRE_COLUMNS)
    enc_missing = _missing_columns(df_input, ENCODED_COLUMNS)

    raise ValueError(
        "Input CSV format not recognized.\n"
        f"- Missing for raw Kaggle-style input: {raw_missing}\n"
        f"- Missing for preprocessed 19-column input: {pre_missing}\n"
        f"- Missing for encoded 21-column input: {enc_missing}"
    )


def predict_csv(input_csv_path: str):
    df_input = _read_csv_like_notebook(input_csv_path)

    actual_labels = _extract_actual_labels(df_input)
    reason_df = _extract_reason_frame(df_input)

    X_new = _prepare_features(df_input)

    pred_encoded = model.predict(X_new)
    pred_probs = model.predict_proba(X_new)

    pred_labels = [label_map[p] for p in pred_encoded]
    reason_summaries = _build_reason_summaries(reason_df, pred_labels)

    result_df = df_input.copy()
    result_df["Predicted_Class"] = pred_labels
    result_df["Prob_At_Risk"] = pred_probs[:, 0]
    result_df["Prob_Regular"] = pred_probs[:, 1]
    result_df["Prob_Exceptional"] = pred_probs[:, 2]
    result_df["Alert_Flag"] = (result_df["Predicted_Class"] == "At-risk").astype(int)
    result_df["Risk_Reason_Summary"] = reason_summaries

    predicted_at_risk_count = int((result_df["Predicted_Class"] == "At-risk").sum())

    actual_at_risk_available = False
    correctly_flagged_at_risk = None

    if actual_labels is not None and len(actual_labels) == len(result_df):
        actual_at_risk_available = True
        correctly_flagged_at_risk = int(
            ((result_df["Predicted_Class"] == "At-risk") & (actual_labels == "At-risk")).sum()
        )
        result_df["Actual_Class"] = actual_labels

    output_filename = f"predictions_{uuid.uuid4().hex}.csv"
    output_path = os.path.join(DATA_DIR, output_filename)
    result_df.to_csv(output_path, index=False)

    summary = {
        "total_students": int(len(result_df)),
        "at_risk_count": predicted_at_risk_count,
        "regular_count": int((result_df["Predicted_Class"] == "Regular").sum()),
        "exceptional_count": int((result_df["Predicted_Class"] == "Exceptional").sum()),
        "actual_at_risk_available": actual_at_risk_available,
        "correctly_flagged_at_risk": correctly_flagged_at_risk,
        "output_path": output_path,
    }

    return summary
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
    # raw Kaggle-style input
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

    # preprocessed-before-encoding input
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
def predict_single_student(form_data: dict):
    """
    Manual-entry prediction:
    - accepts user-friendly values for simple categorical fields
    - keeps complex coded fields numeric
    - converts everything into the raw Kaggle-style schema expected by preprocessing
    """

    def _clean(v):
        if v is None:
            return None
        if isinstance(v, str):
            v = v.strip()
            return v if v != "" else None
        return v

    def _map_value(value, mapping, field_name):
        value = _clean(value)
        if value is None:
            return None

        # already numeric-like -> keep as-is for later numeric conversion
        try:
            float(value)
            return value
        except Exception:
            pass

        key = str(value).strip().lower()
        if key in mapping:
            return mapping[key]

        raise ValueError(
            f"Invalid value for '{field_name}': '{value}'. "
            f"Allowed values: {', '.join(mapping.keys())} or the dataset numeric code."
        )

    # friendly mappings for fields that are easy to humanize
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

    # copy and normalize
    mapped_data = {k: _clean(v) for k, v in form_data.items()}

    # convert human-friendly values for simple binary fields
    mapped_data["Daytime/evening attendance"] = _map_value(
        mapped_data.get("Daytime/evening attendance"),
        attendance_map,
        "Daytime/evening attendance"
    )

    mapped_data["Displaced"] = _map_value(
        mapped_data.get("Displaced"),
        yes_no_map,
        "Displaced"
    )

    mapped_data["Educational special needs"] = _map_value(
        mapped_data.get("Educational special needs"),
        yes_no_map,
        "Educational special needs"
    )

    mapped_data["Debtor"] = _map_value(
        mapped_data.get("Debtor"),
        yes_no_map,
        "Debtor"
    )

    mapped_data["Tuition fees up to date"] = _map_value(
        mapped_data.get("Tuition fees up to date"),
        yes_no_map,
        "Tuition fees up to date"
    )

    mapped_data["Gender"] = _map_value(
        mapped_data.get("Gender"),
        gender_map,
        "Gender"
    )

    mapped_data["Scholarship holder"] = _map_value(
        mapped_data.get("Scholarship holder"),
        yes_no_map,
        "Scholarship holder"
    )

    # build one-row dataframe from mapped form input
    df_input = pd.DataFrame([mapped_data]).copy()

    # convert blank strings to missing
    df_input = df_input.replace("", pd.NA)

    # confirm all required raw columns exist
    missing = [col for col in RAW_REQUIRED_COLUMNS if col not in df_input.columns]
    if missing:
        raise ValueError(
            "Manual entry is missing required fields: " + ", ".join(missing)
        )

    # convert every required field to numeric
    # some fields are still expected as dataset numeric codes:
    # Course, Marital status, Nationality, parents' qualification/occupation, etc.
    for col in RAW_REQUIRED_COLUMNS:
        df_input[col] = pd.to_numeric(df_input[col], errors="coerce")

    # validate missing/invalid numeric-coded fields
    missing_values = [col for col in RAW_REQUIRED_COLUMNS if df_input[col].isna().iloc[0]]
    if missing_values:
        raise ValueError(
            "Please fill all required manual-entry fields. "
            "Some fields still require dataset numeric codes. "
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