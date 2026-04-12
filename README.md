# Student's Holistic Performance Classifier

**Course:** Applications of Artificial Intelligence and Machine Learning (PROG74000)  
**Group:** 11: Gitesh, Henil, Jerin, Kahan

---

## Overview

This project builds and deploys a machine learning classifier that predicts a student's holistic academic standing as one of three labels:

| Label | Meaning |
|---|---|
| **At-Risk** | Student dropped out before completing their degree |
| **Regular** | Non-dropout student performing at or below the 75th percentile composite score |
| **Exceptional** | Non-dropout student in the top quartile of a composite academic + financial score |

Rather than relying on raw grades alone, the classifier accounts for socioeconomic background, course load, semester progression, and macroeconomic conditions at the time of enrolment, producing a more contextually meaningful label than a simple pass/fail outcome.

The model is trained on the **Higher Education Predictors of Student Retention** dataset (Realinho et al., *Data* 2022, 7, 146 | 4,424 students, 35 original features) and is deployed as a live REST API and browser interface.

---

## Live Deployment

The application is deployed on Render and accessible without any local setup:

**Base URL:** `https://student-risk-predictor-inuu.onrender.com`

| Interface | URL |
|---|---|
| Browser UI | `https://student-risk-predictor-inuu.onrender.com/` |
| Health check | `https://student-risk-predictor-inuu.onrender.com/health` |
| Prediction API | `https://student-risk-predictor-inuu.onrender.com/predict` |

See [API Usage](#api-usage) below for request/response details.

---

## Project Structure

```
project/
├── app/
│   ├── __init__.py          # Flask app factory
│   ├── routes.py            # API endpoints (/health, /predict, /)
│   ├── predictor.py         # Model loading and inference logic
│   ├── templates/           # Browser UI (HTML)
│   └── static/              # CSS / JS assets
├── models/                  # Serialised trained model(s)
├── project.ipynb            # Full preprocessing + EDA + training notebook
├── run.py                   # Flask entry point
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Preprocessing & Feature Engineering

All data preparation is documented and executable in `project.ipynb`. Key steps:

**Feature reduction**: 14 redundant or highly collinear columns removed (credited units, enrolled units, approved units, without-evaluation counts, application mode/order, International flag, parental qualifications, previous qualification).

**Feature engineering:**
- `Marital status` -> binary `Has_Spouse` (married / facto union vs. all others)
- `Nationality` -> binary `Developed_Nation` (EU/developed vs. other)
- `Mother's occupation` + `Father's occupation` -> ordinal `Parent_Income_Proxy` (1 low / 2 medium / 3 high), mapped from descriptor Appendix A occupation codes
- Semester 1 and Semester 2 features kept **separate** (`S1_Grade`, `S2_Grade`, `S1_Evaluations`, `S2_Evaluations`) to allow models to weight early vs. late performance independently

**Course toggle**: two flags control field-of-study handling:
```python
REMOVE_COURSE   = False   # True -> drop Course entirely
COURSE_GROUPING = True    # True -> 5 domain groups | False -> 17 individual courses
```

**Label redesign**: original administrative labels (`Dropout / Enrolled / Graduate`) replaced by holistic performance labels using a composite score:
```
Score = 0.5 × mean_semester_grade + 0.3 × S2_approved_units + 0.2 × (tuition_up_to_date × 10)
```
Exceptional threshold = 75th percentile of the non-dropout pool.

**Imputation**: 120 MCAR values injected across three targeted columns and recovered with justifiable strategies (global median for age, group-median-by-label for S1 grade, mode for tuition flag).

**Encoding**: categorical features one-hot encoded; continuous features z-scored (StandardScaler); binary flags left unscaled.

---

## Models

Three classifiers are trained and compared:

| Model | Role |
|---|---|
| **ANN** (Artificial Neural Network) | Primary model; captures non-linear relationships across all features |
| **Random Forest** | Interpretable ensemble; provides feature importance rankings |
| **KNN** | Naive baseline; used to validate that ANN/RF add meaningful signal over a simple distance-based rule |

The deployed endpoint uses the Random Forest model (`random_forest_student_risk`), confirmed by the `/health` response.

---

## API Usage

### 1. Health Check

Verify the service is running before sending predictions.

```bash
curl https://student-risk-predictor-inuu.onrender.com/health
```

**Response:**
```json
{
  "status": "ok",
  "model": "random_forest_student_risk"
}
```

---

### 2. Prediction: CSV Upload

Send a CSV file as `multipart/form-data` using field name `file`.

```bash
curl -X POST https://student-risk-predictor-inuu.onrender.com/predict \
  -F "file=@data/dataset.csv"
```

**Response:**
```json
{
  "total_students": 4424,
  "at_risk_count": 1234,
  "regular_count": 2800,
  "exceptional_count": 390
}
```

---

### 3. Prediction: JSON Body

Send a JSON body with a top-level `students` key containing a list of student records.

```bash
curl -X POST https://student-risk-predictor-inuu.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "students": [
      {
        "Course": 171,
        "Daytime/evening attendance": 1,
        "Displaced": 1,
        "Educational special needs": 0,
        "Debtor": 0,
        "Tuition fees up to date": 1,
        "Gender": 1,
        "Scholarship holder": 0,
        "Age at enrollment": 20,
        "Curricular units 1st sem (evaluations)": 6,
        "Curricular units 1st sem (grade)": 12,
        "Curricular units 2nd sem (evaluations)": 6,
        "Curricular units 2nd sem (grade)": 11,
        "Unemployment rate": 10.8,
        "Inflation rate": 1.4,
        "GDP": 1.74,
        "Marital status": 1,
        "Nationality": 1,
        "Mother'\''s qualification": 13,
        "Father'\''s qualification": 27,
        "Mother'\''s occupation": 10,
        "Father'\''s occupation": 7
      }
    ]
  }'
```

**Response:**
```json
{
  "total_students": 1,
  "at_risk_count": 0,
  "regular_count": 1,
  "exceptional_count": 0,
  "predictions": [
    {
      "predicted_class": "Regular",
      "prob_at_risk": 0.12,
      "prob_regular": 0.80,
      "prob_exceptional": 0.08,
      "alert_flag": 0,
      "risk_reason_summary": ""
    }
  ]
}
```

The per-student prediction object includes the predicted class, probability for each class, an `alert_flag` (1 if At-Risk is predicted), and a `risk_reason_summary` string when risk factors are detected.

---

### 4. Browser Interface

Open `https://student-risk-predictor-inuu.onrender.com/` in any browser to:
- Upload a CSV file and view aggregate predictions
- Enter a single student's details manually via a form
- View results rendered in-page without writing any code

---

## Local Setup

### Prerequisites

- [Anaconda](https://www.anaconda.com/) (we used `conda 25.5.1`)

### 1. Create the environment

```cmd
conda create --prefix ./aiml_project python=3.11 ipykernel
conda activate ./aiml_project
```

> **Tip:** shorten the shell prompt prefix:
> ```cmd
> conda config --set env_prompt (aiml_project)
> ```

### 2. Install dependencies

```cmd
pip install -r requirements.txt
```

### 3. Run the Flask app locally

```cmd
python run.py
```

The app starts on `http://127.0.0.1:5000` by default (debug mode enabled).

### 4. Run the notebook

Launch Jupyter from within the activated environment:

```cmd
jupyter notebook project.ipynb
```

The notebook covers the full pipeline: data loading -> feature engineering -> EDA -> label redesign -> imputation -> encoding -> model training.

---

## Dataset

**Source:** Realinho, V.; Machado, J.; Baptista, L.; Martins, M.V. *Predicting Student Dropout and Academic Success*. Data 2022, 7, 146. https://doi.org/10.3390/data7110146

- 4,424 student records from the Polytechnic Institute of Portalegre (Portugal)
- 17 undergraduate programmes, academic years 2008/2009 – 2018/2019
- 35 original features: demographic, socioeconomic, macroeconomic, and academic
- No missing values in the raw file; fully anonymised under GDPR
- License: CC BY 4.0