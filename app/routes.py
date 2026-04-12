import os
import uuid
import tempfile
import pandas as pd

from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_file,
    jsonify,
)

from app.predictor import (
    predict_csv,
    predict_single_student,
    COURSE_OPTIONS,
    MARITAL_STATUS_OPTIONS,
    NATIONALITY_OPTIONS,
    PARENT_QUALIFICATION_OPTIONS,
    PARENT_OCCUPATION_OPTIONS,
)

main = Blueprint("main", __name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)


def _dropdown_options():
    return {
        "course_options": COURSE_OPTIONS,
        "marital_status_options": MARITAL_STATUS_OPTIONS,
        "nationality_options": NATIONALITY_OPTIONS,
        "mother_qualification_options": PARENT_QUALIFICATION_OPTIONS,
        "father_qualification_options": PARENT_QUALIFICATION_OPTIONS,
        "mother_occupation_options": PARENT_OCCUPATION_OPTIONS,
        "father_occupation_options": PARENT_OCCUPATION_OPTIONS,
    }


# ─────────────────────────────────────────────
# Health check
# GET /health
# ─────────────────────────────────────────────
@main.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "random_forest_student_risk"})


# ─────────────────────────────────────────────
# API prediction endpoint
# POST /predict
#
# Supports:
#   A) JSON body: {"students": [{...raw row...}, ...]}
#   B) CSV upload: multipart/form-data with field name "file"
# ─────────────────────────────────────────────
@main.route("/predict", methods=["POST"])
def predict_api():
    content_type = request.content_type or ""

    # ── Format B: CSV file upload ──
    if "multipart" in content_type:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file uploaded. Use field name 'file'."}), 400
        if not file.filename.lower().endswith(".csv"):
            return jsonify({"error": "Only CSV files are accepted."}), 400

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        try:
            summary = predict_csv(tmp_path)
            summary.pop("output_path", None)  # do not expose server path
            return jsonify(summary)
        except ValueError as e:
            return jsonify({"error": str(e)}), 422
        except Exception as e:
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # ── Format A: JSON body ──
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Send a JSON body with a 'students' key, or upload a CSV file."}), 400

    students = data.get("students")
    if not students or not isinstance(students, list):
        return jsonify({"error": "'students' must be a non-empty list of student objects."}), 400

    try:
        results = []
        for student in students:
            result = predict_single_student(student)
            results.append(result)

        at_risk = sum(1 for r in results if r["predicted_class"] == "At-risk")
        regular = sum(1 for r in results if r["predicted_class"] == "Regular")
        exceptional = sum(1 for r in results if r["predicted_class"] == "Exceptional")

        return jsonify({
            "total_students": len(results),
            "at_risk_count": at_risk,
            "regular_count": regular,
            "exceptional_count": exceptional,
            "predictions": results,
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# ─────────────────────────────────────────────
# Main web UI
# ─────────────────────────────────────────────
@main.route("/", methods=["GET", "POST"])
def index():
    dropdown_options = _dropdown_options()

    if request.method == "POST":
        action = request.form.get("action")

        # ---------------- CSV upload flow ----------------
        if action == "upload_csv":
            if "file" not in request.files:
                flash("No file part found.")
                return redirect(url_for("main.index"))

            file = request.files["file"]

            if file.filename == "":
                flash("Please choose a CSV file.")
                return redirect(url_for("main.index"))

            if not file.filename.lower().endswith(".csv"):
                flash("Only CSV files are allowed.")
                return redirect(url_for("main.index"))

            upload_name = f"{uuid.uuid4().hex}_{file.filename}"
            upload_path = os.path.join(UPLOAD_DIR, upload_name)

            try:
                file.save(upload_path)
                summary = predict_csv(upload_path)
                output_filename = os.path.basename(summary["output_path"])

                results_df = pd.read_csv(summary["output_path"])

                at_risk_rows = results_df[
                    results_df["Predicted_Class"] == "At-risk"
                ].to_dict(orient="records")

                prediction_rows = results_df.to_dict(orient="records")

                return render_template(
                    "results.html",
                    total_students=summary["total_students"],
                    at_risk_count=summary["at_risk_count"],
                    regular_count=summary["regular_count"],
                    exceptional_count=summary["exceptional_count"],
                    actual_at_risk_available=summary.get("actual_at_risk_available", False),
                    correctly_flagged_at_risk=summary.get("correctly_flagged_at_risk"),
                    output_filename=output_filename,
                    single_result=None,
                    at_risk_rows=at_risk_rows,
                    prediction_rows=prediction_rows,
                )

            except Exception as e:
                flash(f"Prediction failed: {str(e)}")
                return redirect(url_for("main.index"))

        # ---------------- Manual single-student flow ----------------
        elif action == "manual_entry":
            try:
                form_data = {
                    "Course": request.form.get("Course"),
                    "Daytime/evening attendance": request.form.get("Daytime/evening attendance"),
                    "Displaced": request.form.get("Displaced"),
                    "Educational special needs": request.form.get("Educational special needs"),
                    "Debtor": request.form.get("Debtor"),
                    "Tuition fees up to date": request.form.get("Tuition fees up to date"),
                    "Gender": request.form.get("Gender"),
                    "Scholarship holder": request.form.get("Scholarship holder"),
                    "Age at enrollment": request.form.get("Age at enrollment"),
                    "Curricular units 1st sem (evaluations)": request.form.get("Curricular units 1st sem (evaluations)"),
                    "Curricular units 1st sem (grade)": request.form.get("Curricular units 1st sem (grade)"),
                    "Curricular units 2nd sem (evaluations)": request.form.get("Curricular units 2nd sem (evaluations)"),
                    "Curricular units 2nd sem (grade)": request.form.get("Curricular units 2nd sem (grade)"),
                    "Unemployment rate": request.form.get("Unemployment rate"),
                    "Inflation rate": request.form.get("Inflation rate"),
                    "GDP": request.form.get("GDP"),
                    "Marital status": request.form.get("Marital status"),
                    "Nationality": request.form.get("Nationality"),
                    "Mother's qualification": request.form.get("Mother's qualification"),
                    "Father's qualification": request.form.get("Father's qualification"),
                    "Mother's occupation": request.form.get("Mother's occupation"),
                    "Father's occupation": request.form.get("Father's occupation"),
                }

                single_result = predict_single_student(form_data)

                return render_template(
                    "results.html",
                    total_students=1,
                    at_risk_count=1 if single_result["predicted_class"] == "At-risk" else 0,
                    regular_count=1 if single_result["predicted_class"] == "Regular" else 0,
                    exceptional_count=1 if single_result["predicted_class"] == "Exceptional" else 0,
                    actual_at_risk_available=False,
                    correctly_flagged_at_risk=None,
                    output_filename=None,
                    single_result=single_result,
                    at_risk_rows=None,
                    prediction_rows=None,
                )

            except Exception as e:
                flash(f"Manual prediction failed: {str(e)}")
                return redirect(url_for("main.index"))

    return render_template("index.html", **dropdown_options)


@main.route("/download/<filename>")
def download_file(filename):
    file_path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(file_path):
        flash("Requested prediction file was not found.")
        return redirect(url_for("main.index"))

    return send_file(file_path, as_attachment=True)