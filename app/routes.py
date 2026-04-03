import os
import uuid

from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_file,
)

from app.predictor import predict_csv, predict_single_student

main = Blueprint("main", __name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)


@main.route("/", methods=["GET", "POST"])
def index():
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
                )

            except Exception as e:
                flash(f"Manual prediction failed: {str(e)}")
                return redirect(url_for("main.index"))

    return render_template("index.html")


@main.route("/download/<filename>")
def download_file(filename):
    file_path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(file_path):
        flash("Requested prediction file was not found.")
        return redirect(url_for("main.index"))

    return send_file(file_path, as_attachment=True)