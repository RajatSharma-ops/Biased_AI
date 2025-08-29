import os
import uuid
import shutil
from datetime import datetime
from typing import Any, Union

from flask import (
    Flask, request, jsonify, send_from_directory,
    render_template, url_for, redirect, flash
)
from werkzeug.utils import secure_filename
from flask_cors import CORS

# --- Import ML pipeline ---
from aiml.preprocessing import preprocess_dataset
from aiml.model_training import train_models
from aiml.bias_metrices import evaluate_bias
from aiml.visualization import plot_selection_rates
from aiml.report import generate_report

import numpy as np
import pandas as pd
try:
    from fairlearn.metrics import MetricFrame
except ImportError:
    MetricFrame = None

# --- Config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
CHART_DIR = os.path.join(BASE_DIR, "static", "charts")
ALLOWED_EXTENSIONS = {"csv"}

os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "dev-secret"
CORS(app)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _to_native(obj: Any) -> Union[int, float, bool, dict, list, str]:
    """Convert numpy/pandas/Fairlearn objects to pure Python for JSON/templates"""

    # --- Fairlearn MetricFrame ---
    if MetricFrame is not None and isinstance(obj, MetricFrame):
        return obj.by_group.to_dict()

    # --- Pandas ---
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="list")

    # --- Numpy types ---
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()

    # --- Containers ---
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return type(obj)(_to_native(v) for v in obj)

    return obj


def _save_chart_return_path(maybe_path_or_fig, out_path):
    """Ensure charts end up saved in reports/ folder"""
    import matplotlib.figure
    if isinstance(maybe_path_or_fig, str) and os.path.exists(maybe_path_or_fig):
        shutil.move(maybe_path_or_fig, out_path)
        return out_path
    if isinstance(maybe_path_or_fig, matplotlib.figure.Figure):
        maybe_path_or_fig.savefig(out_path, bbox_inches="tight")
        return out_path
    return out_path


# --- Routes ---
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/home")
def home():
    return render_template("home.html")


@app.route("/submit_model")
def submit_model():
    return render_template("submit_model.html")


@app.route("/payment")
def payment():
    return render_template("payment.html")


@app.route("/reports/<path:filename>")
def get_report(filename):
    return send_from_directory(REPORT_DIR, filename, as_attachment=False)

@app.route("/results", methods=["POST"])
def results():
    print(">>> POST /results HIT")

    # --- 1. File validation ---
    if "dataset" not in request.files:
        flash("Missing file", "error")
        return redirect(url_for("submit_model"))

    file = request.files["dataset"]
    if not file or file.filename == "" or not allowed_file(file.filename):
        flash("Invalid file", "error")
        return redirect(url_for("submit_model"))

    # --- 2. Form fields ---
    target_col = request.form.get("target_col", "").strip()
    sensitive_col = request.form.get("sensitive_col", "").strip()
    model_name = request.form.get("model_name", "").strip()
    if not target_col or not sensitive_col or not model_name:
        flash("Please fill all fields", "error")
        return redirect(url_for("submit_model"))

    uid = str(uuid.uuid4())[:8]
    file_path = os.path.join(UPLOAD_DIR, f"{uid}_{secure_filename(file.filename)}")
    file.save(file_path)

    try:
        # --- Preprocess ---
        X_train, X_test, y_train, y_test, A_train, A_test = preprocess_dataset(
            file_path, target_col, sensitive_col
        )

        # --- Train only requested model ---
        model = train_models(X_train, y_train).get(model_name)
        if model is None:
            flash(f"Model '{model_name}' not available", "error")
            return redirect(url_for("submit_model"))

        # --- Evaluate ---
        metrics, group_rates, y_pred = evaluate_bias(model, X_test, y_test, A_test)

        # --- Chart ---
        chart_name = f"chart_{uid}.png"
        chart_path = _save_chart_return_path(
            plot_selection_rates(y_pred, A_test),
            os.path.join(CHART_DIR, chart_name)
        )

        # --- Minimal report (optional, lightweight PDF) ---
        report_name = f"report_{uid}.pdf"
        report_path = os.path.join(REPORT_DIR, report_name)
        final_report = generate_report(
            metrics=_to_native(metrics),
            chart_path=chart_path,
            group_rates=group_rates,
            sensitive_col=sensitive_col,
            chosen_model_name=model_name,
            sensitive_series=A_test,
            output_path=report_path
        )

        # --- Clean up large variables ---
        del X_train, X_test, y_train, y_test, A_train, A_test, y_pred, model

        # --- Render results ---
        return render_template(
            "results.html",
            model_name=model_name,
            metrics=_to_native(metrics),
            group_rates=_to_native(group_rates),
            chart_url=url_for("static", filename=f"charts/{chart_name}"),
            report_url=url_for("get_report", filename=os.path.basename(final_report))
        )

    except Exception as e:
        app.logger.error("Error in /results: %s", str(e))
        flash(f"Error: {e}", "error")
        return redirect(url_for("submit_model"))

    finally:
        try:
            os.remove(file_path)
        except Exception:
            pass


@app.route("/run-bias", methods=["POST"])
def run_bias():
    if "dataset" not in request.files:
        return jsonify({"ok": False, "error": "Missing dataset"}), 400

    file = request.files["dataset"]
    if not file or file.filename == "" or not allowed_file(file.filename):
        return jsonify({"ok": False, "error": "Invalid file"}), 400

    target_col = request.form.get("target_col", "").strip()
    sensitive_col = request.form.get("sensitive_col", "").strip()
    model_name = request.form.get("model_name", "").strip()
    if not target_col or not sensitive_col or not model_name:
        return jsonify({"ok": False, "error": "Missing fields"}), 400

    uid = str(uuid.uuid4())[:8]
    file_path = os.path.join(UPLOAD_DIR, f"{uid}_{secure_filename(file.filename)}")
    file.save(file_path)

    try:
        X_train, X_test, y_train, y_test, A_train, A_test = preprocess_dataset(
            file_path, target_col, sensitive_col
        )
        models = train_models(X_train, y_train)
        if model_name not in models:
            return jsonify({"ok": False, "error": f"Model '{model_name}' not available"}), 400

        model = models[model_name]
        metrics, group_rates, y_pred = evaluate_bias(model, X_test, y_test, A_test)

        chart_name = f"chart_{uid}.png"
        chart_path = _save_chart_return_path(
            plot_selection_rates(y_pred, A_test),
            os.path.join(CHART_DIR, chart_name)
        )

        report_name = f"report_{uid}.pdf"
        report_path = os.path.join(REPORT_DIR, report_name)
        final_report = generate_report(
            metrics=_to_native(metrics),
            chart_path=chart_path,
            group_rates=group_rates,
            sensitive_col=sensitive_col,
            chosen_model_name=model_name,
            sensitive_series=A_test,
            output_path=report_path
        )
        if not final_report or not os.path.exists(final_report):
            final_report = chart_path

        return jsonify({
            "ok": True,
            "model_name": model_name,
            "metrics": _to_native(metrics),
            "group_rates": _to_native(group_rates),
            "report_url": f"/reports/{os.path.basename(final_report)}",
            "charts": [f"/reports/{os.path.basename(chart_path)}"]
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

    finally:
        try:
            os.remove(file_path)
        except Exception:
            pass


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
