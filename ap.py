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

    # --- 3. Save uploaded CSV ---
    uid = str(uuid.uuid4())[:8]
    file_path = os.path.join(UPLOAD_DIR, f"{uid}_{secure_filename(file.filename)}")
    file.save(file_path)

    try:
        # --- 4. Preprocess + train ---
        X_train, X_test, y_train, y_test, A_train, A_test = preprocess_dataset(
            file_path, target_col, sensitive_col
        )
        models = train_models(X_train, y_train)
        if model_name not in models:
            flash(f"Model '{model_name}' not available", "error")
            return redirect(url_for("submit_model"))

        model = models[model_name]
        metrics, group_rates, y_pred = evaluate_bias(model, X_test, y_test, A_test)

        # --- 5. Chart ---
        chart_name = f"chart_{uid}.png"
        chart_path = _save_chart_return_path(
            plot_selection_rates(y_pred, A_test),
            os.path.join(CHART_DIR, chart_name)
        )

        # --- 6. Report ---
        report_name = f"report_{uid}.pdf"
        report_path = os.path.join(REPORT_DIR, report_name)
        final_report = generate_report(
            metrics=_to_native(metrics),
            chart_path=chart_path,
            group_rates=group_rates,        # pass raw object to report
            sensitive_col=sensitive_col,
            chosen_model_name=model_name,
            sensitive_series=A_test,
            output_path=report_path
        )
        if not final_report or not os.path.exists(final_report):
            final_report = report_path

        print(">>> Report saved at:", final_report)

        # --- 7. Render results page ---
        return render_template(
            "results.html",
            model_name=model_name,
            metrics=_to_native(metrics),
            group_rates=_to_native(group_rates),   # dict for Jinja
            chart_url = url_for("static", filename=f"charts/{chart_name}"),
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
