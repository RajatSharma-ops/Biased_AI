from fpdf import FPDF
import numpy as np
import pandas as pd
def generate_report(metrics, chart_path, group_rates, sensitive_col, chosen_model_name,
                    output_path="reports/report.pdf", sensitive_mapping=None, sensitive_series=None):
    """
    Generate a detailed PDF fairness report with model metrics, group rates, and interpretation.
    Automatically uses actual sensitive column values for group labels.
    """
    pdf = FPDF()
    pdf.add_page()

    # Display group mapping if available
    if sensitive_mapping:
        mapping_items = [f"{str(k)} → {v}" for k, v in sensitive_mapping.items()]
        line_length = 80  # max characters per line
        lines = []
        current_line = ""
        for item in mapping_items:
            if len(current_line) + len(item) + 2 <= line_length:
                current_line += (item + ", ")
            else:
                lines.append(current_line.rstrip(", "))
                current_line = item + ", "
        if current_line:
            lines.append(current_line.rstrip(", "))

        pdf.set_font("Arial", size=10)
        pdf.cell(200, 8, txt="Group Mapping:", ln=True)
        for line in lines:
            pdf.multi_cell(0, 8, line)
        pdf.ln(3)

    # Title
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="BiasGuard - AI Fairness Report", ln=True, align="C")
    pdf.ln(10)

    # Model Overview
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Model Overview", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 8, f"This report analyzes bias in the {chosen_model_name} with respect to sensitive attribute: {sensitive_col}.")
    pdf.ln(5)

    # Fairness Metrics
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Fairness Metrics", ln=True)
    pdf.set_font("Arial", size=10)
    for key, val in metrics.items():
        pdf.cell(200, 8, txt=f"{key}: {val:.4f}", ln=True)
    pdf.ln(5)

    # Handle group_rates if it is a MetricFrame or dict-like
    if hasattr(group_rates, "by_group"):
        group_rates_dict = group_rates.by_group.to_dict()
    elif hasattr(group_rates, "to_dict"):
        group_rates_dict = group_rates.to_dict()
    elif hasattr(group_rates, "to_frame"):
        group_rates_dict = dict(group_rates.to_frame().iloc[:, 0])
    else:
        group_rates_dict = dict(group_rates)

    groups_list =[]
    rates_list = []

    # Write table header
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Selection Rates by {sensitive_col}", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.cell(95, 8, f"{sensitive_col} Group", border=1)
    pdf.cell(95, 8, "Selection Rate", border=1, ln=True)

    # Create a mapping from group_rates_dict keys to actual sensitive values if sensitive_series is provided
    group_key_to_label = {}
    if sensitive_series is not None:
        # Get unique values in sensitive_series in order of appearance
        unique_vals = list(pd.unique(sensitive_series))
        # Map each group key to the corresponding unique sensitive value if possible
        for key in group_rates_dict.keys():
            # Try to interpret key as an index to unique_vals if key is numeric
            if isinstance(key, (int, np.integer)):
                if 0 <= key < len(unique_vals):
                    group_key_to_label[key] = str(unique_vals[key])
                else:
                    group_key_to_label[key] = str(key)
            elif isinstance(key, float) and not np.isnan(key):
                int_key = int(key)
                if 0 <= int_key < len(unique_vals):
                    group_key_to_label[key] = str(unique_vals[int_key])
                else:
                    group_key_to_label[key] = str(key)
            else:
                # Non-numeric key, use as is
                group_key_to_label[key] = str(key)
    else:
        # No sensitive_series, fallback to sensitive_mapping or str
        for key in group_rates_dict.keys():
            if sensitive_mapping:
                group_key_to_label[key] = sensitive_mapping.get(key, sensitive_mapping.get(str(key), str(key)))
            else:
                group_key_to_label[key] = str(key)

    sorted_items = sorted(
        group_rates_dict.items(),
        key=lambda x: float(group_key_to_label.get(x[0], x[0]))
        if str(group_key_to_label.get(x[0], x[0])).replace('.', '', 1).isdigit()
        else str(group_key_to_label.get(x[0], x[0]))
    )

    # Populate table rows with actual sensitive column values
    for group, rate in sorted_items:
        try:
            rate_val = float(rate)
            if rate_val != rate_val:  # NaN check
                rate_val = 0.0
        except Exception:
            rate_val = 0.0

        group_label = group_key_to_label.get(group, str(group))

        groups_list.append(group_label)
        rates_list.append(rate_val)

        pdf.cell(95, 8, group_label, border=1)
        pdf.cell(95, 8, f"{rate_val:.2f}", border=1, ln=True)

    pdf.ln(5)

    # Interpretation Section
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Interpretation", ln=True)
    pdf.set_font("Arial", size=10)

    interpretation = []

    if "Accuracy" in metrics:
        acc = metrics["Accuracy"]
        interpretation.append(
            f"The model achieved an accuracy of {acc:.2f}. "
            "However, high accuracy does not imply fairness, as the model may be systematically favoring one group over another."
        )

    if "Disparate Impact" in metrics:
        di = metrics["Disparate Impact"]
        if di == 0.0:
            interpretation.append(
                f"Disparate Impact is {di:.2f}. Severe bias detected: one group has zero selection rate compared to another, meaning it is completely excluded from positive outcomes."
            )
        elif di < 0.8 or di > 1.25:
            interpretation.append(
                f"Disparate Impact is {di:.2f}, which falls outside the fairness threshold [0.8, 1.25]. "
                "This indicates that one group is being disproportionately favored or disfavored compared to another."
            )
        else:
            interpretation.append(
                f"Disparate Impact is {di:.2f}, which lies within the fairness threshold. "
                "This suggests that the model is treating groups more equitably in terms of overall outcomes."
            )

    if "Equal Opportunity Diff" in metrics:
        eod = metrics["Equal Opportunity Diff"]
        if eod == 1.0 or eod == -1.0:
            interpretation.append(
                f"Equal Opportunity Difference is {eod:.2f}. Severe bias detected: one group has perfect true positive rate while another has none, showing extreme unfairness."
            )
        elif abs(eod) > 0.2:
            interpretation.append(
                f"Equal Opportunity Difference is {eod:.2f}, which exceeds the fairness tolerance. "
                "This means the model provides different true positive rates across groups, disadvantaging some."
            )
        else:
            interpretation.append(
                f"Equal Opportunity Difference is {eod:.2f}, which is within the acceptable range. "
                "This implies that the model's ability to correctly identify positives is relatively balanced."
            )

    if "Demographic Parity Diff" in metrics:
        dpd = metrics["Demographic Parity Diff"]
        interpretation.append(
            f"Demographic Parity Difference is {dpd:.2f}. "
            "A higher value indicates unequal positive prediction rates between groups, a key sign of potential bias."
        )

    for line in interpretation:
        pdf.multi_cell(0, 8, line)
    pdf.ln(5)

    # Fairness Summary
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Fairness Summary", ln=True)
    pdf.set_font("Arial", size=10)

    valid_rates = [r for r in rates_list if r is not None and not (isinstance(r, float) and r != r)]

    if valid_rates and groups_list:
        max_rate = max(valid_rates)
        min_rate = min(valid_rates)
        gap = max_rate - min_rate
        fairness_summary = (
            f"The highest selection rate among groups is {max_rate:.2f}, while the lowest is {min_rate:.2f}. "
            f"This results in a gap of {gap:.2f}, indicating {'substantial' if gap > 0.3 else 'minor'} bias in favor of the higher-rated group."
        )
        pdf.multi_cell(0, 8, fairness_summary)
    else:
        pdf.multi_cell(0, 8, "Fairness summary could not be computed due to missing or invalid group selection data.")

    # Selection Rates Chart
    if chart_path:
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Selection Rates Chart", ln=True)
        pdf.image(chart_path, x=10, y=None, w=180)

    pdf.output(output_path)
    return output_path