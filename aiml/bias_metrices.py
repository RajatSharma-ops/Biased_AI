from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference, equalized_odds_ratio
from sklearn.metrics import accuracy_score

def evaluate_bias(models, X_test, y_test, A_test):
    results = {}

    # If a single model is passed, wrap it in a dict
    if not isinstance(models, dict):
        models = {"Model": models}

    for name, model in models.items():
        y_pred = model.predict(X_test)

        metrics = {
            "Accuracy": round(accuracy_score(y_test, y_pred), 2),
            "Demographic Parity Diff": round(demographic_parity_difference(y_test, y_pred, sensitive_features=A_test), 2),
            "Equal Opportunity Diff": round(equalized_odds_ratio(y_test, y_pred, sensitive_features=A_test), 2)
        }

        # Disparate Impact (min selection rate / max selection rate)
        rates = MetricFrame(metrics=selection_rate, y_true=y_test, y_pred=y_pred, sensitive_features=A_test)
        metrics["Disparate Impact"] = round((rates.group_min() / rates.group_max()),2)
        results[name] = metrics

    return metrics, rates, y_pred