import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend for Flask/macOS
import matplotlib.pyplot as plt
import pandas as pd

def plot_selection_rates(y_pred, A_test, save_path="static/charts/selection.png", sensitive_mapping=None):

    df = pd.DataFrame({"y_pred": y_pred, "group": A_test})
    rates = df.groupby("group").mean()["y_pred"].sort_index()

    # Map numeric codes or other values back to readable labels
    if sensitive_mapping:
        group_names = [sensitive_mapping.get(g, str(g)) for g in rates.index]
    else:
        group_names = [str(g) for g in rates.index]

    # Plotting
    plt.figure(figsize=(8, 6))
    bars = plt.bar(group_names, rates.values, color="skyblue", edgecolor="black")

    # Title and axis labels
    plt.title("Selection Rates by Group", fontsize=16, fontweight="bold")
    plt.ylabel("Selection Rate", fontsize=12)
    plt.xlabel("Group", fontsize=12)

    # Ticks formatting
    plt.xticks(rotation=0, fontsize=11)
    plt.yticks(fontsize=11)

    plt.ylim(0, 1)
    plt.grid(color = "grey" ,linestyle=":",linewidth = 1, alpha=0.7)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="black"
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path