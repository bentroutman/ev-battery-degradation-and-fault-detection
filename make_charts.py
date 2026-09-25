# Creates the 8 charts, shown in the notebook and saved to charts/
# Run after clean_data.py (also runs both models)

import warnings
from pathlib import Path

import matplotlib

if __name__ == "__main__":
    matplotlib.use("Agg")  # save charts to files without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parent
CHARTS_PATH = ROOT / "charts"

THEME = {
    "font.family": "Arial", "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.grid": True, "grid.color": "#EBEBEB", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False, "axes.spines.bottom": False,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.color": "#4D4D4D", "ytick.color": "#4D4D4D",
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "axes.labelsize": 12, "axes.labelcolor": "black",
    "legend.frameon": False, "legend.fontsize": 10.5, "legend.title_fontsize": 11.5,
    "lines.linewidth": 2, "figure.dpi": 110, "savefig.dpi": 150,
}
plt.rcParams.update(THEME)  # every chart uses this theme

GREY, BLUE, RED = "#B3B3B3", "blue", "red"
STATUS_COLOR = {0: BLUE, 1: RED}
STATUS_MARKER = {0: "o", 1: "s"}
STATUS_NAME = {0: "Normal", 1: "Fault Reported"}

LABELS = {
    "spread_high_soc_median": "Cell spread at high charge (median)",
    "spread_high_soc_p90": "Cell spread at high charge (90th pct)",
    "spread_median": "Cell spread, whole charge (median)",
    "spread_max_p95": "Cell spread, max (95th pct)",
    "spread_trend": "Cell spread trend (per 10,000 km)",
    "pack_temp_median": "Pack temp while charging (median)",
    "pack_temp_p95": "Pack temp, max (95th pct)",
    "temp_rise_median": "Temp rise during charge (median)",
    "temp_probe_spread_median": "Temp probe spread (median)",
    "start_temp_median": "Temp at plug-in (median)",
    "charge_current_median": "Charging current (median)",
    "fast_charge_share": "Fast-charge share",
    "depth_of_discharge_mean": "Depth of discharge at plug-in",
    "soc_end_median": "Charge level at unplug (median)",
    "mileage_1000km": "Mileage (1,000 km)",
    "min_temp_mean": "Pack Temp During Charge (°C)",
    "prior_fast_charge_share": "Prior Fast-Charge Share",
    "prior_depth_of_discharge": "Prior Depth of Discharge (%)",
    "prior_pack_temp": "Prior Pack Temp (°C)",
}


# Title, subtitle, and an optional note (grey italics) above the plot
def add_titles(ax, title, subtitle, note=None):
    ax.figure.tight_layout()  # lay out the plot first so titles don't squeeze it
    y = 10
    if note:
        ax.annotate(note, xy=(0, 1), xycoords="axes fraction", xytext=(0, y), textcoords="offset points",
                    fontsize=10, color="#666666", style="italic", ha="left", va="bottom")
        y += 17
    ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, y), textcoords="offset points",
                fontsize=11.5, ha="left", va="bottom")
    ax.annotate(title, xy=(0, 1), xycoords="axes fraction", xytext=(0, y + 20), textcoords="offset points",
                fontsize=15, ha="left", va="bottom")


# Legend for fault reported vs normal
def add_status_legend(ax, **kwargs):
    for status in (1, 0):
        ax.plot([], [], ls="none", marker=STATUS_MARKER[status], color=STATUS_COLOR[status], ms=7.5, label=STATUS_NAME[status])
    ax.legend(title="Vehicle Status", alignment="left", **kwargs)


# Data charts
# Mileage range logged for each vehicle (shows the late start for faulty cars)
def vehicle_coverage(sessions_df):
    cars_df = sessions_df.groupby("car").agg(fault=("fault_label", "first"), km_first=("mileage_km", "min"),
                                             km_last=("mileage_km", "max"), capacity=("capacity_ah", "count"))
    cars_df = cars_df.sort_values(["km_first", "km_last"]).reset_index()
    fig, ax = plt.subplots(figsize=(9, 9))
    for i, car in cars_df.iterrows():
        color = STATUS_COLOR[car.fault]
        ax.plot([car.km_first / 1e3, car.km_last / 1e3], [i, i], color=color, lw=2.5, solid_capstyle="round")
        ax.plot(car.km_first / 1e3, i, marker=STATUS_MARKER[car.fault], color=color, ms=6, mec="white", mew=1)
        if car.capacity:
            ax.text(car.km_last / 1e3 + 2, i, "capacity data", va="center", fontsize=9, color="#666666")
    ax.set_yticks(range(len(cars_df)), cars_df.car.astype(str), fontsize=8)
    ax.set_xlabel("Mileage (1,000 km)")
    ax.set_ylabel("Vehicle")
    ax.grid(axis="y", visible=False)
    add_status_legend(ax, loc="lower right")
    add_titles(ax, "Mileage Range Logged per Vehicle", f"{len(cars_df)} vehicles, sorted by first logged mileage",
               note="Vehicles with fault reports start being logged at higher mileage")
    return fig


# Median cell spread and pack temperature by mileage, faulty vs normal
def signals_by_mileage(sessions_df):
    band_edges = np.arange(0, 125_000, 10_000)
    banded_df = sessions_df.assign(band=pd.cut(sessions_df.mileage_km, band_edges, labels=band_edges[:-1] / 1e3 + 5))
    panels = [("cell_spread_high_soc", "Cell-Voltage Spread at High Charge (mV)", 1e3),
              ("max_temp_mean", "Max Pack Temp While Charging (°C)", 1)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, (column, ylabel, scale) in zip(axes, panels):
        for status in (1, 0):
            band_groups = banded_df[banded_df.fault_label == status].groupby("band", observed=True)
            has_3_vehicles = band_groups["car"].nunique() >= 3  # only bands with 3+ vehicles
            x = band_groups[column].median().index.astype(float)[has_3_vehicles]
            ax.fill_between(x, band_groups[column].quantile(0.25)[has_3_vehicles] * scale, band_groups[column].quantile(0.75)[has_3_vehicles] * scale,
                            color=STATUS_COLOR[status], alpha=0.1, lw=0)
            ax.plot(x, band_groups[column].median()[has_3_vehicles] * scale, color=STATUS_COLOR[status], marker=STATUS_MARKER[status], ms=5)
        ax.set_xlabel("Mileage (1,000 km)")
        ax.set_ylabel(ylabel)
    add_status_legend(axes[0], loc="upper left")
    add_titles(axes[0], "Cell Imbalance and Pack Temperature by Mileage",
               "Median per 10,000 km band (bands with 3+ vehicles)", note="Shaded area = middle 50% of sessions")
    return fig


# Degradation model charts
# Every capacity reading vs the GAM's expected curve
def expected_retention(capacity_df, fade_curve_df):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.scatter(capacity_df.mileage_1000km, capacity_df.retention_pct, s=16, color=GREY, edgecolor="#7F7F7F", linewidth=0.3,
               label="Observed Retention (one charging session)")
    curve_df = fade_curve_df[fade_curve_df.mileage_1000km <= capacity_df.mileage_1000km.max()]
    ax.plot(curve_df.mileage_1000km, curve_df.expected_retention_pct, color=BLUE, lw=2.5, label="Expected Retention (GAM)")
    ax.axhline(100, color=RED, ls="--", lw=1.3, label="As-New Capacity")
    ax.set_xlabel("Mileage (1,000 km)")
    ax.set_ylabel("Capacity Retention (%)")
    ax.set_xlim(0, capacity_df.mileage_1000km.max() + 3)
    ax.legend(loc="lower left")
    add_titles(ax, "Expected Capacity Retention by Mileage",
               f"{capacity_df.car.nunique()} vehicles with capacity measurements ({len(capacity_df):,} charging sessions)")
    return fig


# Effect of each stress factor on retention, holding the others fixed
def partial_effects(stress_model, capacity_df, columns):
    fig, axes = plt.subplots(1, len(columns), figsize=(3.1 * len(columns), 3.8), sharey=True)
    for i, (ax, column) in enumerate(zip(axes, columns)):
        grid = stress_model.generate_X_grid(term=i)
        effect, ci = stress_model.partial_dependence(term=i, X=grid, width=0.95)
        center = stress_model.partial_dependence(term=i, X=capacity_df[columns].to_numpy()).mean()  # 0 = average session
        ax.fill_between(grid[:, i], ci[:, 0] - center, ci[:, 1] - center, color=BLUE, alpha=0.12, lw=0)
        ax.plot(grid[:, i], effect - center, color=BLUE)
        ax.axhline(0, color=RED, ls="--", lw=1)
        ax.set_xlim(*np.percentile(capacity_df[column], [1, 99]))
        ax.set_xlabel(LABELS[column], fontsize=10)
    axes[0].set_ylabel("Effect on Retention (pp)")
    add_titles(axes[0], "Partial Effects on Capacity Retention", "GAM with all stress factors, fit on all 8 vehicles",
               note="Shaded band = 95% interval; only mileage improved predictions on unseen vehicles")
    return fig


# Quality score per vehicle with 95% confidence intervals
def quality_scores(scores_df):
    scores_df = scores_df.sort_values("quality_score").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.axvline(0, color=RED, ls="--", lw=1.2)
    for i, car in scores_df.iterrows():
        color = STATUS_COLOR[car.fault_label]
        ax.plot([car.ci_low, car.ci_high], [i, i], color=color, lw=2, alpha=0.35, solid_capstyle="round")
        ax.plot(car.quality_score, i, marker=STATUS_MARKER[car.fault_label], color=color, ms=8, mec="white", mew=1)
        ax.text(scores_df.ci_high.max() + 0.4, i, car.flag.capitalize(), va="center", fontsize=10, color="#4D4D4D")
    ax.set_yticks(range(len(scores_df)), [f"Vehicle {car_id}" for car_id in scores_df.car])
    ax.set_xlabel("Quality Score (percentage points)")
    ax.set_xlim(scores_df.ci_low.min() - 0.5, scores_df.ci_high.max() + 3.4)
    ax.grid(axis="y", visible=False)
    add_status_legend(ax, loc="upper left")
    add_titles(ax, "Battery Quality Score by Vehicle", "Average of observed − expected capacity retention",
               note="Lines show 95% confidence intervals")
    return fig


# Fault model charts
# AUC at each checkpoint, XGBoost vs cell spread alone
def auc_by_mileage(auc_df):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for model, color, marker, label in [("xgb", BLUE, "o", "XGBoost Model"),
                                        ("spread_only", "#7F7F7F", "s", "Cell-Voltage Spread Only")]:
        auc_groups = auc_df[auc_df.model == model].groupby("checkpoint_km").auc
        x = auc_groups.mean().index / 1e3
        if model == "xgb":
            ax.fill_between(x, auc_groups.quantile(0.05), auc_groups.quantile(0.95), color=color, alpha=0.12, lw=0)
        ax.plot(x, auc_groups.mean(), color=color, marker=marker, ms=5, label=label)
    ax.axhline(0.5, color=RED, ls="--", lw=1.3)
    ax.text(auc_df.checkpoint_km.max() / 1e3, 0.52, "Random Guessing", ha="right", va="bottom", fontsize=10.5, color=RED)
    ax.set_ylim(0.1, 1.02)
    ax.set_xlabel("Odometer Checkpoint (1,000 km)")
    ax.set_ylabel("AUC")
    ax.legend(loc="lower right")
    add_titles(ax, "Fault Model Accuracy by Mileage", "Comparing vehicles at the same checkpoint",
               note="Shaded band = range across 25 cross-validation repeats")
    return fig


# Mean absolute SHAP value per feature
def feature_importance(importance):
    importance = importance.sort_values()
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    ax.barh([LABELS[feature] for feature in importance.index], importance.values, color=GREY, edgecolor="black", height=0.7)
    ax.set_xlabel("Mean Absolute SHAP Value")
    ax.grid(axis="y", visible=False)
    add_titles(ax, "What Drives the Fault Model", "Average effect of each feature on predicted fault risk")
    return fig


# Predicted fault risk for every car at one checkpoint
def risk_ranking(ranking_df, checkpoint_km=50_000):
    ranking_df = ranking_df.sort_values("risk").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for i, car in ranking_df.iterrows():
        color = STATUS_COLOR[car.fault_label]
        ax.plot([car.risk_low, car.risk_high], [i, i], color=color, lw=2, alpha=0.35, solid_capstyle="round")
        ax.plot(car.risk, i, marker=STATUS_MARKER[car.fault_label], color=color, ms=7.5, mec="white", mew=1)
    ax.set_yticks(range(len(ranking_df)), [f"Vehicle {car_id}" for car_id in ranking_df.car], fontsize=9.5)
    ax.set_xlabel("Predicted Fault Risk")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.grid(axis="y", visible=False)
    add_status_legend(ax, loc="upper left", bbox_to_anchor=(1.02, 1))
    add_titles(ax, f"Predicted Fault Risk at {checkpoint_km:,} km", f"{len(ranking_df)} vehicles observed at this checkpoint",
               note="Lines show the range across 25 cross-validation repeats")
    return fig


# Runs both models and saves all 8 charts
def main():
    import degradation_model
    import fault_model
    from clean_data import load_sessions

    warnings.filterwarnings("ignore")
    CHARTS_PATH.mkdir(exist_ok=True)
    sessions_df = load_sessions()
    degradation_results = degradation_model.run()
    fault_results = fault_model.run()
    charts = {
        "vehicle_coverage": vehicle_coverage(sessions_df),
        "signals_by_mileage": signals_by_mileage(sessions_df),
        "expected_retention": expected_retention(degradation_results["capacity_df"], degradation_results["fade_curve_df"]),
        "partial_effects": partial_effects(degradation_results["stress_model"], degradation_results["capacity_df"], degradation_model.STRESS_FACTORS),
        "quality_scores": quality_scores(degradation_results["scores_df"]),
        "auc_by_mileage": auc_by_mileage(fault_results["auc_by_checkpoint_df"]),
        "feature_importance": feature_importance(fault_results["importance"]),
        "risk_ranking_50k": risk_ranking(fault_results["ranking_df"]),
    }
    for name, fig in charts.items():
        fig.savefig(CHARTS_PATH / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"saved {len(charts)} charts to {CHARTS_PATH}")


if __name__ == "__main__":
    main()
