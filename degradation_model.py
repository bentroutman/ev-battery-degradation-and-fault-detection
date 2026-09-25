# GAM of expected capacity retention vs mileage, plus a quality score for each vehicle
# Run after clean_data.py
# Outputs results/degradation_*.csv and degradation_summary.json
# Only 8 vehicles have capacity data, so every model is tested by leaving one vehicle out
# Quality score = average of (observed - expected retention) for each vehicle

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pygam import LinearGAM, f, l, s

from clean_data import load_sessions

ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "results"

AS_NEW_KM = 5_000  # as-new capacity = median reading below 5,000 km
LAMBDAS = [1, 10, 100, 1000, 10000]  # smoothing penalties to test
NUM_SPLINES = 8
MIN_BLOCKS = 5  # 5,000 km blocks needed for a confidence interval
SPLINE_COLUMNS = {"mileage_1000km", "min_temp_mean"}  # curves; everything else is a straight line

# Stress factors tested, and the model versions compared
STRESS_FACTORS = ["mileage_1000km", "min_temp_mean", "prior_fast_charge_share", "prior_depth_of_discharge", "prior_pack_temp"]
MODELS = {
    "Mileage": ["mileage_1000km"],
    "Mileage + charging temp": ["mileage_1000km", "min_temp_mean"],
    "Mileage + charging temp + fast charging": ["mileage_1000km", "min_temp_mean", "prior_fast_charge_share"],
    "Mileage + charging temp + depth of discharge": ["mileage_1000km", "min_temp_mean", "prior_depth_of_discharge"],
    "Mileage + charging temp + pack temp": ["mileage_1000km", "min_temp_mean", "prior_pack_temp"],
    "All stress factors": STRESS_FACTORS,
}


# One row per session with a capacity reading, plus stress history from earlier sessions only
def build_capacity_table():
    sessions_df = load_sessions()
    car_groups = sessions_df.groupby("car", sort=False)
    prior_mean = lambda s: s.shift(1).expanding().mean()  # average of earlier sessions only
    sessions_df["prior_fast_charge_share"] = car_groups["is_fast_charge"].transform(prior_mean)
    sessions_df["prior_depth_of_discharge"] = car_groups["depth_of_discharge"].transform(prior_mean)
    sessions_df["prior_pack_temp"] = car_groups["max_temp_mean"].transform(prior_mean)
    sessions_df["prior_sessions"] = car_groups.cumcount()

    capacity_df = sessions_df[sessions_df["capacity_ah"].notna()].copy()
    num_readings = len(capacity_df)
    # Drop BMS glitches: readings more than 1.5 Ah from the car's rolling median
    rolling_median = capacity_df.groupby("car")["capacity_ah"].transform(
        lambda s: s.rolling(15, center=True, min_periods=5).median())
    is_outlier = (capacity_df["capacity_ah"] - rolling_median).abs() > 1.5
    capacity_df = capacity_df[~is_outlier]
    capacity_df = capacity_df[capacity_df["prior_sessions"] >= 10]  # need history for the stress factors
    capacity_df = capacity_df.dropna(subset=["min_temp_mean", "prior_pack_temp", "prior_fast_charge_share", "prior_depth_of_discharge"])

    as_new_capacity = capacity_df.loc[capacity_df["mileage_km"] < AS_NEW_KM, "capacity_ah"].median()
    capacity_df["retention_pct"] = 100.0 * capacity_df["capacity_ah"] / as_new_capacity
    capacity_df["mileage_1000km"] = capacity_df["mileage_km"] / 1000.0
    counts = {"capacity_readings": num_readings, "outliers_removed": int(is_outlier.sum()),
              "sessions_used": len(capacity_df), "vehicles": capacity_df["car"].nunique(), "as_new_capacity_ah": float(as_new_capacity)}
    return capacity_df.reset_index(drop=True), counts


# Builds the pyGAM formula, e.g. s(mileage) + f(vehicle)
def gam_terms(columns, lam, vehicle_baseline):
    terms = None
    for i, column in enumerate(columns):
        term = s(i, n_splines=NUM_SPLINES, lam=lam) if column in SPLINE_COLUMNS else l(i, lam=lam)
        terms = term if terms is None else terms + term
    if vehicle_baseline:
        terms = terms + f(len(columns), lam=lam)
    return terms


# GAM with an optional baseline per vehicle
# predict() uses the average baseline, so it can score cars it hasn't seen
class RetentionGAM:

    def __init__(self, columns, lam, vehicle_baseline=True):
        self.columns, self.lam, self.vehicle_baseline = columns, lam, vehicle_baseline

    def fit(self, capacity_df):
        X = capacity_df[self.columns].to_numpy()
        if self.vehicle_baseline:
            self.car_codes = {car: i for i, car in enumerate(sorted(capacity_df["car"].unique()))}
            X = np.column_stack([X, capacity_df["car"].map(self.car_codes).to_numpy()])
        self.gam = LinearGAM(gam_terms(self.columns, self.lam, self.vehicle_baseline)).fit(X, capacity_df["retention_pct"].to_numpy())
        return self

    def predict(self, capacity_df):
        X = capacity_df[self.columns].to_numpy()
        if not self.vehicle_baseline:
            return self.gam.predict(X)
        X = np.column_stack([X, np.zeros(len(X))])
        num_columns = len(self.columns)
        baseline_input = np.zeros((len(self.car_codes), num_columns + 1))
        baseline_input[:, -1] = list(self.car_codes.values())
        avg_baseline = self.gam.partial_dependence(term=num_columns, X=baseline_input).mean()
        curves = sum(self.gam.partial_dependence(term=i, X=X) for i in range(num_columns))
        return curves + avg_baseline + self.gam.coef_[-1]


# Predicts each vehicle using a model trained on the other 7
def leave_one_vehicle_out(capacity_df, columns, lam, vehicle_baseline):
    predictions = np.full(len(capacity_df), np.nan)
    for car in capacity_df["car"].unique():
        is_test_car = (capacity_df["car"] == car).to_numpy()
        predictions[is_test_car] = RetentionGAM(columns, lam, vehicle_baseline).fit(capacity_df[~is_test_car]).predict(capacity_df[is_test_car])
    return predictions


# RMSE overall and averaged per vehicle
def errors(capacity_df, predictions):
    squared_errors = (capacity_df["retention_pct"] - predictions) ** 2
    return {"rmse": float(np.sqrt(squared_errors.mean())),
            "rmse_avg_per_vehicle": float(np.sqrt(squared_errors.groupby(capacity_df["car"]).mean()).mean())}


# Tests every model version on unseen vehicles and keeps each one's best smoothing level
def compare_models(capacity_df):
    rows = []
    for name, columns in MODELS.items():
        for vehicle_baseline in (False, True):
            best_model = None
            for lam in LAMBDAS:
                model_errors = errors(capacity_df, leave_one_vehicle_out(capacity_df, columns, lam, vehicle_baseline))
                if best_model is None or model_errors["rmse_avg_per_vehicle"] < best_model["rmse_avg_per_vehicle"]:
                    best_model = {"model": name, "vehicle_baseline": vehicle_baseline, "best_lambda": lam, **model_errors}
            rows.append(best_model)
    fleet_average = np.array([capacity_df.loc[capacity_df.car != car, "retention_pct"].mean() for car in capacity_df["car"]])
    rows.append({"model": "Fleet average (no model)", "vehicle_baseline": False, "best_lambda": np.nan, **errors(capacity_df, fleet_average)})
    return pd.DataFrame(rows).sort_values("rmse_avg_per_vehicle").reset_index(drop=True)


# 95% confidence interval for a car's average residual
# Resamples 5,000 km blocks, since nearby sessions are similar
def block_bootstrap_ci(residuals, mileage, block_km=5000, num_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    blocks = (mileage // block_km).astype(int)
    unique_blocks = np.unique(blocks)
    block_means = np.array([residuals[blocks == b].mean() for b in unique_blocks])
    block_sizes = np.array([(blocks == b).sum() for b in unique_blocks])
    samples = rng.integers(0, len(unique_blocks), (num_boot, len(unique_blocks)))
    boot_means = (block_means[samples] * block_sizes[samples]).sum(1) / block_sizes[samples].sum(1)
    return np.percentile(boot_means, [2.5, 97.5]), len(unique_blocks)


# Quality score, confidence interval, and trend for each vehicle
def quality_scores(capacity_df):
    rows = []
    for car, car_df in capacity_df.groupby("car"):
        (low, high), num_blocks = block_bootstrap_ci(car_df["residual"].to_numpy(), car_df["mileage_km"].to_numpy())
        mileage_span = np.ptp(car_df["mileage_km"].to_numpy())
        trend = np.polyfit(car_df["mileage_km"] / 10_000, car_df["residual"], 1)[0] if mileage_span > 10_000 else np.nan
        rows.append({"car": car, "fault_label": int(car_df["fault_label"].iloc[0]), "sessions": len(car_df),
                     "quality_score": car_df["residual"].mean(), "ci_low": low, "ci_high": high,
                     "blocks": num_blocks, "trend_per_10k_km": trend})
    scores_df = pd.DataFrame(rows).sort_values("quality_score").reset_index(drop=True)
    scores_df["flag"] = np.where(scores_df["ci_high"] < 0, "worse than expected",
                                 np.where(scores_df["ci_low"] > 0, "better than expected", "as expected"))
    scores_df.loc[scores_df["blocks"] < MIN_BLOCKS, "flag"] = "insufficient data"
    return scores_df


# Runs the model and saves the results
def run():
    RESULTS_PATH.mkdir(exist_ok=True)
    capacity_df, counts = build_capacity_table()
    comparison_df = compare_models(capacity_df)

    best_model = comparison_df[(comparison_df.model == "Mileage") & comparison_df.vehicle_baseline].iloc[0]
    lam = float(best_model["best_lambda"])
    capacity_df["expected"] = leave_one_vehicle_out(capacity_df, MODELS["Mileage"], lam, vehicle_baseline=True)
    capacity_df["residual"] = capacity_df["retention_pct"] - capacity_df["expected"]
    final_model = RetentionGAM(MODELS["Mileage"], lam).fit(capacity_df)

    # All stress factors, only used for the partial effects chart
    stress_model = LinearGAM(gam_terms(STRESS_FACTORS, lam, False)).fit(capacity_df[STRESS_FACTORS].to_numpy(), capacity_df["retention_pct"].to_numpy())

    fade_curve_df = pd.DataFrame({"mileage_1000km": np.linspace(0, 115, 116)})
    fade_curve_df["expected_retention_pct"] = final_model.predict(fade_curve_df)
    expected_at = lambda thousand_km: float(fade_curve_df.loc[fade_curve_df.mileage_1000km == thousand_km, "expected_retention_pct"].iloc[0])

    scores_df = quality_scores(capacity_df)
    summary = {
        **counts,
        "model": "s(mileage) + f(vehicle)",
        "lambda": lam,
        "rmse_unseen_vehicles": float(best_model["rmse"]),
        "rmse_fleet_average": float(comparison_df.loc[comparison_df.model.str.startswith("Fleet"), "rmse"].iloc[0]),
        "expected_retention_50k_km": expected_at(50),
        "expected_retention_100k_km": expected_at(100),
    }
    scores_df.to_csv(RESULTS_PATH / "degradation_quality_scores.csv", index=False)
    comparison_df.to_csv(RESULTS_PATH / "degradation_model_comparison.csv", index=False)
    (RESULTS_PATH / "degradation_summary.json").write_text(json.dumps(summary, indent=2))
    return {"capacity_df": capacity_df, "stress_model": stress_model, "summary": summary, "scores_df": scores_df,
            "comparison_df": comparison_df, "fade_curve_df": fade_curve_df}


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    results = run()
    print(json.dumps(results["summary"], indent=2))
    print(results["comparison_df"].round(3).to_string())
    print(results["scores_df"].round(2).to_string())
