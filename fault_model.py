# XGBoost model that scores each vehicle's fault risk every 5,000 km
# Run after clean_data.py
# Outputs results/fault_model_ranking_50k.csv and fault_model_summary.json
# Mileage is not a feature because faulty cars were first logged at ~11,000 km vs ~65 km for normal ones
# Cross-validation is grouped by vehicle, and the main AUC only compares cars at the same mileage

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import DMatrix, XGBClassifier

from clean_data import load_sessions

ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "results"

CHECKPOINT_KM = 5_000  # score each car every 5,000 km, up to 120,000 km
MAX_KM = 120_000
LAST_SESSIONS = 40  # features use the car's 40 most recent sessions
MIN_SESSIONS = 15  # skip checkpoints with fewer sessions than this
MAX_GAP_KM = 5_000  # skip checkpoints with no session in the last 5,000 km
NUM_REPEATS = 25
NUM_FOLDS = 5
MIN_PER_CLASS = 3  # checkpoints need 3+ faulty and 3+ normal cars for the AUC

# Kept small and untuned, since there are only ~450 rows from 45 vehicles
# Feature order matters, so keep it fixed
XGB_PARAMS = dict(n_estimators=300, learning_rate=0.03, max_depth=2, min_child_weight=2,
                  subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0, eval_metric="logloss")

FEATURES = [
    "spread_high_soc_median", "spread_high_soc_p90", "spread_median", "spread_max_p95", "spread_trend",
    "pack_temp_median", "pack_temp_p95", "temp_rise_median", "temp_probe_spread_median", "start_temp_median",
    "charge_current_median", "fast_charge_share", "depth_of_discharge_mean", "soc_end_median",
]


# Change in cell spread per 10,000 km over the window
def spread_trend(mileage, spread):
    has_value = ~np.isnan(spread)
    if has_value.sum() < 5 or np.ptp(mileage[has_value]) < 500:
        return np.nan
    return float(np.polyfit(mileage[has_value] / 10_000.0, spread[has_value], 1)[0])


# The 14 features for one car's recent sessions
def window_features(recent_df):
    spread = recent_df["cell_spread_high_soc"]
    return {
        "spread_high_soc_median": spread.median(),
        "spread_high_soc_p90": spread.quantile(0.9),
        "spread_median": recent_df["cell_spread_mean"].median(),
        "spread_max_p95": recent_df["cell_spread_max"].quantile(0.95),
        "spread_trend": spread_trend(recent_df["mileage_km"].to_numpy(), spread.to_numpy()),
        "pack_temp_median": recent_df["max_temp_mean"].median(),
        "pack_temp_p95": recent_df["max_temp_max"].quantile(0.95),
        "temp_rise_median": recent_df["temp_rise"].median(),
        "temp_probe_spread_median": recent_df["temp_spread_mean"].median(),
        "start_temp_median": recent_df["min_temp_start"].median(),
        "charge_current_median": recent_df["charge_current_a"].median(),
        "fast_charge_share": recent_df["is_fast_charge"].mean(),
        "depth_of_discharge_mean": recent_df["depth_of_discharge"].mean(),
        "soc_end_median": recent_df["soc_end"].median(),
    }


# One row per car at each checkpoint, using only sessions logged at or before it
def build_checkpoints():
    sessions_df = load_sessions()
    rows = []
    for car, car_df in sessions_df.groupby("car", sort=False):
        mileage = car_df["mileage_km"].to_numpy()
        for checkpoint in range(CHECKPOINT_KM, MAX_KM + 1, CHECKPOINT_KM):
            past_df = car_df[mileage <= checkpoint]
            if len(past_df) < MIN_SESSIONS or checkpoint - past_df["mileage_km"].iloc[-1] > MAX_GAP_KM:
                continue
            rows.append({"car": car, "checkpoint_km": checkpoint, "fault_label": int(car_df["fault_label"].iloc[0]),
                         **window_features(past_df.tail(LAST_SESSIONS))})
    return pd.DataFrame(rows), sessions_df["car"].nunique()


# XGBoost model with the fixed settings above
def make_xgb(seed):
    return XGBClassifier(**XGB_PARAMS, random_state=seed, n_jobs=4)


# Each car counts equally in training, no matter how many checkpoints it has
def car_weights(checkpoints_df):
    return (1.0 / checkpoints_df.groupby("car")["car"].transform("size")).to_numpy()


# AUC within each checkpoint (cars at the same mileage), weighted by faulty/normal pairs
def matched_mileage_auc(checkpoints_df, score_column):
    rows = []
    for checkpoint, checkpoint_df in checkpoints_df.groupby("checkpoint_km"):
        num_fault, num_normal = int(checkpoint_df.fault_label.sum()), int((1 - checkpoint_df.fault_label).sum())
        if num_fault >= MIN_PER_CLASS and num_normal >= MIN_PER_CLASS:
            rows.append({"checkpoint_km": checkpoint, "num_fault": num_fault, "num_normal": num_normal,
                         "auc": roc_auc_score(checkpoint_df.fault_label, checkpoint_df[score_column])})
    auc_df = pd.DataFrame(rows)
    return float(np.average(auc_df.auc, weights=auc_df.num_fault * auc_df.num_normal)), auc_df


# Grouped cross-validation: every row is scored by a model that never saw that car
# Also fits a mileage-only model to show why mileage alone can be misleading
def cross_validate(checkpoints_df):
    X = checkpoints_df[FEATURES].to_numpy()
    y = checkpoints_df["fault_label"].to_numpy()
    cars = checkpoints_df["car"].to_numpy()
    weights = car_weights(checkpoints_df)
    mileage = checkpoints_df[["checkpoint_km"]].to_numpy() / 1e4
    prediction_dfs = []
    for repeat in range(NUM_REPEATS):
        folds = StratifiedGroupKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=repeat)
        risk, mileage_only = np.zeros(len(checkpoints_df)), np.zeros(len(checkpoints_df))
        for train, test in folds.split(X, y, cars):
            risk[test] = make_xgb(repeat).fit(X[train], y[train], sample_weight=weights[train]).predict_proba(X[test])[:, 1]
            mileage_model = LogisticRegression().fit(mileage[train], y[train], sample_weight=weights[train])
            mileage_only[test] = mileage_model.predict_proba(mileage[test])[:, 1]
        prediction_dfs.append(pd.DataFrame({"repeat": repeat, "row": np.arange(len(checkpoints_df)), "risk": risk, "mileage_only": mileage_only}))
    return pd.concat(prediction_dfs, ignore_index=True)


# AUCs for each repeat: XGBoost, cell spread alone, and mileage alone
def score_repeats(checkpoints_df, predictions_df):
    repeat_rows, auc_dfs = [], []
    checkpoints_df = checkpoints_df.assign(spread_only=checkpoints_df["spread_high_soc_median"])
    for repeat, repeat_df in predictions_df.groupby("repeat"):
        scored_df = checkpoints_df.assign(risk=repeat_df["risk"].to_numpy(), mileage_only=repeat_df["mileage_only"].to_numpy())
        xgb_auc, xgb_auc_df = matched_mileage_auc(scored_df, "risk")
        spread_auc, spread_auc_df = matched_mileage_auc(scored_df, "spread_only")
        repeat_rows.append({"matched_auc": xgb_auc, "matched_auc_spread_only": spread_auc,
                            "pooled_auc": roc_auc_score(scored_df.fault_label, scored_df.risk),
                            "pooled_auc_mileage_only": roc_auc_score(scored_df.fault_label, scored_df.mileage_only)})
        auc_dfs += [xgb_auc_df.assign(repeat=repeat, model="xgb"), spread_auc_df.assign(repeat=repeat, model="spread_only")]
    return pd.DataFrame(repeat_rows), pd.concat(auc_dfs, ignore_index=True)


# Ranks the cars at one checkpoint by fault risk (averaged over 25 repeats)
def rank_vehicles(checkpoints_df, predictions_df, checkpoint_km=50_000):
    risk_by_row = predictions_df.groupby("row")["risk"]
    ranking_df = checkpoints_df[["car", "checkpoint_km", "fault_label"]].assign(
        risk=risk_by_row.mean().to_numpy(), risk_low=risk_by_row.quantile(0.05).to_numpy(), risk_high=risk_by_row.quantile(0.95).to_numpy())
    return ranking_df[ranking_df.checkpoint_km == checkpoint_km].sort_values("risk", ascending=False).reset_index(drop=True)


# Mean absolute SHAP value per feature (how much it moves the predicted risk)
def feature_importance(checkpoints_df):
    model = make_xgb(0).fit(checkpoints_df[FEATURES], checkpoints_df["fault_label"], sample_weight=car_weights(checkpoints_df))
    shap_values = model.get_booster().predict(DMatrix(checkpoints_df[FEATURES]), pred_contribs=True)[:, :-1]
    return pd.Series(np.abs(shap_values).mean(0), index=FEATURES).sort_values(ascending=False)


# Runs the model and saves the results
def run():
    RESULTS_PATH.mkdir(exist_ok=True)
    checkpoints_df, num_cars = build_checkpoints()
    predictions_df = cross_validate(checkpoints_df)
    repeat_results_df, auc_by_checkpoint_df = score_repeats(checkpoints_df, predictions_df)
    ranking_df = rank_vehicles(checkpoints_df, predictions_df)
    importance = feature_importance(checkpoints_df)

    # Mean and 5-95% range of an AUC across the 25 repeats
    def mean_and_range(column):
        values = repeat_results_df[column]
        return {"mean": float(values.mean()), "low": float(values.quantile(0.05)), "high": float(values.quantile(0.95))}

    summary = {
        "checkpoint_rows": len(checkpoints_df),
        "vehicles": int(checkpoints_df.car.nunique()),
        "vehicles_with_fault_reports": int(checkpoints_df.groupby("car").fault_label.first().sum()),
        "vehicles_skipped_too_little_data": num_cars - int(checkpoints_df.car.nunique()),
        "matched_mileage_auc": mean_and_range("matched_auc"),
        "matched_mileage_auc_spread_only": mean_and_range("matched_auc_spread_only"),
        "pooled_auc": mean_and_range("pooled_auc"),
        "pooled_auc_mileage_only": mean_and_range("pooled_auc_mileage_only"),
        "ranking_50k_km": {"vehicles": len(ranking_df), "with_fault_reports": int(ranking_df.fault_label.sum()),
                           "top_5_with_fault_reports": int(ranking_df.head(5).fault_label.sum()),
                           "top_10_with_fault_reports": int(ranking_df.head(10).fault_label.sum())},
        "feature_importance": importance.round(4).to_dict(),
    }
    ranking_df.to_csv(RESULTS_PATH / "fault_model_ranking_50k.csv", index=False)
    (RESULTS_PATH / "fault_model_summary.json").write_text(json.dumps(summary, indent=2))
    return {"checkpoints_df": checkpoints_df, "predictions_df": predictions_df, "summary": summary,
            "auc_by_checkpoint_df": auc_by_checkpoint_df, "ranking_df": ranking_df, "importance": importance}


if __name__ == "__main__":
    results = run()
    print(json.dumps(results["summary"], indent=2))
    print(results["ranking_df"].round(3).to_string())
