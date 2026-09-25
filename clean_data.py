# Cleans the raw EVBattery files into one row per charging session
# Run order: clean_data.py -> make_charts.py (runs both models) -> notebook
# Outputs: data/processed/snippets.parquet and sessions.parquet
# Each raw file is 128 readings, 10 seconds apart (~21 min)
# When capacity = 0, missing

import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_DATA_PATH = ROOT / "data" / "raw" / "battery_dataset3"
PROCESSED_DATA_PATH = ROOT / "data" / "processed"

# Columns of each raw 128 x 8 array
RAW_COLUMNS = ["volt", "current", "soc", "max_cell_volt", "min_cell_volt", "max_temp", "min_temp", "timestamp"]

FAST_CHARGE_A = 30.0  # gap between slow (5-20 A) and fast (50-120 A) charging


# Reads one raw .pkl file and returns (128 x 8 array and metadata dict)
def read_snippet(file_path):
    with open(file_path, "rb") as pkl_file:
        for _ in range(3):  # skip torch.save's 3 headers (no PyTorch needed)
            pickle.load(pkl_file)
        data, metadata = pickle.load(pkl_file)
    return data, dict(metadata)


# Summary stats for one snippet (runs on all 176,327 files)
def summarize_snippet(file_num):
    data, metadata = read_snippet(RAW_DATA_PATH / "data" / f"{file_num}.pkl")
    volt, current, soc, max_volt, min_volt, max_temp, min_temp, _ = data.T
    capacity = float(metadata["capacity"])
    charging_rows = current < 0  # rows where current is flowing
    high_soc_rows = soc >= 80
    cell_spread = max_volt - min_volt
    return {
        "file_num": file_num,
        "car": int(metadata["car"]),
        "charge_segment": int(metadata["charge_segment"]),
        "mileage_km": float(metadata["mileage"]),
        "capacity_ah": capacity if capacity > 0 else np.nan,
        "current_min": current.min(),
        "num_charging_rows": int(charging_rows.sum()),
        "current_charging_sum": float(current[charging_rows].sum()),
        "spread_high_soc_sum": float(cell_spread[high_soc_rows].sum()),
        "num_high_soc_rows": int(high_soc_rows.sum()),
        "temp_ok": bool((max_temp >= min_temp).all()),  # False means a sensor glitch (min > max)
        "soc_first": soc[0],
        "soc_last": soc[-1],
        "cell_spread_mean": cell_spread.mean(),
        "cell_spread_max": cell_spread.max(),
        "max_temp_first": max_temp[0],
        "min_temp_first": min_temp[0],
        "max_temp_mean": max_temp.mean(),
        "max_temp_max": max_temp.max(),
        "min_temp_mean": min_temp.mean(),
        "temp_spread_mean": (max_temp - min_temp).mean(),
    }


# Reads all snippets in parallel and adds each car's fault label
def load_snippets():
    num_files = len(os.listdir(RAW_DATA_PATH / "data"))
    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(summarize_snippet, range(num_files), chunksize=2000))
    snippets_df = pd.DataFrame(rows).sort_values("file_num").reset_index(drop=True)
    labels_df = pd.read_csv(RAW_DATA_PATH / "label" / "label.csv", index_col=0).rename(columns={"label": "fault_label"})
    return snippets_df.merge(labels_df, on="car", how="left")


# Fixes odometer spikes (e.g. a single 1,107,295 km reading) by interpolating
# Only readings that jump away from both neighbors are fixed, since one-way jumps are real logging gaps
def clean_odometer(sessions_df, max_jump_km=5000.0):
    fixed_mileage = []
    for _, car_df in sessions_df.groupby("car", sort=False):
        mileage = car_df["mileage_raw_km"].to_numpy(dtype=float).copy()
        for i in range(1, len(mileage) - 1):
            jump_from_prev, jump_from_next = mileage[i] - mileage[i - 1], mileage[i] - mileage[i + 1]
            if (jump_from_prev > max_jump_km and jump_from_next > max_jump_km) or \
                    (jump_from_prev < -max_jump_km and jump_from_next < -max_jump_km):
                mileage[i] = np.nan
        fixed_mileage.append(pd.Series(mileage, index=car_df.index).interpolate(limit_direction="both"))
    return pd.concat(fixed_mileage).reindex(sessions_df.index)


# Combines snippets into one row per charging session
def build_sessions(snippets_df):
    snippets_df = snippets_df.copy()
    temp_columns = ["max_temp_first", "min_temp_first", "max_temp_mean", "max_temp_max", "min_temp_mean", "temp_spread_mean"]
    snippets_df.loc[~snippets_df["temp_ok"], temp_columns] = np.nan  # ignore glitchy temperature readings
    snippets_df = snippets_df.sort_values("file_num")

    sessions_df = snippets_df.groupby(["car", "charge_segment"], sort=True).agg(
        first_file_num=("file_num", "min"),
        mileage_raw_km=("mileage_km", "first"),
        capacity_ah=("capacity_ah", "first"),
        soc_start=("soc_first", "first"),
        soc_end=("soc_last", "last"),
        num_charging_rows=("num_charging_rows", "sum"),
        current_charging_sum=("current_charging_sum", "sum"),
        spread_high_soc_sum=("spread_high_soc_sum", "sum"),
        num_high_soc_rows=("num_high_soc_rows", "sum"),
        cell_spread_mean=("cell_spread_mean", "mean"),
        cell_spread_max=("cell_spread_max", "max"),
        max_temp_start=("max_temp_first", "first"),
        min_temp_start=("min_temp_first", "first"),
        max_temp_mean=("max_temp_mean", "mean"),
        max_temp_max=("max_temp_max", "max"),
        min_temp_mean=("min_temp_mean", "mean"),
        temp_spread_mean=("temp_spread_mean", "mean"),
        fault_label=("fault_label", "first"),
    ).reset_index()

    # Average current while charging, as a positive number
    num_charging = sessions_df["num_charging_rows"].where(sessions_df["num_charging_rows"] > 0)
    num_high_soc = sessions_df["num_high_soc_rows"].where(sessions_df["num_high_soc_rows"] > 0)
    sessions_df["charge_current_a"] = -sessions_df["current_charging_sum"] / num_charging
    sessions_df["cell_spread_high_soc"] = sessions_df["spread_high_soc_sum"] / num_high_soc
    sessions_df["is_fast_charge"] = (sessions_df["charge_current_a"] > FAST_CHARGE_A).astype(float)
    sessions_df["depth_of_discharge"] = 100.0 - sessions_df["soc_start"]  # how drained the battery was at plug-in
    sessions_df["temp_rise"] = sessions_df["max_temp_mean"] - sessions_df["max_temp_start"]
    sessions_df["mileage_km"] = clean_odometer(sessions_df)
    return sessions_df.drop(columns=["current_charging_sum", "spread_high_soc_sum", "num_high_soc_rows"])


# Loads the cleaned sessions, sorted by car and time (used by both models)
def load_sessions():
    sessions_df = pd.read_parquet(PROCESSED_DATA_PATH / "sessions.parquet")
    return sessions_df.sort_values(["car", "first_file_num"]).reset_index(drop=True)


# Counts of the main data issues handled during cleaning
def data_quality():
    snippets_df = pd.read_parquet(PROCESSED_DATA_PATH / "snippets.parquet")
    sessions_df = load_sessions()
    has_capacity = sessions_df["capacity_ah"].notna()
    return pd.Series({
        "Snippets with no current (plugged in, idle)": int((snippets_df["current_min"] == 0).sum()),
        "Snippets with faulty temperature sensors": int((~snippets_df["temp_ok"]).sum()),
        "Odometer readings corrected": int((sessions_df["mileage_km"] != sessions_df["mileage_raw_km"]).sum()),
        "Sessions with a capacity measurement": int(has_capacity.sum()),
        "Vehicles with a capacity measurement": sessions_df.loc[has_capacity, "car"].nunique(),
    }, name="count")


# Checks whether capacity can be calculated from current and state of charge (Coulomb counting)
# Uses every 3rd session with a capacity reading where at least 40% of charge was added
def coulomb_check():
    snippets_df = pd.read_parquet(PROCESSED_DATA_PATH / "snippets.parquet")
    measured = snippets_df[snippets_df["capacity_ah"].notna()].sort_values("file_num").groupby(["car", "charge_segment"])
    rows = []
    for _, session_df in list(measured)[::3]:
        current = np.concatenate([read_snippet(RAW_DATA_PATH / "data" / f"{file_num}.pkl")[0][:, 1]
                                  for file_num in session_df["file_num"]])
        soc_gain = session_df["soc_last"].iloc[-1] - session_df["soc_first"].iloc[0]
        if soc_gain >= 40:
            amp_hours = -current.sum() * 10 / 3600  # readings are 10 seconds apart
            rows.append({"dataset_capacity": session_df["capacity_ah"].iloc[0], "calculated_capacity": amp_hours / (soc_gain / 100)})
    coulomb_df = pd.DataFrame(rows)
    print(f"{len(coulomb_df)} sessions, correlation with the dataset's capacity: r = {coulomb_df.corr().iloc[0, 1]:.2f}")


# Builds and saves both data files
def main():
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
    snippets_df = load_snippets()
    snippets_df.to_parquet(PROCESSED_DATA_PATH / "snippets.parquet", index=False)
    sessions_df = build_sessions(snippets_df)
    sessions_df.to_parquet(PROCESSED_DATA_PATH / "sessions.parquet", index=False)
    print(f"snippets: {len(snippets_df):,}  sessions: {len(sessions_df):,}  cars: {snippets_df.car.nunique()}")


if __name__ == "__main__":
    main()
