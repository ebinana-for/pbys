"""
Age & Gender KPI Evaluation Script
Separates Driver and Passenger based on spatial bbox_cx.
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from sklearn.metrics import confusion_matrix, classification_report

def extract_seat_dataframe(df_valid, condition):
    """
    Filters the dataframe based on a spatial condition applied to bbox_cx.
    """
    return df_valid[condition(df_valid['bbox_cx'])].copy()

def compute_metrics_for_seat(seat_name, df_seat, gt_age, gt_gender):
    """
    Computes Age MAE and Gender predictions for a specific seat on a single video.
    """
    video_frames = len(df_seat)
    if video_frames == 0:
        return None

    video_mae = (abs(df_seat['age_mean'] - gt_age)).sum() / video_frames
    video_age_max = df_seat['age_mean'].max()
    video_age_min = df_seat['age_mean'].min()
    video_amp = video_age_max - video_age_min
    video_std = df_seat['age_mean'].std() if video_frames > 1 else 0.0
    
    video_y_true = [gt_gender] * video_frames
    video_y_pred = df_seat['gender_mean'].tolist()
    tn, fp, fn, tp = confusion_matrix(video_y_true, video_y_pred, labels=[0, 1]).ravel()
    video_acc = (tp + tn) / video_frames

    return {
        "Seat": seat_name,
        "Total_Frames": video_frames,
        "Age_MAE": round(video_mae, 2),
        "Age_Amplitude": round(video_amp, 2),
        "Age_Std_Dev": round(video_std, 2) if pd.notna(video_std) else 0.0,
        "Gender_Accuracy": round(video_acc, 4),
        "Gender_TP_Female": tp,
        "Gender_TN_Male": tn,
        "Gender_FP_Male_Predicted_Female": fp,
        "Gender_FN_Female_Predicted_Male": fn,
        "_y_true": video_y_true, # Hidden keys for global aggregation
        "_y_pred": video_y_pred,
        "_age_error_sum": (abs(df_seat['age_mean'] - gt_age)).sum()
    }

def aggregate_global_kpis(results_list, seat_name):
    """
    Aggregates video-level metrics into a global row for the specific seat.
    """
    seat_results = [res for res in results_list if res["Seat"] == seat_name]
    
    if not seat_results:
        return None

    total_frames = sum([res["Total_Frames"] for res in seat_results])
    
    # Age Metrics
    global_mae = np.mean([res["Age_MAE"] for res in seat_results])
    global_amp = np.mean([res["Age_Amplitude"] for res in seat_results])
    global_std = np.mean([res["Age_Std_Dev"] for res in seat_results])
    
    # Gender Metrics
    y_true_global = []
    y_pred_global = []
    for res in seat_results:
        y_true_global.extend(res["_y_true"])
        y_pred_global.extend(res["_y_pred"])
        
    tn_g, fp_g, fn_g, tp_g = confusion_matrix(y_true_global, y_pred_global, labels=[0, 1]).ravel()
    
    report = classification_report(
        y_true_global, 
        y_pred_global, 
        target_names=["Male", "Female"], 
        output_dict=True, 
        zero_division=0
    )
    
    return {
        "Subject": f"=== GLOBAL {seat_name.upper()} ===",
        "Seat": seat_name,
        "Total_Frames": total_frames,
        "Age_MAE": round(global_mae, 2),
        "Age_Amplitude": round(global_amp, 2),
        "Age_Std_Dev": round(global_std, 2),
        "Gender_Accuracy": round(report["accuracy"], 4),
        "Gender_TP_Female": tp_g,
        "Gender_TN_Male": tn_g,
        "Gender_FP_Male_Predicted_Female": fp_g,
        "Gender_FN_Female_Predicted_Male": fn_g,
        "Male_Precision": round(report["Male"]["precision"], 4),
        "Male_Recall": round(report["Male"]["recall"], 4),
        "Male_F1": round(report["Male"]["f1-score"], 4),
        "Female_Precision": round(report["Female"]["precision"], 4),
        "Female_Recall": round(report["Female"]["recall"], 4),
        "Female_F1": round(report["Female"]["f1-score"], 4)
    }

def compute_kpis(csv_dir, output_csv, split_x):
    all_csvs = glob.glob(os.path.join(csv_dir, "*_out_age.csv"))
    results_list = []
    
    print(f"Analyzing {len(all_csvs)} files. Spatial split for seats at X={split_x}px.")
    
    for csv_file in all_csvs:
        filename = Path(csv_file).stem
        parts = filename.split('_')
        
        try:
            gt_age = int(parts[2])
            gt_gender_str = parts[3].upper()
            gt_gender = 0 if gt_gender_str == 'M' else 1
        except (IndexError, ValueError):
            print(f"Skipped (cannot parse GT from filename): {filename}")
            continue
            
        df = pd.read_csv(csv_file)
        
        # Filter valid frames (Boolean Trap filters to -1)
        df_valid = df[df['age_mean'] != -1].copy()
        
        if df_valid.empty:
            continue

        # Split Dataframes based on spatial position
        df_driver = extract_seat_dataframe(df_valid, lambda cx: cx < split_x)
        df_passenger = extract_seat_dataframe(df_valid, lambda cx: cx >= split_x)

        # Compute Metrics
        res_driver = compute_metrics_for_seat("Driver", df_driver, gt_age, gt_gender)
        res_passenger = compute_metrics_for_seat("Passenger", df_passenger, gt_age, gt_gender)

        if res_driver:
            res_driver["Subject"] = filename.replace("_out_age", "")
            results_list.append(res_driver)
            
        if res_passenger:
            res_passenger["Subject"] = filename.replace("_out_age", "")
            results_list.append(res_passenger)
            
    if not results_list:
        print("No valid frames found across all CSVs.")
        return

    # Generate Global Summaries
    global_driver = aggregate_global_kpis(results_list, "Driver")
    global_passenger = aggregate_global_kpis(results_list, "Passenger")
    
    if global_driver:
        results_list.append(global_driver)
    if global_passenger:
        results_list.append(global_passenger)

    # Clean up hidden variables used for global calculation
    for res in results_list:
        res.pop("_y_true", None)
        res.pop("_y_pred", None)
        res.pop("_age_error_sum", None)
        
        # Fill missing string columns for individual video rows
        for col in ["Male_Precision", "Male_Recall", "Male_F1", "Female_Precision", "Female_Recall", "Female_F1"]:
            if col not in res:
                res[col] = ""

    # Reorder columns
    cols_order = ["Subject", "Seat", "Total_Frames", "Age_MAE", "Age_Amplitude", "Age_Std_Dev", 
                  "Gender_Accuracy", "Gender_TP_Female", "Gender_TN_Male", "Gender_FP_Male_Predicted_Female", 
                  "Gender_FN_Female_Predicted_Male", "Male_Precision", "Male_Recall", "Male_F1", 
                  "Female_Precision", "Female_Recall", "Female_F1"]
                  
    df_kpi = pd.DataFrame(results_list)[cols_order]
    
    # Sort for readability (All Driver videos, then Global Driver, then Passenger...)
    df_kpi.sort_values(by=["Seat", "Subject"], inplace=True)
    
    df_kpi.to_csv(output_csv, index=False, sep=";")
    print(f"\nSuccess! Seat-separated KPIs have been exported to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Seat-Separated KPIs for Age/Gender.")
    parser.add_argument("--input", required=True, help="Source folder containing the _out_age.csv files")
    parser.add_argument("--output", default="kpi_results.csv", help="Output CSV file path")
    parser.add_argument("--split_x", type=float, default=720.0, help="X coordinate to split Driver (left) from Passenger (right).")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    compute_kpis(args.input, args.output, args.split_x)