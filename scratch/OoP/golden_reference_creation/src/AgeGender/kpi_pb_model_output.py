import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

def compute_kpis(csv_dir, output_csv="kpi_results.csv"):
    all_csvs = glob.glob(os.path.join(csv_dir, "*_out_age.csv"))
    
    total_frames_global = 0
    age_error_sum_global = 0
    y_true_gender_global = []
    y_pred_gender_global = []
    age_amplitudes_global = []
    age_std_devs_global = []
    
    results_list = []
    
    print(f"Analyzing {len(all_csvs)} files...")
    
    for csv_file in all_csvs:
        filename = Path(csv_file).stem
        parts = filename.split('_')
        
        try:
            gt_age = int(parts[2])
            gt_gender_str = parts[3].upper()
            gt_gender = 0 if gt_gender_str == 'M' else 1
        except (IndexError, ValueError):
            print(f"Skipped (cannot parse GT): {filename}")
            continue
            
        df = pd.read_csv(csv_file)
        df_valid = df[df['age_mean'] != -1]
        
        video_frames = len(df_valid)
        
        if video_frames > 0:
            video_mae = (abs(df_valid['age_mean'] - gt_age)).sum() / video_frames
            video_age_max = df_valid['age_mean'].max()
            video_age_min = df_valid['age_mean'].min()
            video_amp = video_age_max - video_age_min
            video_std = df_valid['age_mean'].std() if video_frames > 1 else 0.0
            
            video_y_true = [gt_gender] * video_frames
            video_y_pred = df_valid['gender_mean'].tolist()
            tn, fp, fn, tp = confusion_matrix(video_y_true, video_y_pred, labels=[0, 1]).ravel()
            video_acc = (tp + tn) / video_frames
            
            results_list.append({
                "Subject": filename.replace("_out_age", ""),
                "Total_Frames": video_frames,
                "Age_MAE": round(video_mae, 2),
                "Age_Amplitude": round(video_amp, 2),
                "Age_Std_Dev": round(video_std, 2) if pd.notna(video_std) else 0.0,
                "Gender_Accuracy": round(video_acc, 4),
                "Gender_TP_Female": tp,
                "Gender_TN_Male": tn,
                "Gender_FP_Male_Predicted_Female": fp,
                "Gender_FN_Female_Predicted_Male": fn,
                "Male_Precision": "",
                "Male_Recall": "",
                "Male_F1": "",
                "Female_Precision": "",
                "Female_Recall": "",
                "Female_F1": ""
            })
            
            total_frames_global += video_frames
            age_error_sum_global += (abs(df_valid['age_mean'] - gt_age)).sum()
            y_true_gender_global.extend(video_y_true)
            y_pred_gender_global.extend(video_y_pred)
            age_amplitudes_global.append(video_amp)
            if pd.notna(video_std):
                age_std_devs_global.append(video_std)
                
    if total_frames_global > 0:
        global_mae = age_error_sum_global / total_frames_global
        global_amp = np.mean(age_amplitudes_global)
        global_std = np.mean(age_std_devs_global) if age_std_devs_global else 0.0
        
        tn_g, fp_g, fn_g, tp_g = confusion_matrix(y_true_gender_global, y_pred_gender_global, labels=[0, 1]).ravel()
        
        report = classification_report(
            y_true_gender_global, 
            y_pred_gender_global, 
            target_names=["Male", "Female"], 
            output_dict=True, 
            zero_division=0
        )
        
        results_list.append({
            "Subject": "=== GLOBAL ===",
            "Total_Frames": total_frames_global,
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
        })
        
        df_kpi = pd.DataFrame(results_list)
        df_kpi.to_csv(output_csv, index=False, sep=";")
        print(f"\nSuccess! KPIs have been exported to: {output_csv}")
        
    else:
        print("No valid frames found in the CSVs.")

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Calculate KPIs for age and gender estimation.")
    parser.add_argument("--input", required=True, help="Source folder containing the _out_age.csv files")
    parser.add_argument("--output", 
                        default="/mnt/data/OoP/data/source_code/golden_reference_creation/output/AgeGender/kpi_results.csv", 
                        help="Full path for the output CSV file")
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    compute_kpis(args.input, args.output)
