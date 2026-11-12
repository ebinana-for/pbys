import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

def compute_kpis(csv_dir, annotation_csv_path, output_csv="kpi_results.csv"):
    
    # --- NOUVEAU : Chargement de la vérité terrain (Ground Truth) ---
    print(f"Loading Ground Truth from: {annotation_csv_path}")
    try:
        # header=2 car la vraie ligne de colonnes est la 3ème ligne du fichier
        df_gt = pd.read_csv(annotation_csv_path, encoding='iso-8859-1', sep=';', header=2)
        
        # Création d'un dictionnaire { "YL_S03_1": {"Age": 33, "Sex": 0} }
        gt_dict = {}
        for index, row in df_gt.dropna(subset=['Video ID']).iterrows():
            video_name = str(row['Video ID']).replace('.mp4', '')
            try:
                age = int(row['Age'])
                # Conversion du sexe ('M' = 0, 'F' = 1)
                sex_str = str(row['Sex']).strip().upper()
                gender = 0 if sex_str == 'M' else (1 if sex_str == 'F' else -1)
                
                if gender != -1:
                    gt_dict[video_name] = {"Age": age, "Sex": gender}
            except ValueError:
                continue # Ignore les lignes sans âge valide
    except Exception as e:
        print(f"Error loading annotations.csv: {e}")
        return

    # ---------------------------------------------------------------
    
    all_csvs = glob.glob(os.path.join(csv_dir, "*_out_age.csv"))
    
    # Initialisation des variables globales
    total_frames_global = 0
    age_error_sum_global = 0
    y_true_gender_global = []
    y_pred_gender_global = []
    age_amplitudes_global = []
    age_std_devs_global = []
    results_list = []
    
    print(f"Analyzing {len(all_csvs)} prediction files...")
    
    for csv_file in all_csvs:
        filename = Path(csv_file).stem # ex: YL_S03_1_out_age
        
        # On retire "_out_age" pour matcher avec la clé du dictionnaire "YL_S03_1"
        base_video_name = filename.replace("_out_age", "")
        
        # --- NOUVEAU : Récupération depuis le dictionnaire ---
        if base_video_name not in gt_dict:
            print(f"Skipped (No Ground Truth found in annotations.csv): {base_video_name}")
            continue
            
        gt_age = gt_dict[base_video_name]["Age"]
        gt_gender = gt_dict[base_video_name]["Sex"]
        # -----------------------------------------------------

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
                
    if len(results_list) > 0:
        # Calcul de la moyenne PAR SUJET
        global_mae = np.mean([res["Age_MAE"] for res in results_list])
        global_amp = np.mean([res["Age_Amplitude"] for res in results_list])
        global_std = np.mean([res["Age_Std_Dev"] for res in results_list])
        
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
    
    # NOUVEAU PARAMETRE POUR LES ANNOTATIONS
    parser.add_argument("--annotations", required=True, help="Path to the annotations.csv file")
    
    parser.add_argument("--output", 
                        default="/mnt/data/OoP/data/source_code/golden_reference_creation/output/AgeGender/kpi_results.csv", 
                        help="Full path for the output CSV file")
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    compute_kpis(args.input, args.annotations, args.output)