"""
KPI calculation and comparison script (Drop check) 
between Python evaluation (ONNX) and C++ inference.
"""

import argparse
import os
import glob
import pandas as pd
import numpy as np

def analyze_kpi_and_alignment(eval_dir, inf_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Search for matching files
    eval_files = {os.path.basename(f): f for f in glob.glob(os.path.join(eval_dir, "*_out_age.csv"))}
    inf_files = {os.path.basename(f): f for f in glob.glob(os.path.join(inf_dir, "*_out_age.csv"))}
    
    common_files = set(eval_files.keys()).intersection(set(inf_files.keys()))
    
    if not common_files:
        print("Error: No common files found between the two directories.")
        return
        
    print(f"Analyzing {len(common_files)} files...\n")
    
    detailed_results = []
    
    # --- Accumulateurs Globaux pour une vraie moyenne pondérée ---
    global_stats = {
        'py_valid': 0, 'py_age_err': 0.0, 'py_gen_corr': 0,
        'cpp_valid': 0, 'cpp_age_err': 0.0, 'cpp_gen_corr': 0,
        'py_mean_valid': 0, 'py_age_mean_err': 0.0, 'py_gen_mean_corr': 0,
        'cpp_mean_valid': 0, 'cpp_age_mean_err': 0.0, 'cpp_gen_mean_corr': 0,
        'align_frames': 0, 'align_age_err': 0.0, 'align_gen_corr': 0,
        'max_score_diff': 0.0
    }

    for file_name in sorted(common_files):
        df_py = pd.read_csv(eval_files[file_name])
        df_cpp = pd.read_csv(inf_files[file_name])
        
        # Merge on index
        df_cpp = df_cpp.rename(columns={
            "age": "age_cpp", "gender": "gender_cpp", "score_gender": "score_gender_cpp",
            "age_mean": "age_mean_cpp", "gender_mean": "gender_mean_cpp"
        })
        
        df_merged = pd.merge(df_py, df_cpp, on="index", how="inner")
        
        # --- KPI MASKS ---
        # Masks pour les prédictions brutes
        mask_gt_py = (df_merged['age'] != -1) & (df_merged['GT_age'] != -1)
        mask_gt_cpp = (df_merged['age_cpp'] != -1) & (df_merged['GT_age'] != -1)
        
        # Masks dédiés pour les colonnes lissées (exclut les -1 propagés par Python ou écrits par C++)
        mask_gt_py_mean = (df_merged['age_mean'] != -1) & (df_merged['GT_age'] != -1)
        mask_gt_cpp_mean = (df_merged['age_mean_cpp'] != -1) & (df_merged['GT_age'] != -1)

        # Python KPIs (Fichier local)
        py_valid_frames = mask_gt_py.sum()
        py_age_mae = (df_merged.loc[mask_gt_py, 'age'] - df_merged.loc[mask_gt_py, 'GT_age']).abs().mean() if py_valid_frames > 0 else np.nan
        py_gender_acc = (df_merged.loc[mask_gt_py, 'gender'] == df_merged.loc[mask_gt_py, 'GT_gender']).mean() * 100 if py_valid_frames > 0 else np.nan
        
        py_mean_valid = mask_gt_py_mean.sum()
        py_age_mean_mae = (df_merged.loc[mask_gt_py_mean, 'age_mean'] - df_merged.loc[mask_gt_py_mean, 'GT_age']).abs().mean() if py_mean_valid > 0 else np.nan
        py_gender_mean_acc = (df_merged.loc[mask_gt_py_mean, 'gender_mean'] == df_merged.loc[mask_gt_py_mean, 'GT_gender']).mean() * 100 if py_mean_valid > 0 else np.nan

        # C++ KPIs (Fichier local)
        cpp_valid_frames = mask_gt_cpp.sum()
        cpp_age_mae = (df_merged.loc[mask_gt_cpp, 'age_cpp'] - df_merged.loc[mask_gt_cpp, 'GT_age']).abs().mean() if cpp_valid_frames > 0 else np.nan
        cpp_gender_acc = (df_merged.loc[mask_gt_cpp, 'gender_cpp'] == df_merged.loc[mask_gt_cpp, 'GT_gender']).mean() * 100 if cpp_valid_frames > 0 else np.nan
        
        cpp_mean_valid = mask_gt_cpp_mean.sum()
        cpp_age_mean_mae = (df_merged.loc[mask_gt_cpp_mean, 'age_mean_cpp'] - df_merged.loc[mask_gt_cpp_mean, 'GT_age']).abs().mean() if cpp_mean_valid > 0 else np.nan
        cpp_gender_mean_acc = (df_merged.loc[mask_gt_cpp_mean, 'gender_mean_cpp'] == df_merged.loc[mask_gt_cpp_mean, 'GT_gender']).mean() * 100 if cpp_mean_valid > 0 else np.nan

        # Alignment (Fichier local)
        mask_align = (df_merged['age'] != -1) & (df_merged['age_cpp'] != -1)
        align_frames = mask_align.sum()
        
        if align_frames > 0:
            align_age_mae = (df_merged.loc[mask_align, 'age'] - df_merged.loc[mask_align, 'age_cpp']).abs().mean()
            align_gender_match = (df_merged.loc[mask_align, 'gender'] == df_merged.loc[mask_align, 'gender_cpp']).mean() * 100
            max_score_diff = (df_merged.loc[mask_align, 'score_gender'] - df_merged.loc[mask_align, 'score_gender_cpp']).abs().max()
        else:
            align_age_mae = np.nan
            align_gender_match = np.nan
            max_score_diff = np.nan

        # --- UPDATE DES ACCUMULATEURS GLOBAUX ---
        global_stats['py_valid'] += py_valid_frames
        global_stats['py_age_err'] += (df_merged.loc[mask_gt_py, 'age'] - df_merged.loc[mask_gt_py, 'GT_age']).abs().sum()
        global_stats['py_gen_corr'] += (df_merged.loc[mask_gt_py, 'gender'] == df_merged.loc[mask_gt_py, 'GT_gender']).sum()
        
        global_stats['cpp_valid'] += cpp_valid_frames
        global_stats['cpp_age_err'] += (df_merged.loc[mask_gt_cpp, 'age_cpp'] - df_merged.loc[mask_gt_cpp, 'GT_age']).abs().sum()
        global_stats['cpp_gen_corr'] += (df_merged.loc[mask_gt_cpp, 'gender_cpp'] == df_merged.loc[mask_gt_cpp, 'GT_gender']).sum()
        
        global_stats['py_mean_valid'] += py_mean_valid
        global_stats['py_age_mean_err'] += (df_merged.loc[mask_gt_py_mean, 'age_mean'] - df_merged.loc[mask_gt_py_mean, 'GT_age']).abs().sum()
        global_stats['py_gen_mean_corr'] += (df_merged.loc[mask_gt_py_mean, 'gender_mean'] == df_merged.loc[mask_gt_py_mean, 'GT_gender']).sum()
        
        global_stats['cpp_mean_valid'] += cpp_mean_valid
        global_stats['cpp_age_mean_err'] += (df_merged.loc[mask_gt_cpp_mean, 'age_mean_cpp'] - df_merged.loc[mask_gt_cpp_mean, 'GT_age']).abs().sum()
        global_stats['cpp_gen_mean_corr'] += (df_merged.loc[mask_gt_cpp_mean, 'gender_mean_cpp'] == df_merged.loc[mask_gt_cpp_mean, 'GT_gender']).sum()
        
        global_stats['align_frames'] += align_frames
        global_stats['align_age_err'] += (df_merged.loc[mask_align, 'age'] - df_merged.loc[mask_align, 'age_cpp']).abs().sum()
        global_stats['align_gen_corr'] += (df_merged.loc[mask_align, 'gender'] == df_merged.loc[mask_align, 'gender_cpp']).sum()
        if not np.isnan(max_score_diff):
            global_stats['max_score_diff'] = max(global_stats['max_score_diff'], max_score_diff)
        
        detailed_results.append({
            "file_name": file_name,
            "PY_valid_frames": py_valid_frames, "CPP_valid_frames": cpp_valid_frames,
            "PY_age_mae": py_age_mae, "CPP_age_mae": cpp_age_mae,
            "PY_gender_acc_%": py_gender_acc, "CPP_gender_acc_%": cpp_gender_acc,
            "PY_age_mean_mae": py_age_mean_mae, "CPP_age_mean_mae": cpp_age_mean_mae,
            "PY_gender_mean_acc_%": py_gender_mean_acc, "CPP_gender_mean_acc_%": cpp_gender_mean_acc,
            "ALIGN_frames": align_frames, "ALIGN_age_diff": align_age_mae,
            "ALIGN_gender_match_%": align_gender_match, "ALIGN_max_score_diff": max_score_diff
        })

    df_detailed = pd.DataFrame(detailed_results)
    
    # --- GLOBAL DASHBOARD DISPLAY (Valeurs Pondérées) ---
    print("="*60)
    print("        📊 GLOBAL REPORT: KPI & ALIGNMENT (DROP)")
    print("="*60)
    
    g_py_age_mae = global_stats['py_age_err'] / global_stats['py_valid'] if global_stats['py_valid'] > 0 else 0
    g_py_gen_acc = (global_stats['py_gen_corr'] / global_stats['py_valid']) * 100 if global_stats['py_valid'] > 0 else 0
    g_cpp_age_mae = global_stats['cpp_age_err'] / global_stats['cpp_valid'] if global_stats['cpp_valid'] > 0 else 0
    g_cpp_gen_acc = (global_stats['cpp_gen_corr'] / global_stats['cpp_valid']) * 100 if global_stats['cpp_valid'] > 0 else 0
    
    g_py_age_mean_mae = global_stats['py_age_mean_err'] / global_stats['py_mean_valid'] if global_stats['py_mean_valid'] > 0 else 0
    g_py_gen_mean_acc = (global_stats['py_gen_mean_corr'] / global_stats['py_mean_valid']) * 100 if global_stats['py_mean_valid'] > 0 else 0
    g_cpp_age_mean_mae = global_stats['cpp_age_mean_err'] / global_stats['cpp_mean_valid'] if global_stats['cpp_mean_valid'] > 0 else 0
    g_cpp_gen_mean_acc = (global_stats['cpp_gen_mean_corr'] / global_stats['cpp_mean_valid']) * 100 if global_stats['cpp_mean_valid'] > 0 else 0

    g_align_age = global_stats['align_age_err'] / global_stats['align_frames'] if global_stats['align_frames'] > 0 else 0
    g_align_gen = (global_stats['align_gen_corr'] / global_stats['align_frames']) * 100 if global_stats['align_frames'] > 0 else 0

    print("\n--- 🎯 1. KPI: PERFORMANCE (vs Ground Truth) ---")
    print(f"{'Metric':<25} | {'PYTHON (ONNX)':<15} | {'C++ (Inference)':<15}")
    print("-" * 60)
    print(f"{'Total Detections':<25} | {global_stats['py_valid']:<15.0f} | {global_stats['cpp_valid']:<15.0f}")
    print(f"{'Gender Accuracy':<25} | {g_py_gen_acc:<13.2f} % | {g_cpp_gen_acc:<13.2f} %")
    print(f"{'Gender Accuracy (Mean)':<25} | {g_py_gen_mean_acc:<13.2f} % | {g_cpp_gen_mean_acc:<13.2f} %")
    print(f"{'Age Error MAE':<25} | {g_py_age_mae:<11.3f} yrs | {g_cpp_age_mae:<11.3f} yrs")
    print(f"{'Age Error MAE (Mean)':<25} | {g_py_age_mean_mae:<11.3f} yrs | {g_cpp_age_mean_mae:<11.3f} yrs")

    print("\n--- ⚙️ 2. ALIGNMENT: PYTHON vs C++ (Sanity Check) ---")
    print(f"Common frames compared: {global_stats['align_frames']:.0f}")
    print(f"Exact Gender match: {g_align_gen:.2f} %")
    print(f"Average Age difference: {g_align_age:.4f} yrs")
    print(f"Max Score Divergence: {global_stats['max_score_diff']:.8f}")
    
    print("\nSTATUS:")
    if g_align_gen < 99.9 or g_align_age > 0.05:
        print("⚠️  DROP DETECTED: C++ predictions diverge from Python's.")
    elif global_stats['py_valid'] != global_stats['cpp_valid']:
        print("⚠️  DROP DETECTED: C++ pipeline does not detect the same number of faces as Python.")
    else:
        print("🟢  PERFECT ALIGNMENT: No drop. Variations (if < 1e-4) are due to float rounding.")
    print("="*60)

    # Export
    out_csv = os.path.join(output_dir, "kpi_and_alignment_summary.csv")
    df_detailed.to_csv(out_csv, index=False)
    print(f"\nDetailed report saved to: {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate KPIs and check for drop between Python and C++.")
    parser.add_argument("--eval_dir", required=True, help="Python CSV directory (with GT)")
    parser.add_argument("--inf_dir", required=True, help="C++ CSV directory")
    parser.add_argument("--output_dir", required=True, help="Directory to save the final CSV")
    args = parser.parse_args()
    
    analyze_kpi_and_alignment(args.eval_dir, args.inf_dir, args.output_dir)