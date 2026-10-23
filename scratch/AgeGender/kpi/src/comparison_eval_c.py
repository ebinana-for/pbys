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

    for file_name in sorted(common_files):
        df_py = pd.read_csv(eval_files[file_name])
        df_cpp = pd.read_csv(inf_files[file_name])
        
        # Merge on index (Python contains GT, we keep it)
        # Rename columns to avoid confusion
        df_cpp = df_cpp.rename(columns={
            "age": "age_cpp", "gender": "gender_cpp", "score_gender": "score_gender_cpp",
            "age_mean": "age_mean_cpp", "gender_mean": "gender_mean_cpp"
        })
        
        df_merged = pd.merge(df_py, df_cpp, on="index", how="inner")
        
        # --- KPI CALCULATION (vs Ground Truth) ---
        # Masks to isolate valid detections with a real GT
        mask_gt_py = (df_merged['age'] != -1) & (df_merged['GT_age'] != -1)
        mask_gt_cpp = (df_merged['age_cpp'] != -1) & (df_merged['GT_age'] != -1)
        
        # Python KPIs
        py_valid_frames = mask_gt_py.sum()
        py_age_mae = (df_merged.loc[mask_gt_py, 'age'] - df_merged.loc[mask_gt_py, 'GT_age']).abs().mean()
        py_gender_acc = (df_merged.loc[mask_gt_py, 'gender'] == df_merged.loc[mask_gt_py, 'GT_gender']).mean() * 100
        
        py_age_mean_mae = (df_merged.loc[mask_gt_py, 'age_mean'] - df_merged.loc[mask_gt_py, 'GT_age']).abs().mean()
        py_gender_mean_acc = (df_merged.loc[mask_gt_py, 'gender_mean'] == df_merged.loc[mask_gt_py, 'GT_gender']).mean() * 100

        # C++ KPIs
        cpp_valid_frames = mask_gt_cpp.sum()
        cpp_age_mae = (df_merged.loc[mask_gt_cpp, 'age_cpp'] - df_merged.loc[mask_gt_cpp, 'GT_age']).abs().mean()
        cpp_gender_acc = (df_merged.loc[mask_gt_cpp, 'gender_cpp'] == df_merged.loc[mask_gt_cpp, 'GT_gender']).mean() * 100
        
        cpp_age_mean_mae = (df_merged.loc[mask_gt_cpp, 'age_mean_cpp'] - df_merged.loc[mask_gt_cpp, 'GT_age']).abs().mean()
        cpp_gender_mean_acc = (df_merged.loc[mask_gt_cpp, 'gender_mean_cpp'] == df_merged.loc[mask_gt_cpp, 'GT_gender']).mean() * 100

        # --- ALIGNMENT CALCULATION (Python vs C++) ---
        mask_align = (df_merged['age'] != -1) & (df_merged['age_cpp'] != -1)
        align_frames = mask_align.sum()
        
        if align_frames > 0:
            align_age_mae = (df_merged.loc[mask_align, 'age'] - df_merged.loc[mask_align, 'age_cpp']).abs().mean()
            align_gender_match = (df_merged.loc[mask_align, 'gender'] == df_merged.loc[mask_align, 'gender_cpp']).mean() * 100
            max_score_diff = (df_merged.loc[mask_align, 'score_gender'] - df_merged.loc[mask_align, 'score_gender_cpp']).abs().max()
        else:
            align_age_mae = 0
            align_gender_match = 0
            max_score_diff = 0
        
        # Save report for this file (ALTERNATING PY and CPP for easier comparison)
        detailed_results.append({
            "file_name": file_name,
            
            "PY_valid_frames": py_valid_frames,
            "CPP_valid_frames": cpp_valid_frames,
            
            "PY_age_mae": py_age_mae,
            "CPP_age_mae": cpp_age_mae,
            
            "PY_gender_acc_%": py_gender_acc,
            "CPP_gender_acc_%": cpp_gender_acc,
            
            "PY_age_mean_mae": py_age_mean_mae,
            "CPP_age_mean_mae": cpp_age_mean_mae,
            
            "PY_gender_mean_acc_%": py_gender_mean_acc,
            "CPP_gender_mean_acc_%": cpp_gender_mean_acc,
            
            "ALIGN_frames": align_frames,
            "ALIGN_age_diff": align_age_mae,
            "ALIGN_gender_match_%": align_gender_match,
            "ALIGN_max_score_diff": max_score_diff
        })

    df_detailed = pd.DataFrame(detailed_results)
    
    # --- GLOBAL DASHBOARD DISPLAY ---
    print("="*60)
    print("        📊 GLOBAL REPORT: KPI & ALIGNMENT (DROP)")
    print("="*60)
    
    # Global weighted averages
    print("\n--- 🎯 1. KPI: PERFORMANCE (vs Ground Truth) ---")
    print(f"{'Metric':<25} | {'PYTHON (ONNX)':<15} | {'C++ (Inference)':<15}")
    print("-" * 60)
    print(f"{'Total Detections':<25} | {df_detailed['PY_valid_frames'].sum():<15.0f} | {df_detailed['CPP_valid_frames'].sum():<15.0f}")
    print(f"{'Gender Accuracy':<25} | {df_detailed['PY_gender_acc_%'].mean():<13.2f} % | {df_detailed['CPP_gender_acc_%'].mean():<13.2f} %")
    print(f"{'Gender Accuracy (Mean)':<25} | {df_detailed['PY_gender_mean_acc_%'].mean():<13.2f} % | {df_detailed['CPP_gender_mean_acc_%'].mean():<13.2f} %")
    print(f"{'Age Error MAE':<25} | {df_detailed['PY_age_mae'].mean():<11.3f} yrs | {df_detailed['CPP_age_mae'].mean():<11.3f} yrs")
    print(f"{'Age Error MAE (Mean)':<25} | {df_detailed['PY_age_mean_mae'].mean():<11.3f} yrs | {df_detailed['CPP_age_mean_mae'].mean():<11.3f} yrs")

    print("\n--- ⚙️ 2. ALIGNMENT: PYTHON vs C++ (Sanity Check) ---")
    
    mean_align_gender = df_detailed['ALIGN_gender_match_%'].mean()
    mean_align_age = df_detailed['ALIGN_age_diff'].mean()
    global_max_score_diff = df_detailed['ALIGN_max_score_diff'].max()
    
    print(f"Common frames compared: {df_detailed['ALIGN_frames'].sum():.0f}")
    print(f"Exact Gender match: {mean_align_gender:.2f} %")
    print(f"Average Age difference: {mean_align_age:.4f} yrs")
    print(f"Max Score Divergence: {global_max_score_diff:.8f}")
    
    print("\nSTATUS:")
    if mean_align_gender < 99.9 or mean_align_age > 0.05:
        print("⚠️  DROP DETECTED: C++ predictions diverge from Python's.")
    elif df_detailed['PY_valid_frames'].sum() != df_detailed['CPP_valid_frames'].sum():
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
