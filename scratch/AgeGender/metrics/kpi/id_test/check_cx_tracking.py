import os
import glob
import pandas as pd
import numpy as np
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# All code and comments are in English

def calculate_cpp_face_bbox_center(tracking_dir, output_plot="cpp_face_bbox_cx_distribution.png"):
    """
    Analyzes tracking CSV files to calculate the Face Bounding Box center (X-axis) 
    using the exact mathematical formula and constraints from the C++ inference code.
    """
    csv_files = glob.glob(os.path.join(tracking_dir, "*.csv"))
    if not csv_files:
        print(f"[Error] No CSV files found in {tracking_dir}")
        return

    print(f"[Info] Analyzing {len(csv_files)} files using C++ Face BBox logic...")

    face_cx_0_list = []
    face_cx_1_list = []

    # SmartEye Keypoint Indices mapping based on foms_utils
    idx_leye = 1
    idx_reye = 2
    idx_lear = 3
    idx_rear = 4
    idx_chin = 25

    for file in csv_files:
        try:
            df = pd.read_csv(file)
        except pd.errors.EmptyDataError:
            continue

        for occupant_id, cx_list in [(0, face_cx_0_list), (1, face_cx_1_list)]:
            
            # Construct column names for qualities
            q_leye = f'/trackingState/occupants/{occupant_id}/bodyKeyPoints/{idx_leye}/quality'
            q_reye = f'/trackingState/occupants/{occupant_id}/bodyKeyPoints/{idx_reye}/quality'
            q_lear = f'/trackingState/occupants/{occupant_id}/bodyKeyPoints/{idx_lear}/quality'
            q_rear = f'/trackingState/occupants/{occupant_id}/bodyKeyPoints/{idx_rear}/quality'
            q_chin = f'/trackingState/occupants/{occupant_id}/bodyKeyPoints/{idx_chin}/quality'
            
            # Construct column names for X coordinates
            x_lear = f'/trackingState/occupants/{occupant_id}/bodyKeyPoints/{idx_lear}/value/0'
            x_rear = f'/trackingState/occupants/{occupant_id}/bodyKeyPoints/{idx_rear}/value/0'

            required_cols = [q_leye, q_reye, q_lear, q_rear, q_chin, x_lear, x_rear]

            if all(col in df.columns for col in required_cols):
                
                # C++ Constraint 1: All 5 keypoints must be valid (quality > minBodyPtQuality, assumed 0.1)
                valid_mask = (
                    (df[q_leye] > 0.1) & 
                    (df[q_reye] > 0.1) & 
                    (df[q_lear] > 0.1) & 
                    (df[q_rear] > 0.1) & 
                    (df[q_chin] > 0.1)
                )
                
                valid_frames = df[valid_mask]
                
                if not valid_frames.empty:
                    # C++ Logic: const double center_x = (REarkeyPt.x + LEarkeyPt.x) / 2.0;
                    face_cx = (valid_frames[x_lear] + valid_frames[x_rear]) / 2.0
                    cx_list.extend(face_cx.tolist())

    # Convert to NumPy arrays
    cx_0_arr = np.array(face_cx_0_list)
    cx_1_arr = np.array(face_cx_1_list)

    print("\n[Statistics] --- C++ Face BBox Center X (Based on Ears) ---")
    if len(cx_0_arr) > 0:
        print(f"ID 0 : {len(cx_0_arr)} valid C++ crops | Min: {cx_0_arr.min():.1f} | Max: {cx_0_arr.max():.1f} | Mean: {cx_0_arr.mean():.1f}")
    else:
        print("ID 0 : No valid C++ crops found.")

    if len(cx_1_arr) > 0:
        print(f"ID 1 : {len(cx_1_arr)} valid C++ crops | Min: {cx_1_arr.min():.1f} | Max: {cx_1_arr.max():.1f} | Mean: {cx_1_arr.mean():.1f}")
    else:
        print("ID 1 : No valid C++ crops found.")

    # Generate Histogram
    plt.figure(figsize=(10, 6))
    
    if len(cx_0_arr) > 0:
        plt.hist(cx_0_arr, bins=60, alpha=0.6, label='ID 0 (C++ Face BBox Center)', color='blue')
    if len(cx_1_arr) > 0:
        plt.hist(cx_1_arr, bins=60, alpha=0.6, label='ID 1 (C++ Face BBox Center)', color='orange')

    plt.title("Spatial Distribution of C++ Face BBox Centers (X-Axis)")
    plt.xlabel("Face Bounding Box Center X Position (Pixels)")
    plt.ylabel("Valid Frame Count")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    plt.savefig(output_plot, dpi=150)
    plt.close()
    
    print(f"\n[Output] Graph successfully saved as: {output_plot}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate occupant IDs based on C++ Face BBox logic.")
    parser.add_argument("--tracking_dir", type=str, required=True, help="Path to the directory containing tracking CSV files.")
    args = parser.parse_args()
    
    calculate_cpp_face_bbox_center(args.tracking_dir)