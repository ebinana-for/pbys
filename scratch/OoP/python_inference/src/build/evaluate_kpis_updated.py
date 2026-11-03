import os
import glob
import numpy as np
import pandas as pd
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, average_precision_score

pd.set_option('future.no_silent_downcasting', True)

CPP_CLASS_ORDER = [
    "closetoairbag",
    "legsondashboard",
    "torsodissimetry",
    "reachingback",
    "lateralbodycontact",
    "oneshoulderincontact",
    "pelvisrotation",
    "pelvisdrift",
]

# "correctly seated" class: GT = none of the 8 classes active, prediction = no class >= threshold
CS = "correctly seated"
ALL_CLASSES = CPP_CLASS_ORDER + [CS]

VIDEO_COLS = ["Sequence", "Subject", "Class", "Support", "TP", "TN", "FP", "FN",
              "Precision", "Recall", "F1_Score", "Mean_Prob", "Max_Prob", "Total_Frames"]

# UPDATED: Integration of Absolute GT and End-to-End metrics
SUMMARY_COLS = ["Class", "Absolute_GT_Frames", "Tracked_GT_Frames", "Lost_Track_Frames", 
                "Support_Videos", "TP", "TN", "FP", "FN",
                "Precision", "Model_Recall", "End_to_End_Recall", "F1_Score", 
                "Specificity", "Accuracy", "FP_per_1000", "AP", "Total_Frames"]

SUBJECT_COLS = ["Subject", "Class", "Support", "TP", "TN", "FP", "FN",
                "Precision", "Recall", "F1_Score", "Total_Frames"]


def _div(a, b):
    return round(a / b, 4) if b > 0 else np.nan


def _get_arrays(df, cls, threshold):
    if cls == CS:
        cols = [c for c in CPP_CLASS_ORDER if f"{c}_gt" in df.columns and c in df.columns]
        if not cols: return None
        gt_cols = [f"{c}_gt" for c in cols]
        valid = (df[gt_cols] != -1).all(axis=1) & (df[cols] != -1.0).all(axis=1)
        if not valid.any(): return None
        gt = (df.loc[valid, gt_cols] == 0).all(axis=1).astype(int).to_numpy()
        prob = df.loc[valid, cols].max(axis=1).to_numpy()
        y_pred = (prob < threshold).astype(int)
        return gt, prob, 1.0 - prob, y_pred

    col_gt = f"{cls}_gt"
    if col_gt not in df.columns or cls not in df.columns: return None
    valid = (df[col_gt] != -1) & (df[cls] != -1.0)
    if not valid.any(): return None
    gt = df.loc[valid, col_gt].astype(int).to_numpy()
    prob = df.loc[valid, cls].to_numpy()
    y_pred = (prob >= threshold).astype(int)
    return gt, prob, prob, y_pred


def _counts_and_metrics(gt, y_pred):
    tn, fp, fn, tp = confusion_matrix(gt, y_pred, labels=[0, 1]).ravel()
    total = int(tn + fp + fn + tp)
    return {
        "Support": int(tp + fn), "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "Precision": _div(tp, tp + fp), "Recall": _div(tp, tp + fn),
        "F1_Score": _div(2 * tp, 2 * tp + fp + fn), "Specificity": _div(tn, tn + fp),
        "Accuracy": _div(tp + tn, total),
        "FP_per_1000": round(1000 * fp / total, 2) if total > 0 else np.nan,
        "Total_Frames": total,
    }


def _invalid_row(**ident):
    row = dict(ident)
    for k in ["Support", "Support_Frames", "Absolute_GT_Frames", "Tracked_GT_Frames", 
              "Lost_Track_Frames", "Support_Videos", "TP", "TN", "FP", "FN",
              "Precision", "Recall", "Model_Recall", "End_to_End_Recall", "F1_Score", 
              "Specificity", "Accuracy", "FP_per_1000", "AP", "Mean_Prob", "Max_Prob", "Total_Frames"]:
        row[k] = -1
    return row


def _video_rows(df_merged, sequence_name, subject, threshold):
    rows = []
    for cls in ALL_CLASSES:
        arr = _get_arrays(df_merged, cls, threshold)
        if arr is None:
            rows.append(_invalid_row(Sequence=sequence_name, Subject=subject, Class=cls))
            continue
        gt, prob, _, y_pred = arr
        row = {"Sequence": sequence_name, "Subject": subject, "Class": cls}
        row.update(_counts_and_metrics(gt, y_pred))
        row["Mean_Prob"] = round(float(prob.mean()), 4)
        row["Max_Prob"] = round(float(prob.max()), 4)
        rows.append(row)
    return rows


def _summary_rows(df_all, video_rows, threshold):
    df_vid = pd.DataFrame(video_rows)
    rows = []
    for cls in ALL_CLASSES:
        
        # 1. CALCUL DE LA VÉRITÉ TERRAIN ABSOLUE (Avant filtrage des pertes de tracking)
        absolute_gt_frames = 0
        if cls == CS:
            cols = [c for c in CPP_CLASS_ORDER if f"{c}_gt" in df_all.columns and c in df_all.columns]
            if cols:
                gt_cols = [f"{c}_gt" for c in cols]
                valid_gt_only = (df_all[gt_cols] != -1).all(axis=1) # On ignore juste les annotations invalides
                absolute_gt_frames = (df_all.loc[valid_gt_only, gt_cols] == 0).all(axis=1).sum()
        else:
            col_gt = f"{cls}_gt"
            if col_gt in df_all.columns:
                valid_gt_only = df_all[col_gt] != -1
                absolute_gt_frames = (df_all.loc[valid_gt_only, col_gt] == 1).sum()

        # 2. RÉCUPÉRATION DES ARRAYS FILTRÉS
        arr = _get_arrays(df_all, cls, threshold)
        if arr is None:
            rows.append(_invalid_row(Class=cls))
            continue
            
        gt, _, score, y_pred = arr
        m = _counts_and_metrics(gt, y_pred)
        sub = df_vid[(df_vid["Class"] == cls) & (df_vid["Support"] > 0)]
        ap = round(float(average_precision_score(gt, score)), 4) if 0 < gt.sum() < len(gt) else np.nan
        
        # 3. DÉDUCTIONS DES NOUVELLES MÉTRIQUES
        tracked_gt_frames = m["Support"]
        lost_track_frames = absolute_gt_frames - tracked_gt_frames
        e2e_recall = _div(m["TP"], absolute_gt_frames)
        
        rows.append({
            "Class": cls, 
            "Absolute_GT_Frames": absolute_gt_frames,
            "Tracked_GT_Frames": tracked_gt_frames,
            "Lost_Track_Frames": lost_track_frames,
            "Support_Videos": int(len(sub)),
            "TP": m["TP"], "TN": m["TN"], "FP": m["FP"], "FN": m["FN"],
            "Precision": m["Precision"], 
            "Model_Recall": m["Recall"],
            "End_to_End_Recall": e2e_recall,
            "F1_Score": m["F1_Score"],
            "Specificity": m["Specificity"], "Accuracy": m["Accuracy"],
            "FP_per_1000": m["FP_per_1000"], "AP": ap, "Total_Frames": m["Total_Frames"],
        })

    df_s = pd.DataFrame(rows)
    pos = df_s[df_s["Tracked_GT_Frames"] > 0]
    macro = {"Class": "MACRO (classes with positives)"}
    for k in ["Precision", "Model_Recall", "End_to_End_Recall", "F1_Score", "AP"]:
        macro[k] = round(float(pos[k].mean()), 4) if len(pos) and pos[k].notna().any() else np.nan
    rows.append(macro)
    return rows


def _subject_rows(df_all, threshold, classes_with_support):
    rows = []
    for subject, df_sub in df_all.groupby("subject"):
        for cls in classes_with_support:
            arr = _get_arrays(df_sub, cls, threshold)
            if arr is None: continue
            gt, _, _, y_pred = arr
            row = {"Subject": subject, "Class": cls}
            row.update(_counts_and_metrics(gt, y_pred))
            rows.append(row)
    return rows


def _make_dashboard(df_sum, df_subj, df_vid, out_path, threshold):
    df_cls = df_sum[df_sum["Class"].isin(ALL_CLASSES)]
    cls_pos = df_cls[df_cls["Tracked_GT_Frames"] > 0]

    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    fig.suptitle(f"Inference Evaluation (threshold = {threshold})", fontsize=16, fontweight="bold")

    ax = axes[0]
    if len(cls_pos):
        x = np.arange(len(cls_pos))
        w = 0.26
        # Mise à jour des libellés avec Model_Recall
        for i, (k, c) in enumerate([("Precision", "#4C72B0"), ("Model_Recall", "#55A868"), ("F1_Score", "#C44E52")]):
            ax.bar(x + (i - 1) * w, cls_pos[k].astype(float).fillna(0), w, label=k, color=c)
        if cls_pos["AP"].notna().any():
            ax.scatter(x, cls_pos["AP"].astype(float), marker="D", color="black", zorder=5, label="AP (indicative)")
        ax.set_xticks(x)
        # Affichage du Tracked_GT_Frames sur l'axe X
        ax.set_xticklabels([f"{c}\n{int(f)} frames / {int(v)} videos" for c, f, v in zip(cls_pos["Class"], cls_pos["Tracked_GT_Frames"], cls_pos["Support_Videos"])], fontsize=8)
        ax.set_ylim(0, 1.2)
        ax.legend(fontsize=8, ncol=4, loc="upper center")
    ax.set_title("Performance per class (classes with positives in GT)")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    if len(df_subj):
        piv = df_subj.pivot_table(index="Subject", columns="Class", values="F1_Score", aggfunc="first")
        piv = piv.reindex(columns=[c for c in ALL_CLASSES if c in piv.columns])
        data = np.ma.masked_invalid(piv.to_numpy(dtype=float))
        cmap = plt.get_cmap("RdYlGn").copy()
        cmap.set_bad("#DDDDDD")
        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels(piv.columns, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(piv.shape[0]))
        ax.set_yticklabels(piv.index, fontsize=8)
        if piv.shape[0] <= 40:
            for i in range(piv.shape[0]):
                for j in range(piv.shape[1]):
                    if not np.ma.is_masked(data[i, j]): ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.03)
    ax.set_title("F1 per subject (gray = undefined)")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Calculates KPIs from predictions and GT.")
    parser.add_argument("--predictions_dir", type=str, default="S03/mp4/predictions", help="Folder containing prediction CSVs.")
    parser.add_argument("--gt_csv", type=str, default="ground_truth_all_videos.csv", help="Generated ground truth CSV file.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold.")
    parser.add_argument("--full_detail", action="store_true", help="Keep ALL video x class rows in KPI_per_video.csv.")
    args = parser.parse_args()

    print(f"📊 Loading ground truth: {args.gt_csv}")
    df_gt_master = pd.read_csv(args.gt_csv)
    
    pred_files = glob.glob(os.path.join(args.predictions_dir, "*_predictions.csv"))
    if not pred_files:
        print(f"❌ No predictions found in: {args.predictions_dir}")
        return
        
    print(f"✅ {len(pred_files)} prediction files found. Detecting occupants and applying max-pooling...")

    merged_dfs_by_occ = {}

    for pred_path in pred_files:
        filename = os.path.basename(pred_path)
        video_base_name = filename.replace("_predictions.csv", "")
        video_mp4_name = f"{video_base_name}.mp4"

        row_gt = df_gt_master[df_gt_master['Video ID'] == video_mp4_name]
        if row_gt.empty:
            continue
        gt_dict = row_gt[[f"{c}_gt" for c in CPP_CLASS_ORDER]].iloc[0].to_dict()

        try:
            df_pred = pd.read_csv(pred_path)
        except pd.errors.EmptyDataError:
            continue
        
        if "occupant_id" not in df_pred.columns:
            df_pred["occupant_id"] = 0

        for occ_id in df_pred["occupant_id"].unique():
            df_occ = df_pred[df_pred["occupant_id"] == occ_id].copy()
            
            has_valid_predictions = (df_occ[CPP_CLASS_ORDER] != -1.0).any(axis=None)
            if not has_valid_predictions:
                continue 

            df_gt_frames = pd.DataFrame([gt_dict] * len(df_occ))
            df_gt_frames['index'] = df_occ['index'].values
            
            df_merged = pd.merge(df_occ, df_gt_frames, on="index", how="inner")
            df_merged["subject"] = video_base_name.split("_")[0]
            
            if occ_id not in merged_dfs_by_occ:
                merged_dfs_by_occ[occ_id] = {}
            merged_dfs_by_occ[occ_id][video_base_name] = df_merged

        df_pred_replaced = df_pred.replace(-1.0, np.nan)
        
        df_pooled = df_pred_replaced.groupby('index')[CPP_CLASS_ORDER].max().reset_index()
        
        df_pooled = df_pooled.fillna(-1.0)
        
        has_valid_pooled = (df_pooled[CPP_CLASS_ORDER] != -1.0).any(axis=None)
        if has_valid_pooled:
            df_gt_frames_pooled = pd.DataFrame([gt_dict] * len(df_pooled))
            df_gt_frames_pooled['index'] = df_pooled['index'].values
            
            df_merged_best = pd.merge(df_pooled, df_gt_frames_pooled, on="index", how="inner")
            df_merged_best["subject"] = video_base_name.split("_")[0]
            
            if "best" not in merged_dfs_by_occ:
                merged_dfs_by_occ["best"] = {}
            merged_dfs_by_occ["best"][video_base_name] = df_merged_best

    if not merged_dfs_by_occ:
        print("❌ Cannot calculate KPIs: no valid predictions found (only -1s or no GT matches).")
        return

    for occ_id, merged_dfs_dict in merged_dfs_by_occ.items():
        print(f"\n=======================================================")
        print(f"⚙️ GENERATING RESULTS FOR OCCUPANT {occ_id}")
        print(f"=======================================================")
        
        video_rows = []
        for seq_name, df_merged in merged_dfs_dict.items():
            video_rows.extend(_video_rows(df_merged, seq_name, df_merged["subject"].iloc[0], args.threshold))

        global_df = pd.concat(list(merged_dfs_dict.values()), ignore_index=True)
        summary_rows = _summary_rows(global_df, video_rows, args.threshold)

        df_vid_full = pd.DataFrame(video_rows)
        order = {c: i for i, c in enumerate(ALL_CLASSES)}
        df_vid_full = df_vid_full.assign(_o=df_vid_full["Class"].map(order)).sort_values(["Sequence", "_o"]).drop(columns="_o")
        df_vid_full = df_vid_full[VIDEO_COLS].reset_index(drop=True)

        if args.full_detail:
            df_vid_out = df_vid_full
        else:
            keep = (df_vid_full["Support"] > 0) | (df_vid_full["FP"] > 0) | (df_vid_full["Support"] == -1)
            df_vid_out = df_vid_full[keep].reset_index(drop=True)

        df_sum = pd.DataFrame(summary_rows)[SUMMARY_COLS]
        for c in ["Absolute_GT_Frames", "Tracked_GT_Frames", "Lost_Track_Frames", "Support_Videos", "TP", "TN", "FP", "FN", "Total_Frames"]:
            df_sum[c] = df_sum[c].astype("Int64")

        classes_with_support = df_sum[(df_sum["Tracked_GT_Frames"] > 0) & (df_sum["Class"].isin(ALL_CLASSES))]["Class"].tolist()
        df_subj = pd.DataFrame(_subject_rows(global_df, args.threshold, classes_with_support), columns=SUBJECT_COLS)

        out_dir = os.path.join(args.predictions_dir, f"kpi_occupant_{occ_id}")
        os.makedirs(out_dir, exist_ok=True)
        
        path_sum = os.path.join(out_dir, "KPI_summary.csv")
        path_vid = os.path.join(out_dir, "KPI_per_video.csv")
        path_subj = os.path.join(out_dir, "KPI_per_subject.csv")
        path_png = os.path.join(out_dir, "KPI_dashboard.png")

        df_sum.to_csv(path_sum, index=False, na_rep="N/A")
        df_vid_out.to_csv(path_vid, index=False, na_rep="N/A")
        df_subj.to_csv(path_subj, index=False, na_rep="N/A")
        _make_dashboard(df_sum, df_subj, df_vid_full, path_png, args.threshold)

        print(f"✅ Folder exported: {out_dir}")
        # Affichage enrichi dans le terminal
        print(df_sum[["Class", "Absolute_GT", "Tracked_GT", "Lost_Track", "TP", "FP", "FN", "Precision", "Model_Recall", "E2E_Recall", "F1_Score"]].rename(columns={"Absolute_GT_Frames": "Absolute_GT", "Tracked_GT_Frames": "Tracked_GT", "Lost_Track_Frames": "Lost_Track", "End_to_End_Recall": "E2E_Recall"}).to_string(index=False, na_rep="N/A"))

    print(f"\n🎉 Evaluation of all occupants and the vehicle ('best') is complete!")

if __name__ == "__main__":
    main()