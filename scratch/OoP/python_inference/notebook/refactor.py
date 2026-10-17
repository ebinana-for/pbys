import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
from IPython.display import display

# ==========================================
# 1. CONFIGURATION
# ==========================================

COLUMN_MAPPING = {
    "tooclosetoairbag": "closetoairbag",
}

THRESHOLD = 0.5

PATH_GT = "/mnt/data/OoP/data/source_code/python_inference/output/results_eval_inference/letterboxing/eval_inference_python_20_08_with_keypoints_draw_with_keypoints_onnx/thibault_vincent_28_M_178_68_eval_with_gt.csv"


# ==========================================
# 2. UTILS
# ==========================================

def clean_column_name(col):
    c = str(col).lower().strip()

    is_gt = c.endswith("_gt") or c.endswith("gt")
    if is_gt:
        c = c.replace("_gt", "").replace("gt", "")

    c = c.replace(" ", "").replace("_", "")

    if c in COLUMN_MAPPING:
        c = COLUMN_MAPPING[c]

    return f"{c}_gt" if is_gt else c


def load_and_prepare(path, suffix, has_header=True, ref_cols=None):
    if has_header:
        df = pd.read_csv(path)
        df.columns = [clean_column_name(c) if c != "index" else "index" for c in df.columns]
        if "index" not in df.columns and len(df.columns) > 0:
            df.rename(columns={df.columns[0]: "index"}, inplace=True)
    else:
        df = pd.read_csv(path, header=None)
        df.columns = ref_cols

    rename_dict = {}
    for col in df.columns:
        if col != "index" and not str(col).endswith("_gt"):
            rename_dict[col] = f"{col}_{suffix}"
    df.rename(columns=rename_dict, inplace=True)
    return df


def metric_column_name(model_name, metric_key):
    return f"{model_name}_{metric_key}"


def display_metric_table(df, metric_key, title, model_names):
    print(f"\n{title}")
    cols_extract = ["Class"] + [metric_column_name(m, metric_key) for m in model_names]
    df_display = df[cols_extract].copy()
    df_display.columns = ["Class"] + model_names
    display(df_display)


# ==========================================
# 3. DATA LOADING
# ==========================================

def load_merged_dataset(path_gt=PATH_GT, models_config=None):
    if models_config is None:
        models_config = MODELS_CONFIG

    df_gt_full = load_and_prepare(path_gt, "ignore")
    gt_cols_to_keep = [c for c in df_gt_full.columns if c == "index" or str(c).endswith("_gt")]
    df_merged = df_gt_full[gt_cols_to_keep]

    for model_name, config in models_config.items():
        df_model = load_and_prepare(config["path"], config["suffix"])
        cols_to_keep = [c for c in df_model.columns if not str(c).endswith("_gt")]
        df_merged = pd.merge(df_merged, df_model[cols_to_keep], on="index", how="inner")

    first_pred_col = [c for c in df_merged.columns if c != "index" and not str(c).endswith("_gt")][0]
    df_merged = df_merged.dropna(subset=[first_pred_col])

    print(f"✅ Successfully merged {len(models_config)} data sources WITH the Ground Truth. Total valid frames: {len(df_merged)}")
    return df_merged


# ==========================================
# 4. KPI
# ==========================================

def compute_kpi_by_class(df_merged, models_config=None, threshold=THRESHOLD):
    if models_config is None:
        models_config = MODELS_CONFIG

    available_classes = [c.replace("_gt", "") for c in df_merged.columns if str(c).endswith("_gt")]
    kpi_data = []

    if not available_classes:
        print("⚠️ Error: No ground truth classes found. Check the column cleaning logic.")
        return pd.DataFrame()

    for cls in available_classes:
        col_gt = f"{cls}_gt"
        if col_gt not in df_merged.columns:
            continue

        row_data = {"Class": cls}
        gt_full = df_merged[col_gt]

        for model_name, config in models_config.items():
            col_model = f"{cls}_{config['suffix']}"

            if col_model in df_merged.columns:
                model_full = df_merged[col_model]

                valid_mask = (gt_full != -1.0) & (model_full != -1.0)
                gt_valid = gt_full[valid_mask]
                model_probs = model_full[valid_mask]

                if len(gt_valid) > 0:
                    preds = (model_probs >= threshold).astype(int)

                    row_data[metric_column_name(model_name, "Prec")] = round(precision_score(gt_valid, preds, zero_division=0), 3)
                    row_data[metric_column_name(model_name, "Rec")] = round(recall_score(gt_valid, preds, zero_division=0), 3)
                    row_data[metric_column_name(model_name, "F1")] = round(f1_score(gt_valid, preds, zero_division=0), 3)

                    row_data[metric_column_name(model_name, "TP")] = ((preds == 1) & (gt_valid == 1)).sum()
                    row_data[metric_column_name(model_name, "FP")] = ((preds == 1) & (gt_valid == 0)).sum()
                    row_data[metric_column_name(model_name, "FN")] = ((preds == 0) & (gt_valid == 1)).sum()
                else:
                    for metric in ["Prec", "Rec", "F1", "TP", "FP", "FN"]:
                        row_data[metric_column_name(model_name, metric)] = None

        kpi_data.append(row_data)

    kpi_df = pd.DataFrame(kpi_data)
    return kpi_df


def plot_kpi_charts(kpi_df, model_names, threshold=THRESHOLD):
    print(f"📊 GENERATING {len(model_names)}-WAY METRIC COMPARISON CHARTS...\n")

    if kpi_df.empty:
        print("⚠️ Error: 'kpi_df' is empty.")
        return

    if not model_names:
        print("⚠️ Error: 'model_names' is empty.")
        return

    model_names = list(dict.fromkeys(model_names))
    classes = kpi_df["Class"].tolist()
    x = np.arange(len(classes))
    n_models = len(model_names)

    metrics_to_plot = [
        ("Precision", "Prec"),
        ("Recall", "Rec"),
        ("F1-Score", "F1"),
        ("False Positives (FP)", "FP"),
        ("False Negatives (FN)", "FN"),
        ("True Positives (TP)", "TP"),
    ]

    n_cols = 3 if len(metrics_to_plot) > 1 else 1
    n_rows = (len(metrics_to_plot) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 6 * n_rows), squeeze=False)
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, n_models))
    bar_width = 0.8 / max(1, n_models)

    for i, (title, suffix) in enumerate(metrics_to_plot):
        ax = axes[i]
        for j, model in enumerate(model_names):
            col_name = metric_column_name(model, suffix)
            if col_name not in kpi_df.columns:
                continue

            values = kpi_df[col_name].fillna(0).to_numpy()
            offset = (j - (n_models - 1) / 2) * bar_width
            ax.bar(x + offset, values, width=bar_width, label=model, color=colors[j], alpha=0.9)

        ax.set_title(f"{title} Comparison", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha="right")
        if suffix in ["Prec", "Rec", "F1"]:
            ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

        if i == 0:
            ax.legend(loc="lower right", ncol=min(n_models, 4))

    for j in range(len(metrics_to_plot), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

# ==========================================
# 5. MISMATCH ANALYSIS
# ==========================================

def compute_pairwise_mismatches(df_merged, pairs, threshold=THRESHOLD):
    available_classes = [c.replace("_gt", "") for c in df_merged.columns if str(c).endswith("_gt")]
    diff_data = []

    for cls in available_classes:
        row_data = {"Class": cls}
        col_gt = f"{cls}_gt"

        if col_gt in df_merged.columns:
            gt_full = df_merged[col_gt]
            row_data["Total Frames"] = (gt_full != -1.0).sum()

            for pair_name, model_A, model_B in pairs:
                col_A = f"{cls}_{MODELS_CONFIG[model_A]['suffix']}"
                col_B = f"{cls}_{MODELS_CONFIG[model_B]['suffix']}"

                if col_A in df_merged.columns and col_B in df_merged.columns:
                    prob_A = df_merged[col_A]
                    prob_B = df_merged[col_B]

                    valid_mask = (gt_full != -1.0) & (prob_A != -1.0) & (prob_B != -1.0)
                    pA_valid = prob_A[valid_mask]
                    pB_valid = prob_B[valid_mask]

                    if len(pA_valid) > 0:
                        class_A = (pA_valid >= threshold).astype(int)
                        class_B = (pB_valid >= threshold).astype(int)
                        diff_count = (class_A != class_B).sum()

                        row_data[f"{pair_name} Count"] = diff_count
                        row_data[f"{pair_name} Pct"] = f"{(diff_count / len(pA_valid)) * 100:.2f}%"
                        row_data[f"{pair_name} MeanDiff"] = f"{np.abs(pA_valid - pB_valid).mean() * 100:.2f}%"
                    else:
                        row_data[f"{pair_name} Count"] = None
                        row_data[f"{pair_name} Pct"] = "N/A"
                        row_data[f"{pair_name} MeanDiff"] = "N/A"

            diff_data.append(row_data)

    if not diff_data:
        return pd.DataFrame()

    diff_df = pd.DataFrame(diff_data)
    return diff_df


def print_mismatch_summary(diff_df, pairs):
    if diff_df.empty:
        print("⚠️ Error: Could not compute differences.")
        return

    print("\n🔀 MISMATCHED FRAMES COUNT (Different Classification Output > 0.5)")
    cols_count = ["Class", "Total Frames"] + [f"{p[0]} Count" for p in pairs]
    df_count = diff_df[cols_count].copy()
    df_count.columns = ["Class", "Total Frames"] + [p[0] for p in pairs]
    display(df_count)

    print("\n📊 MISMATCHED FRAMES PERCENTAGE (%)")
    cols_pct = ["Class"] + [f"{p[0]} Pct" for p in pairs]
    df_pct = diff_df[cols_pct].copy()
    df_pct.columns = ["Class"] + [p[0] for p in pairs]
    display(df_pct)

    print("\n📉 AVERAGE PROBABILITY DIFFERENCE (%)")
    cols_diff = ["Class"] + [f"{p[0]} MeanDiff" for p in pairs]
    df_mean_diff = diff_df[cols_diff].copy()
    df_mean_diff.columns = ["Class"] + [p[0] for p in pairs]
    display(df_mean_diff)


# ==========================================
# 6. TIMELINES
# ==========================================

def plot_timelines_for_all_classes(df_merged, models_config=None):
    if models_config is None:
        models_config = MODELS_CONFIG

    available_classes = [c.replace("_gt", "") for c in df_merged.columns if str(c).endswith("_gt")]

    print("📈 GENERATING TIMELINES FOR ALL CLASSES...\n")

    if not available_classes:
        print("⚠️ Error: No classes found to plot.")
        return

    plot_styles = [
        {"color": "blue", "linestyle": "-", "alpha": 0.5, "linewidth": 1.5},
        {"color": "purple", "linestyle": ":", "alpha": 0.7, "linewidth": 2},
        {"color": "orange", "linestyle": "-", "alpha": 0.7, "linewidth": 1.5},
        {"color": "red", "linestyle": "-.", "alpha": 0.8, "linewidth": 1.5},
    ]

    for cls in available_classes:
        col_gt = f"{cls}_gt"
        cols_models = [f"{cls}_{cfg['suffix']}" for cfg in models_config.values()]

        if all(c in df_merged.columns for c in cols_models + [col_gt]):
            plt.figure(figsize=(18, 5))

            gt_values = df_merged[col_gt].replace(-1.0, 0.0)
            plt.plot(df_merged["index"], gt_values * 1.05, label="Ground Truth", color="green", linestyle="--", linewidth=2)

            for idx, (model_name, config) in enumerate(models_config.items()):
                col_model = f"{cls}_{config['suffix']}"
                style = plot_styles[idx % len(plot_styles)]
                plt.plot(
                    df_merged["index"],
                    df_merged[col_model].replace(-1.0, 0.0),
                    label=model_name,
                    **style,
                )

            plt.title(f"Timeline Comparison: {cls}", fontsize=14, fontweight="bold")
            plt.xlabel("Frame Number (Index)")
            plt.ylabel("Probability (0 to 1)")
            plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
            plt.grid(True, alpha=0.4)
            plt.ylim(-0.05, 1.1)
            plt.tight_layout()
            plt.show()
        else:
            print(f"⚠️ Cannot display plot: missing columns for class '{cls}'.")


# ==========================================
# 7. MAIN ENTRY POINT
# ==========================================

def run_analysis(path_gt=PATH_GT, models_config=None, threshold=THRESHOLD):
    if models_config is None:
        models_config = MODELS_CONFIG

    df_merged = load_merged_dataset(path_gt, models_config)
    kpi_df = compute_kpi_by_class(df_merged, models_config, threshold)

    model_names = list(models_config.keys())
    if not kpi_df.empty:
        print("\n📊 F1-SCORE COMPARISON")
        display_metric_table(kpi_df, "F1", "📊 F1-SCORE COMPARISON", model_names)

        print("\n📊 PRECISION COMPARISON")
        display_metric_table(kpi_df, "Prec", "📊 PRECISION COMPARISON", model_names)

        print("\n📊 RECALL COMPARISON")
        display_metric_table(kpi_df, "Rec", "📊 RECALL COMPARISON", model_names)

        print("\n🚨 FALSE POSITIVES")
        display_metric_table(kpi_df, "FP", "🚨 FALSE POSITIVES (FP) - Hallucinations (Lower is better)", model_names)

        print("\n🙈 FALSE NEGATIVES")
        display_metric_table(kpi_df, "FN", "🙈 FALSE NEGATIVES (FN) - Missed Actions (Lower is better)", model_names)

        print("\n🎯 TRUE POSITIVES")
        display_metric_table(kpi_df, "TP", "🎯 TRUE POSITIVES (TP) - Correct Detections (Higher is better)", model_names)

    pairs = [
        ("With keypoints With Draw onnx vs artifact", "with_kpt_with_draw", "without_kpt_without_draw"),
    ]

    diff_df = compute_pairwise_mismatches(df_merged, pairs, threshold)
    if not diff_df.empty:
        print_mismatch_summary(diff_df, pairs)

    plot_kpi_charts(kpi_df, model_names, threshold)
    plot_timelines_for_all_classes(df_merged, models_config)

    return {
        "df_merged": df_merged,
        "kpi_df": kpi_df,
        "diff_df": diff_df,
    }


if __name__ == "__main__":
    run_analysis()