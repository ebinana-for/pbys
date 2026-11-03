"""
Compare les KPI frame-level entre 4 sources de prédictions sur un LOT de vidéos :

1. Eval (pipeline Python d'évaluation classique) : Dossier contenant les CSV de 
   prédictions ET la vérité terrain (colonnes "<classe>_gt").
2. Inf32 (inférence float32) : Dossier contenant les CSV bruts.
3. Inf8 (inférence int8) : Dossier contenant les CSV bruts.
4. InfBoard (inférence sur la cible embarquée) : Dossier contenant les CSV bruts.

Modifié avec l'alignement strict des frames par vidéo : les 4 sources sont filtrées
pour ne conserver que l'intersection de leurs index. Les frames invalides (NaN ou -1)
sont synchronisées à travers les 4 sources pour garantir un Total_Frames strictement
identique avant le calcul des métriques par vidéo et globales.
"""
import os
import glob
import numpy as np
import pandas as pd
from fire import Fire
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Pour tous mes codes, j'utilise des environnements Conda.
# Never remove my personal comments from the code.

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

CORRECTLY_SEATED_COL = "correctly_seated"


def _to_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _reconstruct_correctly_seated(df, cpp_class_order):
    df = df.copy()
    pred_cols = [c for c in cpp_class_order if c in df.columns]
    gt_cols = [f"{c}_gt" for c in cpp_class_order if f"{c}_gt" in df.columns]

    if not pred_cols:
        return df

    pred_pos = df[pred_cols]
    valid_pred = (~pred_pos.isna()) & (pred_pos != -1.0)
    row_valid_pred = valid_pred.all(axis=1)

    pred_masked = pred_pos.where(valid_pred)
    pred_correctly_seated = 1.0 - pred_masked.max(axis=1).clip(0, 1)
    pred_correctly_seated[~row_valid_pred] = np.nan 

    df[CORRECTLY_SEATED_COL] = pred_correctly_seated

    if gt_cols:
        gt_pos = df[gt_cols]
        valid_gt = gt_pos != -1
        row_valid_gt = valid_gt.all(axis=1)
        gt_correctly_seated = (gt_pos.where(valid_gt).sum(axis=1) == 0).astype(float)
        gt_correctly_seated[~row_valid_gt] = -1 
        df[f"{CORRECTLY_SEATED_COL}_gt"] = gt_correctly_seated

    return df


def _compute_kpis(df, class_order, threshold, source_name, sequence_name):
    kpi_rows = []
    for cls in class_order:
        col_gt = f"{cls}_gt"
        if col_gt not in df.columns or cls not in df.columns:
            continue

        gt_full = df[col_gt]
        pred_full = df[cls]

        valid_mask = (~pred_full.isna()) & (pred_full != -1.0) & (gt_full != -1)
        gt_valid = gt_full[valid_mask]
        pred_probs = pred_full[valid_mask]

        if len(gt_valid) == 0:
            continue

        y_pred = (pred_probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(gt_valid, y_pred, labels=[0, 1]).ravel()

        kpi_rows.append({
            "Sequence": sequence_name,
            "Source": source_name,
            "Class": cls,
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "Precision": round(precision_score(gt_valid, y_pred, zero_division=0), 4),
            "Recall": round(recall_score(gt_valid, y_pred, zero_division=0), 4),
            "F1_Score": round(f1_score(gt_valid, y_pred, zero_division=0), 4),
            "Total_Frames": int(len(gt_valid)),
        })

    return kpi_rows


def _missing_source_kpis(class_order, source_name, sequence_name):
    return [
        {
            "Sequence": sequence_name,
            "Source": source_name,
            "Class": cls,
            "TP": -1,
            "TN": -1,
            "FP": -1,
            "FN": -1,
            "Precision": -1,
            "Recall": -1,
            "F1_Score": -1,
            "Total_Frames": -1,
        }
        for cls in class_order
    ]


def _find_csv(directory, seq):
    """Recherche intelligemment le fichier CSV correspondant à la séquence."""
    if not directory or not os.path.isdir(directory):
        return ""
    for suffix in ["_predictions.csv", "_out_positions.csv", ".csv"]:
        p = os.path.join(directory, f"{seq}{suffix}")
        if os.path.isfile(p):
            return p
    return os.path.join(directory, f"{seq}_predictions.csv")


def process_single_video(eval_csv, inf32_csv, inf8_csv, infboard_csv, threshold, cpp_class_order, reconstruct_correctly_seated, sequence_name):
    """Traite les CSV correspondants à une même séquence, aligne les frames, et renvoie les DataFrames traités."""
    df_eval = pd.read_csv(eval_csv, low_memory=False)
    
    def load_inf_csv(path):
        if not os.path.isfile(path):
            return None
        df = pd.read_csv(path, low_memory=False)
        if "index" not in df.columns:
            df = pd.read_csv(path, header=None, low_memory=False)
            ref_cols = ["index"] + cpp_class_order
            if len(df.columns) == len(ref_cols) - 1:
                df.columns = ref_cols[1:]
                df.reset_index(names='index', inplace=True)
            else:
                df.columns = ref_cols[:len(df.columns)]
        return _to_numeric(df, cpp_class_order)

    df_inf32 = load_inf_csv(inf32_csv)
    df_inf8 = load_inf_csv(inf8_csv)
    df_infboard = load_inf_csv(infboard_csv)

    eval_cols = cpp_class_order + [f"{c}_gt" for c in cpp_class_order]
    df_eval = _to_numeric(df_eval, eval_cols)

    # --- ALIGNEMENT STRICT DES FRAMES (Plus Petit Dénominateur Commun) ---
    common_indices = set(df_eval["index"])
    if df_inf32 is not None: common_indices &= set(df_inf32["index"])
    if df_inf8 is not None: common_indices &= set(df_inf8["index"])
    if df_infboard is not None: common_indices &= set(df_infboard["index"])
    
    def enforce_common_and_fill(df, is_missing):
        if is_missing:
            dummy = pd.DataFrame({"index": sorted(list(common_indices))})
            for cls in cpp_class_order:
                dummy[cls] = -1.0
            return dummy
        return df[df["index"].isin(common_indices)].sort_values("index").reset_index(drop=True).copy()

    df_eval = enforce_common_and_fill(df_eval, False)
    df_inf32 = enforce_common_and_fill(df_inf32, df_inf32 is None)
    df_inf8 = enforce_common_and_fill(df_inf8, df_inf8 is None)
    df_infboard = enforce_common_and_fill(df_infboard, df_infboard is None)

    # --- SYNCHRONISATION DES INVALIDITÉS ---
    for cls in cpp_class_order:
        if cls in df_eval.columns:
            invalid_mask = (df_eval[cls].isna()) | (df_eval[cls] == -1.0)
            if df_inf32 is not None and cls in df_inf32.columns:
                invalid_mask |= (df_inf32[cls] == -1.0)
            if df_inf8 is not None and cls in df_inf8.columns:
                invalid_mask |= (df_inf8[cls] == -1.0)
            if df_infboard is not None and cls in df_infboard.columns:
                invalid_mask |= (df_infboard[cls] == -1.0)

            df_eval.loc[invalid_mask, cls] = np.nan
            df_inf32.loc[invalid_mask, cls] = -1.0
            df_inf8.loc[invalid_mask, cls] = -1.0
            df_infboard.loc[invalid_mask, cls] = -1.0

    print(f"  -> ✅ {len(common_indices)} frames alignées et synchronisées avec succès.")

    # --- EXTRACTION GT ET MERGE ---
    gt_cols = [c for c in df_eval.columns if c == "index" or str(c).endswith("_gt")]
    df_gt = df_eval[gt_cols]

    if len(gt_cols) <= 1:
        print(f"⚠️ [{sequence_name}] Aucune colonne '*_gt' trouvée dans eval_csv.")
        return None, None, None, None, None

    df_inf32_merged = pd.merge(df_inf32, df_gt, on="index", how="inner")
    df_inf8_merged = pd.merge(df_inf8, df_gt, on="index", how="inner")
    df_infboard_merged = pd.merge(df_infboard, df_gt, on="index", how="inner")

    class_order = list(cpp_class_order)
    if reconstruct_correctly_seated:
        df_eval = _reconstruct_correctly_seated(df_eval, cpp_class_order)
        df_inf32_merged = _reconstruct_correctly_seated(df_inf32_merged, cpp_class_order)
        df_inf8_merged = _reconstruct_correctly_seated(df_inf8_merged, cpp_class_order)
        df_infboard_merged = _reconstruct_correctly_seated(df_infboard_merged, cpp_class_order)
        class_order = [CORRECTLY_SEATED_COL] + class_order

    return df_eval, df_inf32_merged, df_inf8_merged, df_infboard_merged, class_order


def main(
    eval_csv_dir,
    inf32_csv_dir,
    inf8_csv_dir,
    infboard_csv_dir,
    output_csv="comparaison_multi_videos_aligned.csv",
    threshold=0.5,
    cpp_class_order=None,
    reconstruct_correctly_seated=True
):
    """
    Compare l'Eval, l'inférence 32 bits, l'inférence 8 bits, et l'inférence embarquée (board)
    sur un dossier complet de vidéos, en appliquant un alignement strict des frames.

    Args:
        eval_csv_dir (str): Dossier contenant les CSV de l'évaluation ("*_eval_with_gt.csv").
        inf32_csv_dir (str): Dossier contenant les CSV de prédictions Inf32.
        inf8_csv_dir (str): Dossier contenant les CSV de prédictions Inf8.
        infboard_csv_dir (str): Dossier contenant les CSV de prédictions InfBoard.
        output_csv (str): Fichier CSV de sortie (par vidéo + GLOBAL).
    """
    if cpp_class_order is None:
        cpp_class_order = list(CPP_CLASS_ORDER)

    threshold = float(threshold)

    eval_files = glob.glob(os.path.join(eval_csv_dir, "*_eval_with_gt.csv"))
    if not eval_files:
        print(f"❌ Aucun fichier trouvé dans {eval_csv_dir}")
        return

    all_eval_dfs = []
    all_inf32_dfs = []
    all_inf8_dfs = []
    all_infboard_dfs = []
    
    all_kpi_rows = []
    final_class_order = None
    
    all_inf32_missing = True
    all_inf8_missing = True
    all_infboard_missing = True

    for eval_csv in sorted(eval_files):
        basename = os.path.basename(eval_csv)
        sequence_name = basename.replace("_eval_with_gt.csv", "")

        inf32_csv = _find_csv(inf32_csv_dir, sequence_name)
        inf8_csv = _find_csv(inf8_csv_dir, sequence_name)
        infboard_csv = _find_csv(infboard_csv_dir, sequence_name)

        inf32_missing = not os.path.isfile(inf32_csv)
        inf8_missing = not os.path.isfile(inf8_csv)
        infboard_missing = not os.path.isfile(infboard_csv)

        if not inf32_missing: all_inf32_missing = False
        if not inf8_missing: all_inf8_missing = False
        if not infboard_missing: all_infboard_missing = False

        if inf32_missing: print(f"⚠️ Séquence {sequence_name} : Inf32 introuvable.")
        if inf8_missing: print(f"⚠️ Séquence {sequence_name} : Inf8 introuvable.")
        if infboard_missing: print(f"⚠️ Séquence {sequence_name} : InfBoard introuvable.")

        print(f"🔄 Traitement de la séquence : {sequence_name}")

        df_eval, df_inf32, df_inf8, df_infboard, class_order = process_single_video(
            eval_csv, inf32_csv, inf8_csv, infboard_csv, threshold, cpp_class_order, reconstruct_correctly_seated, sequence_name
        )

        if df_eval is not None:
            final_class_order = class_order
            all_eval_dfs.append(df_eval)
            all_inf32_dfs.append(df_inf32)
            all_inf8_dfs.append(df_inf8)
            all_infboard_dfs.append(df_infboard)

            all_kpi_rows.extend(_compute_kpis(df_eval, class_order, threshold, "Eval", sequence_name))
            
            if inf32_missing:
                all_kpi_rows.extend(_missing_source_kpis(class_order, "Inf32", sequence_name))
            else:
                all_kpi_rows.extend(_compute_kpis(df_inf32, class_order, threshold, "Inf32", sequence_name))
                
            if inf8_missing:
                all_kpi_rows.extend(_missing_source_kpis(class_order, "Inf8", sequence_name))
            else:
                all_kpi_rows.extend(_compute_kpis(df_inf8, class_order, threshold, "Inf8", sequence_name))
                
            if infboard_missing:
                all_kpi_rows.extend(_missing_source_kpis(class_order, "InfBoard", sequence_name))
            else:
                all_kpi_rows.extend(_compute_kpis(df_infboard, class_order, threshold, "InfBoard", sequence_name))

    # --- CALCUL GLOBAL (Concaténation) ---
    if all_eval_dfs and final_class_order:
        print(f"\n📊 Calcul des KPI GLOBAUX sur {len(all_eval_dfs)} vidéo(s)...")
        
        global_eval = pd.concat(all_eval_dfs, ignore_index=True)
        global_inf32 = pd.concat(all_inf32_dfs, ignore_index=True)
        global_inf8 = pd.concat(all_inf8_dfs, ignore_index=True)
        global_infboard = pd.concat(all_infboard_dfs, ignore_index=True)

        all_kpi_rows.extend(_compute_kpis(global_eval, final_class_order, threshold, "Eval", "GLOBAL"))
        
        if all_inf32_missing:
            all_kpi_rows.extend(_missing_source_kpis(final_class_order, "Inf32", "GLOBAL"))
        else:
            all_kpi_rows.extend(_compute_kpis(global_inf32, final_class_order, threshold, "Inf32", "GLOBAL"))

        if all_inf8_missing:
            all_kpi_rows.extend(_missing_source_kpis(final_class_order, "Inf8", "GLOBAL"))
        else:
            all_kpi_rows.extend(_compute_kpis(global_inf8, final_class_order, threshold, "Inf8", "GLOBAL"))

        if all_infboard_missing:
            all_kpi_rows.extend(_missing_source_kpis(final_class_order, "InfBoard", "GLOBAL"))
        else:
            all_kpi_rows.extend(_compute_kpis(global_infboard, final_class_order, threshold, "InfBoard", "GLOBAL"))

        df_final = pd.DataFrame(all_kpi_rows)
        
        is_global = df_final['Sequence'] == 'GLOBAL'
        df_seq = df_final[~is_global].sort_values(['Sequence', 'Source', 'Class'])
        df_global = df_final[is_global].sort_values(['Source', 'Class'])
        df_final_sorted = pd.concat([df_seq, df_global]).reset_index(drop=True)

        df_final_sorted.to_csv(output_csv, index=False)
        print(f"\n💾 KPI multi-vidéos exportés dans {output_csv}")
    else:
        print("\n❌ Aucun calcul n'a pu être effectué. Vérifiez les chemins des dossiers.")


if __name__ == "__main__":
    Fire(main)