"""
Compare les KPI frame-level entre 3 sources de prédictions sur un LOT de vidéos :

1. Eval (pipeline Python d'évaluation classique) : Dossier contenant les CSV de 
   prédictions ET la vérité terrain (colonnes "<classe>_gt").
2. Inf_Py (script d'inférence Python imitant le C) : Dossier contenant les CSV bruts.
3. Inf_C (sortie du code C++ embarqué, "golden reference") : Dossier contenant les CSV bruts.

Modifié avec l'alignement strict des frames par vidéo : les 3 sources sont filtrées
pour ne conserver que l'intersection de leurs index. Les frames invalides (NaN ou -1)
sont synchronisées à travers les 3 sources pour garantir un Total_Frames strictement
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


def _missing_source_kpis(class_order, sequence_name):
    return [
        {
            "Sequence": sequence_name,
            "Source": "Inf_C",
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


def process_single_video(eval_csv, py_inf_csv, c_inf_csv, threshold, cpp_class_order, reconstruct_correctly_seated, sequence_name):
    """Traite les 3 CSV correspondants à une même séquence, aligne les frames, et renvoie les DataFrames traités."""
    df_eval = pd.read_csv(eval_csv, low_memory=False)
    df_py = pd.read_csv(py_inf_csv, low_memory=False)
    
    # Chargement intelligent du C++ (Gère l'absence d'en-tête)
    c_inf_missing = not os.path.isfile(c_inf_csv)
    if c_inf_missing:
        common_indices = sorted(set(df_eval["index"]) & set(df_py["index"]))
        df_c = pd.DataFrame({"index": common_indices})
        for cls in cpp_class_order:
            df_c[cls] = -1.0
    else:
        df_c = pd.read_csv(c_inf_csv, low_memory=False)
        if "index" not in df_c.columns:
            df_c = pd.read_csv(c_inf_csv, header=None, low_memory=False)
            ref_cols = ["index"] + cpp_class_order
            # Si la colonne d'index manque carrément dans le fichier, on la recrée
            if len(df_c.columns) == len(ref_cols) - 1:
                df_c.columns = ref_cols[1:]
                df_c.reset_index(names='index', inplace=True)
            else:
                df_c.columns = ref_cols[:len(df_c.columns)]
            
    eval_cols = cpp_class_order + [f"{c}_gt" for c in cpp_class_order]
    df_eval = _to_numeric(df_eval, eval_cols)
    df_py = _to_numeric(df_py, cpp_class_order)
    df_c = _to_numeric(df_c, cpp_class_order)

    # --- ALIGNEMENT STRICT DES FRAMES (Plus Petit Dénominateur Commun) ---
    common_indices = set(df_eval["index"]) & set(df_py["index"]) & set(df_c["index"])
    
    # reset_index garantit que le masque d'invalidité (plus bas) s'appliquera sur des index propres de 0 à N
    df_eval = df_eval[df_eval["index"].isin(common_indices)].sort_values("index").reset_index(drop=True).copy()
    df_py = df_py[df_py["index"].isin(common_indices)].sort_values("index").reset_index(drop=True).copy()
    df_c = df_c[df_c["index"].isin(common_indices)].sort_values("index").reset_index(drop=True).copy()

    # --- SYNCHRONISATION DES INVALIDITÉS ---
    for cls in cpp_class_order:
        if cls in df_eval.columns and cls in df_py.columns and cls in df_c.columns:
            invalid_mask = (
                df_eval[cls].isna() | 
                (df_eval[cls] == -1.0) |
                (df_py[cls] == -1.0)
            )
            if not c_inf_missing:
                invalid_mask |= df_c[cls] == -1.0
            
            df_eval.loc[invalid_mask, cls] = np.nan
            df_py.loc[invalid_mask, cls] = -1.0
            df_c.loc[invalid_mask, cls] = -1.0

    print(f"  -> ✅ {len(common_indices)} frames alignées et synchronisées avec succès.")

    # --- EXTRACTION GT ET MERGE ---
    gt_cols = [c for c in df_eval.columns if c == "index" or str(c).endswith("_gt")]
    df_gt = df_eval[gt_cols]

    if len(gt_cols) <= 1:
        print(f"⚠️ [{sequence_name}] Aucune colonne '*_gt' trouvée dans eval_csv.")
        return None, None, None, None

    df_py_merged = pd.merge(df_py, df_gt, on="index", how="inner")
    df_c_merged = pd.merge(df_c, df_gt, on="index", how="inner")

    class_order = list(cpp_class_order)
    if reconstruct_correctly_seated:
        df_eval = _reconstruct_correctly_seated(df_eval, cpp_class_order)
        df_py_merged = _reconstruct_correctly_seated(df_py_merged, cpp_class_order)
        df_c_merged = _reconstruct_correctly_seated(df_c_merged, cpp_class_order)
        class_order = [CORRECTLY_SEATED_COL] + class_order

    return df_eval, df_py_merged, df_c_merged, class_order


def main(
    eval_csv_dir,
    py_inf_csv_dir,
    c_inf_csv_dir,
    output_csv="comparaison_multi_videos_aligned.csv",
    threshold=0.5,
    cpp_class_order=None,
    reconstruct_correctly_seated=True
):
    """
    Compare l'Eval, l'inférence Python, et l'inférence C sur un dossier complet de vidéos,
    en appliquant un alignement strict des frames.

    Args:
        eval_csv_dir (str): Dossier contenant les CSV de l'évaluation ("*_eval_with_gt.csv").
        py_inf_csv_dir (str): Dossier contenant les CSV de prédictions Python ("*_predictions.csv").
        c_inf_csv_dir (str): Dossier contenant les CSV de prédictions C.
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
    all_py_dfs = []
    all_c_dfs = []
    all_kpi_rows = []
    final_class_order = None
    all_c_inference_missing = True

    for eval_csv in sorted(eval_files):
        basename = os.path.basename(eval_csv)
        sequence_name = basename.replace("_eval_with_gt.csv", "")

        # Chemins exacts basés sur tes dossiers
        py_inf_csv = os.path.join(py_inf_csv_dir, f"{sequence_name}_predictions.csv")
        c_inf_csv = os.path.join(c_inf_csv_dir, f"{sequence_name}_out_positions.csv")

        if not os.path.isfile(py_inf_csv):
            print(f"⚠️ Séquence {sequence_name} ignorée : fichier d'inférence Python introuvable.")
            continue
        c_inf_missing = not os.path.isfile(c_inf_csv)
        if c_inf_missing:
            print(f"⚠️ Fichier d'inférence C introuvable pour {sequence_name} : KPI Inf_C à -1.")
        else:
            all_c_inference_missing = False

        print(f"🔄 Traitement de la séquence : {sequence_name}")

        df_eval, df_py, df_c, class_order = process_single_video(
            eval_csv, py_inf_csv, c_inf_csv, threshold, cpp_class_order, reconstruct_correctly_seated, sequence_name
        )

        if df_eval is not None:
            final_class_order = class_order
            all_eval_dfs.append(df_eval)
            all_py_dfs.append(df_py)
            all_c_dfs.append(df_c)

            all_kpi_rows.extend(_compute_kpis(df_eval, class_order, threshold, "Eval", sequence_name))
            all_kpi_rows.extend(_compute_kpis(df_py, class_order, threshold, "Inf_Py", sequence_name))
            if c_inf_missing:
                all_kpi_rows.extend(_missing_source_kpis(class_order, sequence_name))
            else:
                all_kpi_rows.extend(_compute_kpis(df_c, class_order, threshold, "Inf_C", sequence_name))

    # --- CALCUL GLOBAL (Concaténation) ---
    if all_eval_dfs and final_class_order:
        print(f"\n📊 Calcul des KPI GLOBAUX sur {len(all_eval_dfs)} vidéo(s)...")
        
        global_eval = pd.concat(all_eval_dfs, ignore_index=True)
        global_py = pd.concat(all_py_dfs, ignore_index=True)
        global_c = pd.concat(all_c_dfs, ignore_index=True)

        all_kpi_rows.extend(_compute_kpis(global_eval, final_class_order, threshold, "Eval", "GLOBAL"))
        all_kpi_rows.extend(_compute_kpis(global_py, final_class_order, threshold, "Inf_Py", "GLOBAL"))
        if all_c_inference_missing:
            all_kpi_rows.extend(_missing_source_kpis(final_class_order, "GLOBAL"))
        else:
            all_kpi_rows.extend(_compute_kpis(global_c, final_class_order, threshold, "Inf_C", "GLOBAL"))

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