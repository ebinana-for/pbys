"""
Compare les KPI frame-level entre 3 sources de prédictions pour une même séquence :

1. Eval (pipeline Python d'évaluation classique) : CSV contenant à la fois les
   prédictions ET la vérité terrain (colonnes "<classe>_gt"), produit par
   evaluation_inference_20_08_kpi_golden_ref.py (fichiers "*_eval_with_gt.csv").
2. Inf_Py (script d'inférence Python imitant le C) : CSV brut de prédictions
   (colonnes "index" + classes, SANS GT), produit par run_video_inference()
   (py_inference_v2_same_result_C_kpi_golden_ref.py).
3. Inf_C (sortie du code C++ embarqué, "golden reference") : même format que
   Inf_Py, sans GT non plus.

Modifié avec l'alignement strict des frames : les 3 sources sont filtrées pour
ne conserver que l'intersection de leurs index. Cela garantit une comparaison
parfaitement équitable des FP et TP, même si un modèle a subi des pertes de tracking.
"""
import numpy as np
import pandas as pd
from fire import Fire
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

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
    """Force la conversion en numérique (les CSV peuvent contenir des NaN/valeurs vides)."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _reconstruct_correctly_seated(df, cpp_class_order):
    """
    Reconstruit correctly_seated (absent des CSV bruts) à partir des 8 classes
    positives.
    """
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


def _compute_kpis(df, class_order, threshold, source_name):
    """
    Calcule TP/TN/FP/FN/Precision/Recall/F1 par classe sur un DataFrame contenant
    déjà les prédictions ET la GT.
    """
    kpi_rows = []
    for cls in class_order:
        col_gt = f"{cls}_gt"
        if col_gt not in df.columns or cls not in df.columns:
            print(f"⚠️  [{source_name}] Colonne manquante pour la classe '{cls}', ignorée.")
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

    return pd.DataFrame(kpi_rows)


def main(
    eval_csv,
    py_inf_csv,
    c_inf_csv,
    output_csv="comparaison_3_sources_aligned.csv",
    threshold=0.5,
    cpp_class_order=None,
    reconstruct_correctly_seated=True
):
    if cpp_class_order is None:
        cpp_class_order = list(CPP_CLASS_ORDER)

    threshold = float(threshold)

    # --- 1. CHARGEMENT ---
    df_eval = pd.read_csv(eval_csv)
    df_py = pd.read_csv(py_inf_csv)
    df_c = pd.read_csv(c_inf_csv)

    eval_cols = cpp_class_order + [f"{c}_gt" for c in cpp_class_order]
    df_eval = _to_numeric(df_eval, eval_cols)
    df_py = _to_numeric(df_py, cpp_class_order)
    df_c = _to_numeric(df_c, cpp_class_order)

    print("\n--- AVANT ALIGNEMENT ---")
    print(f"[Eval]  {len(df_eval)} lignes")
    print(f"[Py]    {len(df_py)} lignes")
    print(f"[C]     {len(df_c)} lignes")

    # --- 2. ALIGNEMENT STRICT DES FRAMES (Plus Petit Dénominateur Commun) ---
    common_indices = set(df_eval["index"]) & set(df_py["index"]) & set(df_c["index"])
    
    # Le sort_values("index") est crucial ici pour que les masques correspondent ligne à ligne
    df_eval = df_eval[df_eval["index"].isin(common_indices)].sort_values("index").copy()
    df_py = df_py[df_py["index"].isin(common_indices)].sort_values("index").copy()
    df_c = df_c[df_c["index"].isin(common_indices)].sort_values("index").copy()

    # --- NOUVEAU : SYNCHRONISATION DES INVALIDITÉS ---
    # Si l'un des modèles a échoué (NaN ou -1.0) sur une frame, on l'annule pour les 3.
    for cls in cpp_class_order:
        if cls in df_eval.columns and cls in df_py.columns and cls in df_c.columns:
            # Identifie les frames où au moins une source est invalide
            invalid_mask = (
                df_eval[cls].isna() | 
                (df_eval[cls] == -1.0) | 
                (df_py[cls] == -1.0) | 
                (df_c[cls] == -1.0)
            )
            
            # Applique l'invalidité partout
            df_eval.loc[invalid_mask, cls] = np.nan
            df_py.loc[invalid_mask, cls] = -1.0
            df_c.loc[invalid_mask, cls] = -1.0

    print("\n--- APRÈS ALIGNEMENT ---")
    print(f"✅ {len(common_indices)} frames communes conservées.")
    print(f"✅ Invalidités synchronisées : les sources auront exactement le même Total_Frames.")

    # --- 3. EXTRACTION GT ET MERGE ---
    # La GT extraite de df_eval est désormais elle aussi parfaitement alignée
    gt_cols = [c for c in df_eval.columns if c == "index" or str(c).endswith("_gt")]
    df_gt = df_eval[gt_cols]

    if len(gt_cols) <= 1:
        print("⚠️  Aucune colonne '*_gt' trouvée dans eval_csv : vérifie le format attendu.")

    df_py_merged = pd.merge(df_py, df_gt, on="index", how="inner")
    df_c_merged = pd.merge(df_c, df_gt, on="index", how="inner")

    # --- 4. RECONSTRUCTION DE correctly_seated ---
    class_order = list(cpp_class_order)
    if reconstruct_correctly_seated:
        df_eval = _reconstruct_correctly_seated(df_eval, cpp_class_order)
        df_py_merged = _reconstruct_correctly_seated(df_py_merged, cpp_class_order)
        df_c_merged = _reconstruct_correctly_seated(df_c_merged, cpp_class_order)
        class_order = [CORRECTLY_SEATED_COL] + class_order

    # --- 5. CALCUL DES KPI POUR LES 3 SOURCES ---
    df_kpi_eval = _compute_kpis(df_eval, class_order, threshold, "Eval")
    df_kpi_py = _compute_kpis(df_py_merged, class_order, threshold, "Inf_Py")
    df_kpi_c = _compute_kpis(df_c_merged, class_order, threshold, "Inf_C")

    df_final = pd.concat([df_kpi_eval, df_kpi_py, df_kpi_c], ignore_index=True)
    df_final.to_csv(output_csv, index=False)

    print(f"\n💾 KPI des 3 sources exportés dans {output_csv}")
    print(df_final.to_string(index=False))


if __name__ == "__main__":
    Fire(main)