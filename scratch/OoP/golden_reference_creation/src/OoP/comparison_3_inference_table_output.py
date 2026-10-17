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

Comme Inf_Py et Inf_C n'ont pas de GT, on réutilise la GT extraite du CSV Eval
(colonnes "*_gt") pour les merger et calculer leurs KPI — en supposant que les
3 fichiers portent sur la même séquence (merge sur la colonne "index").

IMPORTANT — deux points calqués EXACTEMENT sur evaluation_inference_20_08_kpi_golden_ref.py :

- Frames invalides : les prédictions invalides sont marquées par NaN dans le CSV
  Eval (pas -1 ; -1 ne sert que pour une classe entièrement absente du modèle).
  Le filtrage se fait donc sur isna() ET != -1, exactement comme
  compute_detailed_kpis (comparison_avg_10.py / evaluation_inference_20_08...).
- "correctly_seated" est ABSENT des 3 CSV bruts (il est exclu de final_columns
  dans le script d'export, calculé à part pour le KPI_summary.xlsx). On le
  reconstruit ici avec la même formule que partout ailleurs dans ton code :
      pred_correctly_seated = 1 - clip(max(preds_positifs_valides), 0, 1)
      gt_correctly_seated   = 1 si aucune classe positive valide n'est à 1
  Un frame est invalide pour correctly_seated si AU MOINS une des 8 classes
  positives est invalide (NaN ou -1) — même logique que
  "valid_mask = ~np.isnan(pos_preds).any(axis=1)" dans tes scripts existants.
  NOTE : ceci est une reconstruction, pas la valeur brute du dataset — à
  vérifier une fois contre KPI_summary.xlsx si tu veux la validation finale.
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
    positives, EXACTEMENT comme le fait chaque script existant :
        pred_correctly_seated = 1 - clip(max(pred_positifs_valides), 0, 1)
        gt_correctly_seated   = 1 si aucune classe positive valide n'est à 1, sinon 0
    Un frame est invalide dès qu'au moins une classe positive est invalide
    (NaN ou -1), même logique que "valid_mask = ~np.isnan(pos_preds).any(axis=1)".
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
    pred_correctly_seated[~row_valid_pred] = np.nan  # invalide -> NaN, cohérent avec le reste

    df[CORRECTLY_SEATED_COL] = pred_correctly_seated

    if gt_cols:
        gt_pos = df[gt_cols]
        valid_gt = gt_pos != -1
        row_valid_gt = valid_gt.all(axis=1)
        gt_correctly_seated = (gt_pos.where(valid_gt).sum(axis=1) == 0).astype(float)
        gt_correctly_seated[~row_valid_gt] = -1  # invalide -> sentinel -1, cohérent avec le reste
        df[f"{CORRECTLY_SEATED_COL}_gt"] = gt_correctly_seated

    return df


def _compute_kpis(df, class_order, threshold, source_name):
    """
    Calcule TP/TN/FP/FN/Precision/Recall/F1 par classe sur un DataFrame contenant
    déjà les prédictions ET la GT (colonnes "<classe>_gt"). Seuillage ">= threshold",
    frames invalides (NaN ou -1) exclus indépendamment par classe — même logique
    que compute_detailed_kpis (comparison_avg_10.py / evaluation_inference_20_08...).
    """
    kpi_rows = []
    for cls in class_order:
        col_gt = f"{cls}_gt"
        if col_gt not in df.columns or cls not in df.columns:
            print(f"⚠️  [{source_name}] Colonne manquante pour la classe '{cls}', ignorée.")
            continue

        gt_full = df[col_gt]
        pred_full = df[cls]

        # Invalides = NaN (frame non calculée) OU -1 (classe/colonne absente) —
        # jamais un simple "!= -1" seul, sinon les NaN passent à travers.
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
    output_csv="comparaison_3_sources.csv",
    threshold=0.5,
    cpp_class_order=None,
    reconstruct_correctly_seated=True
):
    """
    Compare l'Eval, l'inférence Python (imitant le C) et l'inférence C réelle
    ("golden reference"), en utilisant la vérité terrain contenue dans eval_csv.

    Args:
        eval_csv (str): CSV de l'évaluation ("*_eval_with_gt.csv"), contenant à
            la fois les prédictions (colonnes classes) et la GT ("<classe>_gt"),
            plus une colonne "index".
        py_inf_csv (str): CSV brut de prédictions de l'inférence Python imitant
            le C (colonnes "index" + classes, sans GT).
        c_inf_csv (str): CSV brut de prédictions du code C ("golden reference"),
            même format que py_inf_csv.
        output_csv (str): Chemin du CSV de sortie combinant les 3 jeux de KPI.
        threshold (float): Seuil de binarisation, ">=" (défaut 0.5).
        cpp_class_order (list, optional): Ordre des 8 classes positives à
            utiliser. Par défaut CPP_CLASS_ORDER.
        reconstruct_correctly_seated (bool): Si True (défaut), reconstruit et
            inclut la classe correctly_seated (absente des 3 CSV bruts) via la
            même formule que le reste de ton code. Mets False pour l'ignorer.
    """
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

    print(f"[Eval]  {len(df_eval)} lignes, index {df_eval['index'].min()}–{df_eval['index'].max()}")
    print(f"[Py]    {len(df_py)} lignes, index {df_py['index'].min()}–{df_py['index'].max()}")
    print(f"[C]     {len(df_c)} lignes, index {df_c['index'].min()}–{df_c['index'].max()}")

    # Extraction de la vérité terrain depuis le CSV Eval (colonnes "*_gt" + index)
    gt_cols = [c for c in df_eval.columns if c == "index" or str(c).endswith("_gt")]
    df_gt = df_eval[gt_cols]

    if len(gt_cols) <= 1:
        print("⚠️  Aucune colonne '*_gt' trouvée dans eval_csv : vérifie le format attendu.")

    # --- 2. MERGE de Py et C avec la GT extraite de l'Eval ---
    df_py_merged = pd.merge(df_py, df_gt, on="index", how="inner")
    print(f"[Merge Py] {len(df_py_merged)} lignes après fusion sur 'index'")

    df_c_merged = pd.merge(df_c, df_gt, on="index", how="inner")
    print(f"[Merge C]  {len(df_c_merged)} lignes après fusion sur 'index'")

    # --- 3. RECONSTRUCTION DE correctly_seated (absent des 3 CSV bruts) ---
    class_order = list(cpp_class_order)
    if reconstruct_correctly_seated:
        df_eval = _reconstruct_correctly_seated(df_eval, cpp_class_order)
        df_py_merged = _reconstruct_correctly_seated(df_py_merged, cpp_class_order)
        df_c_merged = _reconstruct_correctly_seated(df_c_merged, cpp_class_order)
        class_order = [CORRECTLY_SEATED_COL] + class_order

    # --- 4. CALCUL DES KPI POUR LES 3 SOURCES (ordre demandé : Eval, Py, C) ---
    df_kpi_eval = _compute_kpis(df_eval, class_order, threshold, "Eval")
    df_kpi_py = _compute_kpis(df_py_merged, class_order, threshold, "Inf_Py")
    df_kpi_c = _compute_kpis(df_c_merged, class_order, threshold, "Inf_C")

    df_final = pd.concat([df_kpi_eval, df_kpi_py, df_kpi_c], ignore_index=True)
    df_final.to_csv(output_csv, index=False)

    print(f"\n💾 KPI des 3 sources exportés dans {output_csv}")
    print(df_final.to_string(index=False))


if __name__ == "__main__":
    Fire(main)