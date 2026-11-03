import pandas as pd
import sys
import os

# Environnement Conda
# Assure-toi que ton environnement conda est activé (ex: conda activate mon_env_pandas)
# Ne supprime pas tes commentaires personnels dans ce code.

def generer_csv_global(fichiers, nom_sortie="resultats_globaux.csv"):
    """
    Génère un fichier CSV unique comparant Inf_C et Eval
    pour plusieurs fichiers.
    """
    lignes = []
    
    for fichier_csv in fichiers:
        nom_fichier = os.path.basename(fichier_csv)
        try:
            df = pd.read_csv(fichier_csv)
            if 'Source' not in df.columns:
                df = pd.read_csv(fichier_csv, delim_whitespace=True, header=None)
                if len(df.columns) == 11:
                    df.columns = ['Sequence', 'Source', 'Class', 'TP', 'TN', 'FP', 'FN', 'Precision', 'Recall', 'F1_Score', 'Total_Frames']
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier {nom_fichier} : {e}")
            continue

        if 'Sequence' in df.columns:
            df_global = df[df['Sequence'] == 'GLOBAL']
            if not df_global.empty:
                df = df_global

        if 'Source' not in df.columns or 'Class' not in df.columns:
            continue

        df_eval = df[df['Source'] == 'Eval'].set_index('Class')
        df_inf = df[df['Source'] == 'Inf_C'].set_index('Class')

        classes = df_eval.index.intersection(df_inf.index)

        for cls in classes:
            ev = df_eval.loc[cls]
            inf = df_inf.loc[cls]
            
            if isinstance(inf, pd.DataFrame): inf = inf.iloc[0]
            if isinstance(ev, pd.DataFrame): ev = ev.iloc[0]

            tp = int(inf['TP'])
            d_tp = tp - int(ev['TP'])
            
            tn = int(inf['TN'])
            d_tn = tn - int(ev['TN'])
            
            fp = int(inf['FP'])
            d_fp = fp - int(ev['FP'])
            
            fn = int(inf['FN'])
            d_fn = fn - int(ev['FN'])

            p = float(inf['Precision'])
            d_p = (p - float(ev['Precision'])) * 100
            
            r = float(inf['Recall'])
            d_r = (r - float(ev['Recall'])) * 100
            
            f1 = float(inf['F1_Score'])
            d_f1 = (f1 - float(ev['F1_Score'])) * 100

            frames = int(inf.get('Total_Frames', inf.get('Total Frames', 0)))

            lignes.append({
                'Fichier': nom_fichier,
                'Class': cls,
                'TP': f"{tp} ({d_tp:+d})",
                'TN': f"{tn} ({d_tn:+d})",
                'FP': f"{fp} ({d_fp:+d})",
                'FN': f"{fn} ({d_fn:+d})",
                'Precision': f"{p:.2f} ({d_p:+.1f}%)",
                'Recall': f"{r:.2f} ({d_r:+.1f}%)",
                'F1_Score': f"{f1:.2f} ({d_f1:+.1f}%)",
                'Total Frames': frames
            })

    if lignes:
        df_out = pd.DataFrame(lignes)
        df_out.to_csv(nom_sortie, index=False, encoding='utf-8')
        print(f"Fichier généré avec succès : {nom_sortie}")
    else:
        print("Aucune donnée à exporter.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generateur_csv_multi.py <fichier1.csv> <fichier2.csv> ...")
    else:
        # sys.argv[1:] récupère tous les fichiers passés en argument
        fichiers_entree = sys.argv[1:]
        nom_sortie = "resultats_globaux.csv"
        # On peut optionnellement passer le nom de sortie avec un flag, 
        # mais on garde ça simple ici.
        generer_csv_global(fichiers_entree, nom_sortie)