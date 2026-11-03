import os
import glob
import pandas as pd
import numpy as np
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Pour tous mes codes, j'utilise des environnements Conda.
# Never remove my personal comments from the code.

def check_tracking_ids(tracking_dir, output_plot="bbox_cx_distribution.png"):
    """
    Analyse les fichiers CSV de tracking pour vérifier si les IDs (0 et 1)
    correspondent strictement à une position spatiale (gauche/droite).
    """
    csv_files = glob.glob(os.path.join(tracking_dir, "*.csv"))
    if not csv_files:
        print(f"❌ Aucun fichier CSV trouvé dans {tracking_dir}")
        return

    print(f"🔍 Analyse de {len(csv_files)} fichiers de tracking...")

    cx_0_list = []
    cx_1_list = []

    for file in csv_files:
        try:
            df = pd.read_csv(file)
        except pd.errors.EmptyDataError:
            continue

        # Noms des colonnes générées par SmartEye
        col_x0 = '/trackingState/occupants/0/boundingBox/upperLeft/0'
        col_w0 = '/trackingState/occupants/0/boundingBox/width'
        col_x1 = '/trackingState/occupants/1/boundingBox/upperLeft/0'
        col_w1 = '/trackingState/occupants/1/boundingBox/width'

        # Extraction pour l'ID 0
        if col_x0 in df.columns and col_w0 in df.columns:
            # On ignore les frames où SmartEye a perdu le tracking (NaN)
            valid_0 = df.dropna(subset=[col_x0, col_w0])
            # Calcul du centre X : Position X gauche + (Largeur / 2)
            cx_0 = valid_0[col_x0] + (valid_0[col_w0] / 2)
            cx_0_list.extend(cx_0.tolist())

        # Extraction pour l'ID 1
        if col_x1 in df.columns and col_w1 in df.columns:
            valid_1 = df.dropna(subset=[col_x1, col_w1])
            cx_1 = valid_1[col_x1] + (valid_1[col_w1] / 2)
            cx_1_list.extend(cx_1.tolist())

    # Conversion en arrays NumPy pour les statistiques
    cx_0_arr = np.array(cx_0_list)
    cx_1_arr = np.array(cx_1_list)

    print("\n📊 --- Statistiques des centres X (bbox_cx) ---")
    if len(cx_0_arr) > 0:
        print(f"ID 0 : {len(cx_0_arr)} frames valides | Min: {cx_0_arr.min():.1f} | Max: {cx_0_arr.max():.1f} | Moyenne: {cx_0_arr.mean():.1f}")
    else:
        print("ID 0 : Aucune donnée trouvée.")

    if len(cx_1_arr) > 0:
        print(f"ID 1 : {len(cx_1_arr)} frames valides | Min: {cx_1_arr.min():.1f} | Max: {cx_1_arr.max():.1f} | Moyenne: {cx_1_arr.mean():.1f}")
    else:
        print("ID 1 : Aucune donnée trouvée.")

    # Création de l'histogramme
    plt.figure(figsize=(10, 6))
    
    if len(cx_0_arr) > 0:
        plt.hist(cx_0_arr, bins=60, alpha=0.6, label='ID 0', color='blue')
    if len(cx_1_arr) > 0:
        plt.hist(cx_1_arr, bins=60, alpha=0.6, label='ID 1', color='orange')

    plt.title("Distribution spatiale des Bounding Boxes (Axe X)")
    plt.xlabel("Position X du centre de la Bbox (Pixels)")
    plt.ylabel("Nombre de frames")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    plt.savefig(output_plot, dpi=150)
    plt.close()
    
    print(f"\n📈 Graphique sauvegardé sous : {output_plot}")
    print("-> Si les couleurs bleue et orange se croisent ou se superposent sur le graphique,")
    print("   cela prouve que SmartEye mélange les IDs. Il faudra alors utiliser bbox_cx dans l'inférence.")
    print("-> Si les couleurs sont parfaitement séparées (ex: bleu à gauche, orange à droite),")
    print("   alors ID 0 = Conducteur et ID 1 = Passager de façon fiable.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vérifie si les IDs SmartEye correspondent aux sièges.")
    parser.add_argument("--tracking_dir", type=str, required=True, help="Chemin vers le dossier contenant les CSV de tracking.")
    args = parser.parse_args()
    
    check_tracking_ids(args.tracking_dir)