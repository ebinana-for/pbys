"""Script pour visualiser et comparer les prédictions d'inférence C++ vs Python."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse

def _plot_single_axis(ax, df_merged, frames, cls, show_legend):
    """Fonction utilitaire pour dessiner une classe sur un axe donné avec les bonnes limites."""
    col_cpp = f"{cls}_cpp"
    col_py = f"{cls}_py"

    # Traçage de la courbe C++ (Bleu, trait continu)
    if col_cpp in df_merged.columns:
        ax.plot(frames, df_merged[col_cpp], label='C++ (TIDL INT8)', color='#1f77b4', alpha=0.8, linewidth=2.5)
        
    # Traçage de la courbe Python (Orange, pointillés)
    if col_py in df_merged.columns:
        ax.plot(frames, df_merged[col_py], label='Python (ONNX FP32)', color='#ff7f0e', alpha=0.9, linestyle='--', linewidth=2)

    # --- LISIBILITÉ ET COSMÉTIQUE (0 à 1) ---
    ax.set_title(f"Classe : {cls}", fontsize=15, fontweight='bold', pad=10)
    ax.set_ylabel("Probabilité", fontsize=11)
    
    # Limites strictes de l'axe Y : de 0 à 1 (avec une très légère marge)
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    
    # Quadrillage et seuil
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.grid(True, axis='x', linestyle=':', alpha=0.3)
    ax.axhline(0.5, color='gray', alpha=0.3, linestyle='-.', linewidth=1)
    
    if show_legend:
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)


def plot_csv_comparison(csv_cpp, csv_python, output_base, mode):
    """
    Charge les CSV et génère les graphiques selon le mode choisi (--all ou --single).
    Les images sont sauvegardées dans le sous-dossier 'plot/'.
    """
    if not os.path.exists(csv_cpp):
        print(f"❌ Erreur : Le fichier C++ '{csv_cpp}' est introuvable.")
        sys.exit(1)
    if not os.path.exists(csv_python):
        print(f"❌ Erreur : Le fichier Python '{csv_python}' est introuvable.")
        sys.exit(1)

    print("Chargement des données...")
    df_cpp = pd.read_csv(csv_cpp)
    df_py = pd.read_csv(csv_python)

    if 'index' not in df_cpp.columns or 'index' not in df_py.columns:
        print("❌ Erreur : La colonne 'index' doit être présente dans les CSV.")
        sys.exit(1)

    # Fusion des dataframes
    df_merged = pd.merge(df_cpp, df_py, on='index', suffixes=('_cpp', '_py'), how='outer').sort_values('index')

    # --- CRÉATION DES COUPURES (GAPS) ---
    # Remplacement des -1.0 par NaN pour interrompre proprement les courbes
    df_merged.replace(-1.0, np.nan, inplace=True)

    classes = [col for col in df_cpp.columns if col != 'index']
    frames = df_merged['index']

    # --- CRÉATION DU DOSSIER DE SORTIE ---
    plot_dir = "plot"
    os.makedirs(plot_dir, exist_ok=True)

    print(f"Génération des graphiques pour {len(classes)} classes (Mode : {mode})...")

    # ==========================================
    # MODE : --all (Toutes les classes sur une image)
    # ==========================================
    if mode == 'all':
        fig, axes = plt.subplots(nrows=len(classes), ncols=1, figsize=(16, 4 * len(classes)), sharex=True)
        if len(classes) == 1: axes = [axes]

        for i, (ax, cls) in enumerate(zip(axes, classes)):
            _plot_single_axis(ax, df_merged, frames, cls, show_legend=(i==0))

        axes[-1].set_xlabel("Frames (Index de la vidéo)", fontsize=13, fontweight='bold', labelpad=15)
        fig.tight_layout(h_pad=2.0)
        
        # Sauvegarde dans le dossier plot
        filename = f"{output_base}.png" if not output_base.endswith('.png') else output_base
        out_path = os.path.join(plot_dir, filename)
        
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"✅ Image globale sauvegardée sous : {out_path}")
        
        plt.show()

    # ==========================================
    # MODE : --single (Une fenêtre par classe)
    # ==========================================
    elif mode == 'single':
        # Nettoyage du nom de base pour éviter de doubler les extensions
        base_name = os.path.basename(output_base)
        base, ext = os.path.splitext(base_name)
        if not ext: ext = ".png"

        for cls in classes:
            fig, ax = plt.subplots(figsize=(12, 5))
            _plot_single_axis(ax, df_merged, frames, cls, show_legend=True)
            
            ax.set_xlabel("Frames (Index de la vidéo)", fontsize=13, fontweight='bold', labelpad=15)
            fig.tight_layout()
            
            # Sauvegarde individuelle dans le dossier plot
            out_path = os.path.join(plot_dir, f"{base}_{cls}{ext}")
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            print(f"✅ Image générée : {out_path}")
        
        # Ouvre toutes les fenêtres simultanément
        print("Affichage des fenêtres...")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparateur visuel des inférences C++ et Python.")
    parser.add_argument("--cpp", type=str, required=True, help="Chemin vers le CSV généré par le C++")
    parser.add_argument("--py", type=str, required=True, help="Chemin vers le CSV généré par le Python")
    parser.add_argument("--out", type=str, default="comparaison_inference", help="Nom de base pour l'image de sortie")
    
    # Création du groupe mutuellement exclusif pour les options d'affichage
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Affiche toutes les classes sur une seule grande figure (comportement par défaut).")
    group.add_argument("--single", action="store_true", help="Ouvre une fenêtre indépendante pour chaque classe.")
    
    args = parser.parse_args()
    
    # Détermination du mode (par défaut 'all' si rien n'est spécifié)
    selected_mode = 'single' if args.single else 'all'
    
    plot_csv_comparison(args.cpp, args.py, args.out, selected_mode)







   python posture_evaluation_new_metrics.py   --model_path=/mnt/data/OoP/data/source_code/training/saved_models/OoP/ASeed/model/ckpt/best_epoch_38.pth \
        --output_dir=/mnt/data/OoP/data/source_code/training/saved_evals/OoP/all/eval_best_model_4_sequences_forvia/seeded_model/epoch38  \
        --config_path=/mnt/data/OoP/data/source_code/training/config/OoP/eval/eval_onnx_july_4_forvia_sequences.yaml

