import os
from pathlib import Path
import subprocess
import pandas as pd

# Configurations des chemins
BASE_DIR = Path("/mnt/data/OoP/data/DATA_HEIGHTWEIGHT")
OUTPUT_BASE_DIR = BASE_DIR / "faces_db_cv5"  # Nouveau dossier pour la Cross-Validation
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

# Chemin vers le script d'extraction individuel (utiliser le chemin de l'environnement)
EXTRACTOR_SCRIPT = "/mnt/data/OoP/data/source_code/git-python/training/src/age_gender/build_forvia_faces_db.py"

def iter_valid_pairs(base_dir: Path, video_list: list):
    """Générateur qui vérifie l'existence du mp4 et du csv pour une liste précise."""
    for video_name in video_list:
        mp4 = base_dir / video_name
        csv_path = mp4.with_suffix(".csv")

        if not mp4.exists():
            print(f"⚠️  SKIP {mp4.name}: Vidéo introuvable dans {base_dir}")
            continue

        if not csv_path.exists():
            print(f"⚠️  SKIP {mp4.name}: Aucun CSV de tracking associé")
            continue

        yield mp4, csv_path

def process_split(split_file: str, fold_dir: Path, task: str):
    """Lit un fichier CSV de split et lance l'extraction pour toutes ses vidéos."""
    if not os.path.exists(split_file):
        print(f"❌ Fichier de split introuvable : {split_file}")
        return

    print(f"\n{'='*50}")
    print(f"🚀 Démarrage du traitement : {task.upper()} (Liste : {split_file})")
    print(f"📂 Destination : {fold_dir}")
    print(f"{'='*50}\n")

    # Lire la liste des vidéos pour ce split
    df_split = pd.read_csv(split_file)
    video_list = df_split['video_name'].tolist()

    for mp4, csv_path in iter_valid_pairs(BASE_DIR, video_list):
        print(f"  -> Traitement: {mp4.name}")

        subprocess.run(
            [
                "python",
                EXTRACTOR_SCRIPT,
                "--video_path", str(mp4),
                "--csv_file", str(csv_path),
                "--out_path", str(fold_dir),
                "--task", task,
            ],
            check=True,
        )

def main():
    # Détection des fichiers de split générés précédemment
    # On suppose qu'ils sont dans le dossier d'exécution actuel (le workspace)
    # Mais pour être robuste, on peut chercher de 1 à 5

    for fold in range(1, 6):
        train_file = f"split_{fold}_train.csv"
        test_file = f"split_{fold}_test.csv"

        if not os.path.exists(train_file) or not os.path.exists(test_file):
            print(f"⏩ Fold {fold} ignoré : fichiers de split introuvables.")
            continue

        print(f"\n{'#'*60}")
        print(f"   LANCEMENT DE L'EXTRACTION POUR LE FOLD {fold}   ")
        print(f"{'#'*60}")

        # Créer un sous-dossier dédié pour ce Fold
        fold_dir = OUTPUT_BASE_DIR / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Les images iront dans fold_X/images (géré par build_forvia_faces_db.py)

        # 1. Traiter la liste d'entraînement
        process_split(train_file, fold_dir, "train")

        # 2. Traiter la liste de test
        process_split(test_file, fold_dir, "test")

    print(f"\n✅ EXTRACTION TERMINÉE. Toutes les bases sont générées dans : {OUTPUT_BASE_DIR}")

if __name__ == "__main__":
    main()
