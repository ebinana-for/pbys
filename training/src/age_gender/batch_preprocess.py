import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Lancer le preprocess sur toutes les paires MP4/CSV."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--script", required=True,
                        help="Chemin vers ton script de preprocess existant")
    parser.add_argument("--task", choices=["train", "test"], default="train")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    script = Path(args.script).resolve()
    out_path = Path(args.out_path).resolve()

    if not input_dir.is_dir():
        parser.error(f"Dossier introuvable : {input_dir}")
    if not script.is_file():
        parser.error(f"Script introuvable : {script}")

    # Recherche également dans les sous-dossiers.
    videos = sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".mp4"
    )

    if not videos:
        print("Aucune vidéo MP4 trouvée.")
        return

    out_path.mkdir(parents=True, exist_ok=True)
    processed = 0
    skipped = 0

    for index, video in enumerate(videos, start=1):
        # Correspondance exacte : video.mp4 -> video.csv
        csv_file = video.with_suffix(".csv")

        if not csv_file.is_file():
            print(f"[{index}/{len(videos)}] IGNORÉ : CSV absent pour {video.name}")
            skipped += 1
            continue

        print(
            f"[{index}/{len(videos)}] Traitement : {video.name}",
            flush=True,
        )

        command = [
            sys.executable,
            str(script),
            "-v", str(video),
            "-l", str(csv_file),
            "-o", str(out_path),
            "-t", args.task,
        ]

        # Traitement séquentiel : ton preprocess modifie un CSV commun.
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as error:
            print(
                f"Échec sur {video.name} (code {error.returncode}). "
                "Arrêt du batch pour éviter de poursuivre sur une sortie partielle.",
                file=sys.stderr,
            )
            sys.exit(error.returncode)

        processed += 1

    print(f"Terminé : {processed} vidéos traitées, {skipped} ignorées.")


if __name__ == "__main__":
    main()
