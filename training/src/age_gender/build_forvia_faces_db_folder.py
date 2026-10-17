from pathlib import Path
import subprocess


BASE_DIR = Path("/mnt/data/OoP/data/DATA_HEIGHTWEIGHT")
OUT_DIR = BASE_DIR / "faces_db"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def iter_valid_pairs(base_dir: Path):
    for mp4 in sorted(base_dir.glob("*.mp4")):
        csv_path = mp4.with_suffix(".csv")

        if not csv_path.exists():
            print(f"SKIP {mp4.name}: no CSV with the same name")
            continue

        if csv_path.name.endswith("_out_age.csv"):
            print(f"SKIP {mp4.name}: _out_age.csv ignored")
            continue

        yield mp4, csv_path


def main():
    images_dir = OUT_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for mp4, csv_path in iter_valid_pairs(BASE_DIR):
        print(f"TRAITEMENT: {mp4.name}")
        print(f"  CSV: {csv_path.name}")
        print(f"  OUT: {OUT_DIR}")

        subprocess.run(
            [
                "python",
                "/mnt/data/OoP/data/source_code/git-python/training/src/age_gender/build_forvia_faces_db.py",
                "--video_path",
                str(mp4),
                "--csv_file",
                str(csv_path),
                "--out_path",
                str(OUT_DIR),
                "--task",
                "train",
            ],
            check=True,
        )

    print(f"\nDONE: images saved in {images_dir}")
    csv_file = OUT_DIR / "forvia_face_train.csv"
    if csv_file.exists():
        print(f"CSV final: {csv_file}")


if __name__ == "__main__":
    main()
