from pathlib import Path
import subprocess

base = Path("/mnt/data/OoP/data/DATA_HEIGHTWEIGHT")

for mp4 in sorted(base.glob("*.mp4")):
    csv = mp4.with_suffix(".csv")
    if not csv.exists():
        continue

    out_dir = base / "faces_db" / mp4.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "python",
            "/mnt/data/OoP/data/source_code/git-python/training/src/age_gender/build_forvia_faces_db.py",
            "--video_path", str(mp4),
            "--csv_file", str(csv),
            "--out_path", str(out_dir),
            "--task", "train",
        ],
        check=True,
    )