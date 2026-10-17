#!/usr/bin/env python3
"""
Collect videos, CSVs and label‑files from multiple test datasets.

Usage:
    python gather_assets.py path/to/config.yaml path/to/output_root
"""

import argparse
import shutil
from pathlib import Path

import yaml


def load_cfg(cfg_path: Path) -> dict:
    with cfg_path.open() as f:
        return yaml.safe_load(f)


def collect_assets(cfg: dict, out_root: Path) -> None:
    test_datasets = cfg["input"]["test"]

    for ds_yaml_name, ds_info in test_datasets.items():
        labels_dir = Path(ds_info["annot_path"])  # …/test/labels
        ds_root = labels_dir.parents[1]  # …/private_datasets/<dataset>
        ds_name = ds_root.name
        sequences = [p.name for p in labels_dir.iterdir() if p.is_dir()]

        print(f"[{ds_yaml_name}: {ds_name}]  {len(sequences)} sequences")

        dst_dir = out_root
        dst_dir.mkdir(parents=True, exist_ok=True)
        for seq in sequences:
            # ----- 1.  Video (.mpeg → .mp4) -------------------------------
            src_video = ds_root / "mpeg" / f"{seq}.mp4"
            dst_video = dst_dir / f"{seq}.mp4"

            if src_video.exists():
                shutil.copy2(src_video, dst_video)
            else:
                print(f"  ! missing video {src_video}")

            # ----- 2.  Keypoints CSV ---------------------------------------
            src_csv = ds_root / "raw" / f"{seq}.sec" / "log" / "tracking_000000.csv"
            dst_csv = dst_dir / f"{seq}.csv"

            if src_csv.exists():
                shutil.copy2(src_csv, dst_csv)
            else:
                print(f"  ! missing CSV   {src_csv}")

            # ----- 3.  Labels (.txt files) --------------------------------
            src_label_seq = labels_dir / seq
            dst_label_seq = dst_dir / "labels" / seq

            if src_label_seq.exists():
                # copytree requires dst not to exist
                if dst_label_seq.exists():
                    shutil.rmtree(dst_label_seq)
                shutil.copytree(src_label_seq, dst_label_seq)
            else:
                print(f"  ! missing label folder {src_label_seq}")

        # ----- 4.  Single synsets.txt file --------------------------------
        src_txt = labels_dir / seq / "synsets.txt"
        dst_txt = dst_dir / "synsets.txt"
        if src_csv.exists():
            shutil.copy2(src_txt, dst_txt)
        else:
            print(f"  ! missing synsets.txt   {src_txt}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=Path,
        help="YAML config file of the model with the description of the datasets",
    )
    parser.add_argument("output_root", type=Path, help="Where to save the assets")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    collect_assets(cfg, args.output_root.absolute())


if __name__ == "__main__":
    main()
