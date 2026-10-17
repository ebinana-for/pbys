"""Diagnostic : compare les frames extraites d'une vidéo .mp4 aux JPG sources
correspondants du dataset (même séquence), pour vérifier :
    1. Le nombre total de frames (vidéo vs CSV tracker vs JPG annotés)
    2. L'alignement d'index (frame i vidéo == frame i JPG, ou décalage ?)
    3. L'ampleur de la dégradation pixel (compression, si alignement OK)

Usage:
    python diagnose_frame_alignment.py \
        --video_path=/mnt/.../thibault_vincent_28_M_178_68.mp4 \
        --jpg_dir=/mnt/data/OoP/data/private_datasets/.../images/thibault_vincent_28_M_178_68 \
        --tracker_csv=/mnt/.../thibault_vincent_28_M_178_68.csv \
        --frame_indices=0,50,100,200
"""

import os
import glob
import cv2
import numpy as np
import pandas as pd
from fire import Fire


def get_jpg_for_frame(jpg_dir, frame_index):
    """Retrouve le JPG correspondant à un frame_index (mêmes 6 derniers
    chiffres du nom de fichier que dans posture_dataset.py)."""
    candidates = glob.glob(os.path.join(jpg_dir, "*.jpg"))
    for c in candidates:
        base = os.path.splitext(os.path.basename(c))[0]
        try:
            idx = int(base[-6:])
        except ValueError:
            continue
        if idx == frame_index:
            return c
    return None


def main(video_path, jpg_dir, tracker_csv=None, frame_indices="0,50,100"):
    frame_indices = [int(x) for x in str(frame_indices).split(",")]

    # --- 1. Comptages globaux ---
    cap = cv2.VideoCapture(video_path)
    n_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_jpg = len(glob.glob(os.path.join(jpg_dir, "*.jpg")))
    print(f"Frames vidéo (CAP_PROP_FRAME_COUNT) : {n_video_frames}")
    print(f"JPG dans le dossier séquence        : {n_jpg}")
    if tracker_csv and os.path.isfile(tracker_csv):
        n_csv = len(pd.read_csv(tracker_csv))
        print(f"Lignes CSV tracker                  : {n_csv}")
    print()

    # --- 2. Comparaison pixel à pixel sur quelques frames ---
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"[frame {idx}] impossible de lire la frame vidéo.")
            continue
        video_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        jpg_path = get_jpg_for_frame(jpg_dir, idx)
        if jpg_path is None:
            print(f"[frame {idx}] pas de JPG correspondant trouvé (frame_index={idx}).")
            continue
        jpg_gray = cv2.imread(jpg_path, cv2.IMREAD_GRAYSCALE)

        if jpg_gray.shape != video_gray.shape:
            print(f"[frame {idx}] TAILLES DIFFÉRENTES : video={video_gray.shape} jpg={jpg_gray.shape}")
            continue

        diff = cv2.absdiff(video_gray, jpg_gray)
        mean_diff = float(np.mean(diff))
        max_diff = int(np.max(diff))

        # Sauvegarde une image côte à côte pour inspection visuelle
        side_by_side = np.hstack([video_gray, jpg_gray, diff])
        out_path = f"/tmp/frame_compare_{idx}.png"
        cv2.imwrite(out_path, side_by_side)

        print(f"[frame {idx}] JPG={os.path.basename(jpg_path)} | "
              f"diff moyenne={mean_diff:.2f} | diff max={max_diff} | "
              f"comparaison sauvegardée -> {out_path}")

        # Un écart moyen > quelques unités (sur 0-255) avec une structure
        # cohérente (pas juste du bruit fin) suggère un décalage d'index,
        # pas juste de la compression. Regarde surtout le PNG de diff :
        # bruit fin uniforme = compression ; silhouette nette = désalignement.

    cap.release()


if __name__ == "__main__":
    Fire(main)