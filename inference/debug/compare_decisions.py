#!/usr/bin/env python3
"""Compare per-frame posture decisions of a candidate model against a reference.

Both inputs are directories of <video>_predictions.csv files in the format written by
py_inference_v2_same_result_C_kpi_golden_ref_multiple_video.py (columns: index + 8 classes).
FOMS C++ *_out_positions.csv files are accepted too. Frames are matched on "index";
frames where either side is -1 (no tracking box) are skipped.

Typical use: run the inference script twice on the same videos, once with the fp32 ONNX
and once with --model_path <TIDL artifacts dir> (DLR), then:

    python3 compare_decisions.py <fp32_dir> <int8_dir>

Reports, per class and overall:
  - decision agreement at the threshold (default 0.5)
  - flips among frames the reference calls positive (second denominator)
  - share of flips whose reference probability is within 0.1 of the threshold:
    high = flips are borderline rounding; low = the candidate is shifted overall
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

CLASSES = ["closetoairbag", "legsondashboard", "torsodissimetry", "reachingback",
           "lateralbodycontact", "oneshoulderincontact", "pelvisrotation", "pelvisdrift"]


def find(d, video):
    for suffix in ("_predictions.csv", "_out_positions.csv"):
        p = os.path.join(d, video + suffix)
        if os.path.isfile(p):
            return p
    return None


def main(ref_dir, cand_dir, threshold=0.5):
    refs = sorted(glob.glob(os.path.join(ref_dir, "*_predictions.csv")) + glob.glob(os.path.join(ref_dir, "*_out_positions.csv")))
    videos = sorted({os.path.basename(p).rsplit("_predictions.csv", 1)[0].rsplit("_out_positions.csv", 1)[0] for p in refs})
    R, C = [], []
    for v in videos:
        cp = find(cand_dir, v)
        if cp is None:
            print(f"skip {v}: no candidate file")
            continue
        r = pd.read_csv(find(ref_dir, v)).set_index("index")[CLASSES]
        c = pd.read_csv(cp).set_index("index")[CLASSES]
        idx = r.index.intersection(c.index)
        r, c = r.loc[idx], c.loc[idx]
        keep = ((r != -1).all(axis=1) & (c != -1).all(axis=1)).to_numpy()
        R.append(r.to_numpy()[keep])
        C.append(c.to_numpy()[keep])
        print(f"{v}: {keep.sum()} frames compared")
    r, c = np.concatenate(R), np.concatenate(C)
    dr, dc = r >= threshold, c >= threshold
    flip = dr != dc
    near = np.abs(r - threshold) < 0.1
    print(f"\n{'class':22s} {'agree%':>8s} {'flips':>6s} {'ref_pos':>7s} {'flips_on_pos':>12s} {'near%':>6s} {'mean|dp|':>9s}")
    for j, name in enumerate(CLASSES):
        f = flip[:, j]
        print(f"{name:22s} {100 * (1 - f.mean()):8.2f} {f.sum():6d} {dr[:, j].sum():7d} {(f & dr[:, j]).sum():12d} "
              f"{100 * (f & near[:, j]).sum() / max(1, f.sum()):6.1f} {np.abs(r[:, j] - c[:, j]).mean():9.4f}")
    print(f"\nframes {len(r)} x {len(CLASSES)} classes: agreement {100 * (1 - flip.mean()):.2f}%, flips {flip.sum()}, "
          f"flips near threshold {100 * (flip & near).sum() / max(1, flip.sum()):.1f}% "
          f"(reference near threshold overall: {100 * near.mean():.1f}%)")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], *(float(x) for x in sys.argv[3:4]))
