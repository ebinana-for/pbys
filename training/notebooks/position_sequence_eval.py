# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: venv
#     language: python
#     name: python3
# ---

# %%
import os

import matplotlib.pyplot as plt
import pandas as pd

# %%
from src.posture import sequence_evaluation

# %%
INPUT_FOLDER = "/mnt/data/OoP/data/source_code/training/eval_cpp_assets"

# %%
labels = sequence_evaluation.read_labels(INPUT_FOLDER)
labels

# %%
list_of_sequences = sequence_evaluation.read_data(INPUT_FOLDER, labels)
all_gt = pd.concat([x[1] for x in list_of_sequences], ignore_index=True)
all_pred = pd.concat([x[0] for x in list_of_sequences], ignore_index=True)
len(list_of_sequences)

# %%
sequence_evaluation.plot_start_end_annotated_segments(
    all_gt,
    all_pred,
    labels,
    sliding_window=10,
    threshold=0.5,
    min_pos_length=25,
)

# %%
results_dict = sequence_evaluation.evaluate_sliding_window(
    all_gt,
    all_pred,
    labels,
    sliding_window=10,
    threshold=0.5,
    detection_warmup=10,
    min_pos_length=25,
)

# %%
sequence_evaluation.print_table_results(results_dict)

# %%
sequence_evaluation.plot_start_end_annotated_segments(
    all_gt, all_pred, labels=labels, sliding_window=10
)

# %%
results_dict["closetoairbag"]

# %%
sequence_evaluation.print_out_perf(results_dict)

# %%
sequence_evaluation.print_table_results(results_dict)

# %%
var_threshold_results_dict = sequence_evaluation.evaluate_varying_threshold(
    all_gt,
    all_pred,
    labels,
    step=0.05,
)

# %%
sequence_evaluation.plot_varying_thresholds(var_threshold_results_dict)

# %% [markdown]
#

# %% [markdown]
#
