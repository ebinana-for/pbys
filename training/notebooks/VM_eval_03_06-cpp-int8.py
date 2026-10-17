# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# This notebook reads three different files from evaluation of the 3 different public datasets and produces various plots and statistics on these predictions.

# %%
import pandas as pd
import plotly
import os
import numpy as np

# %% [markdown]
# ### Gather several files from 3 different folders and load them

# %%
folder = "../../forvia-oms-library/data/public_eval_datasets/utkface"
print(os.listdir(folder))


# %%
def gather_pred_gt_data(pred_filepath, dataset_name):
    pred = pd.read_csv(pred_filepath)
    pred.drop(columns=[x for x in pred.columns if '_mean' in x]+["index"], inplace=True, )
    pred.rename(columns={'age': 'age_pred', 'gender': 'gender_pred'}, inplace=True)

    gt_filepath = pred_filepath.split('_out_age.csv')[0] + '_agegender.csv'
    gt = pd.read_csv(gt_filepath)
    gt.drop(columns=['path'], inplace=True)
    gt.rename(columns={'age': 'age_gt', 'gender': 'gender_gt'}, inplace=True)
    gt['dataset'] = dataset_name

    merged = pd.merge(gt, pred, left_index=True, right_index=True)
    return merged 


# %%
folder = "../../forvia-oms-library/data/public_eval_datasets/"

utkface = gather_pred_gt_data(os.path.join(folder, 'utkface', 'UTK-Face_out_age.csv'), "UTK-Face")
lagenda = gather_pred_gt_data(os.path.join(folder, 'lagenda', 'LAGENDA_out_age.csv'), "LAGENDA")
imdb = gather_pred_gt_data(os.path.join(folder, 'imdb', 'IMDB_out_age.csv'), "IMDB")

df = pd.concat([utkface, lagenda, imdb])
df


# %% [markdown]
# ### Pretreat the dataframe

# %% [markdown]
# **Treat invalid crops per dataset**

# %%
def bin_age(age, bins):
    """
    Arguments:
        age: pd.Series
        bins: List[int] - list of middle points between the age categories
    
    Returns: 
        categories: pd.Series()
    """
    names = [f"0-{bins[0]}"] + [f"{bins[x]+1}-{bins[x+1]}" for x in range(len(bins)-1)] + [f"{bins[-1]+1}+"]
    if bins[0] != 0:
        bins = [0] + bins
    bins.append(np.inf)

    return pd.cut(age, bins, labels=names)


# %%
df

# %%
bins = [2, 18, 35, 65]
df["Age category"] = bin_age(df.age_gt, bins)
df["Age category"].value_counts()

# %%
df['Age Error'] = df.age_pred - df.age_gt
df['Age Average Error'] = (df.age_pred - df.age_gt).abs()
df['Gender accuracy'] = df.gender_pred == df.gender_gt
df

# %%
df.dataset.value_counts()

# %% [markdown]
# ## Filtering of the proper instances

# %%
df[df.age_pred == -1].dataset.value_counts()

# %%
mask = (df.age_pred != -1) & (df.dataset.isin(["UTK-Face", "IMDB"]))
df = df.loc[mask, :]

# %%
df.sort_values('Age category', inplace=True)

# %% [markdown]
# ## Output

# %%
import plotly.express as px
title = "Age prediction, Int-8 model, per age category on UTK-Face and IMDB datasets"
fig = px.box(df.sort_values('Age category'), x="Age category", y='Age Average Error', color='dataset', width=800, height=600, title=title)
fig.update_traces(boxmean=True)
fig.update_yaxes(range=[0, 20])
fig.add_hline(y=8, line_width=2, line_dash="dash")
fig.show()

# %%
import plotly.express as px
title = "Age prediction, Int8 model, per age category on UTK-Face and IMDB datasets"
fig = px.box(df.sort_values('Age category'), x="Age category", y='Age Average Error', width=600, height=600)
fig.update_traces(boxmean=True)
fig.update_yaxes(range=[0, 20])
fig.add_hline(y=8, line_width=2, line_dash="dash")
fig.update_layout(
    title=dict(text=title, font=dict(size=14))
)
fig.show()

# %%
import plotly.express as px
title = "Age prediction, Int8 model, per age category on UTK-Face and IMDB datasets"
fig = px.box(df[df['Age category'] != '65+'].sort_values('Age category'), y='Age Average Error', width=600, height=600)
fig.update_traces(boxmean=True)
fig.update_yaxes(range=[0, 20])
fig.add_hline(y=8, line_width=2, line_dash="dash")
fig.update_layout(
    title=dict(text=title, font=dict(size=14))
)
fig.show()

# %%

# %%
df.groupby('Age category')["Gender accuracy"].mean().plot.bar(ylabel='Gender prediction accuracy', ylim=[0.5, 1.0])

# %%
df.groupby('Age category')["Gender accuracy"].mean()

# %%
df["Gender accuracy"].mean()

# %%
df['Age category'].value_counts()

# %%
df.groupby('Age category')["Age Error"].std()

# %%
