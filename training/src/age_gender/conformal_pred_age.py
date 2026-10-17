import os
import numpy as np
import argparse
import tqdm
import torch
from data.get_dataloader import get_eval_loader
from utils.config import get_config
from train_multi_heads import load_model
import matplotlib.pyplot as plt
import cv2

import pdb

def conformal_calibration(config, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Calibration data
    eval_loader = get_eval_loader(config)

    # Load model
    heads_param = []
    for key, val in config.model.heads_params.items():
        head = val.copy()
        head["name"] = key
        heads_param.append(head)

    age_head_idx = [elt["name"]=="age" for elt in heads_param].index(True)
    model = load_model(config)
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k.removeprefix('module.'): v for k, v in pretrained_dict.items()}
    model_dict = model.state_dict()
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.to(device)
    model.eval()

    # Infer the eval dataset
    dataset_pbar = tqdm.tqdm(
        eval_loader.dataloaders.items(),
        total=len(eval_loader.dataloaders.keys()),
        colour="green",
        leave=None
    )
    
    # evaluate per dataset
    age_pred_full = []
    age_label_full = []
    with torch.no_grad():
        for dataset_name, dataloader in dataset_pbar:
            dataset_pbar.set_description(f"Dataset evaluation '{dataset_name}'")
            for image_tensors, gender_labels, age_labels, _, _ in dataloader:
                # gender_labels = gender_labels.to(device)
                age_labels = age_labels.numpy().tolist()
                preds = model(image_tensors.to(device))
                # age_pred = (preds[age_head_idx]).sigmoid()
                age_preds = preds[age_head_idx]
                age_preds = torch.nn.functional.softmax(age_preds, dim=1).cpu().numpy().tolist()
                age_label_full += age_labels
                age_pred_full += age_preds
                
            ### DEBUG ###
            # break
            #############

    age_label_full = np.array(age_label_full)
    age_pred_full = np.array(age_pred_full)

    # Split the softmax scores into calibration and validation sets (save the shuffling)
    cal_nb = int(len(age_label_full) * 0.5)
    cal_val_idx = np.array([1] * cal_nb + [0] * (age_pred_full.shape[0]-cal_nb)) > 0
    np.random.seed(0)
    np.random.shuffle(cal_val_idx)
    cal_smx, val_smx = age_pred_full[cal_val_idx,:], age_pred_full[~cal_val_idx,:]
    cal_labels, val_labels = age_label_full[cal_val_idx], age_label_full[~cal_val_idx]
    
    alpha = 0.2 # 1-alpha is the desired coverage
    qhat_global, qhat_per_cat = compute_quantile(cal_smx, cal_labels, alpha)

    prediction_sets = val_smx >= (1-qhat_global) # 3: form prediction sets
    prediction_sets_per_cat = val_smx >= (1-qhat_per_cat)

    # Calculate empirical coverage
    mean_empirical_coverage = prediction_sets[np.arange(prediction_sets.shape[0]),np.argmax(val_labels, axis=1)].mean()
    mean_empirical_coverage_per_cat = prediction_sets_per_cat[np.arange(prediction_sets.shape[0]),np.argmax(val_labels, axis=1)].mean()
    print(f"The empirical coverage is: {mean_empirical_coverage}")

    # Calculate empirical coverage per category. Put None if the category is not represented
    empirical_coverage = []
    empirical_coverage_per_cat = []
    max_age = heads_param[age_head_idx]["out_dim"]
    for label in range(max_age):
        prediction_sets0 = prediction_sets[np.argmax(val_labels, axis=1)==label,:]
        prediction_sets_per_cat0 = prediction_sets_per_cat[np.argmax(val_labels, axis=1)==label,:]
        if len(prediction_sets0)>0:
            empirical_coverage.append(prediction_sets0[np.arange(prediction_sets0.shape[0]),label].mean())
        else:
            empirical_coverage.append(np.nan)

        if len(prediction_sets_per_cat0)>0:
            empirical_coverage_per_cat.append(prediction_sets_per_cat0[np.arange(prediction_sets_per_cat0.shape[0]),label].mean())
        else:
            empirical_coverage_per_cat.append(np.nan)

    # Plot histogram of coverage
    bin_edges = [ii for ii in range(max_age)]
    compute_coverage_histograms(bin_edges, 
                                empirical_coverage,
                                "Coverage with global quantile - mean = {0:.1f}%".format(mean_empirical_coverage*100), 
                                'CP_coverage_global_quantile{0:.2f}.png'.format(1-alpha))
    
    compute_coverage_histograms(bin_edges, 
                                empirical_coverage_per_cat,
                                "Coverage with per-class quantile - mean = {0:.1f}%".format(mean_empirical_coverage_per_cat*100),
                                'CP_coverage_per_class_quantile{0:.2f}.png'.format(1-alpha))
    
    # Plot histogram of prediction set sizes according to ages
    empirical_set_sizes_by_age = []
    empirical_set_sizes_by_age_per_cat = []
    for label in range(max_age):
        prediction_sets0 = prediction_sets[np.argmax(val_labels, axis=1)==label,:]
        prediction_sets_per_cat0 = prediction_sets_per_cat[np.argmax(val_labels, axis=1)==label,:]
        if len(prediction_sets0)>0:
            empirical_set_sizes_by_age.append(np.mean(np.sum(prediction_sets0, axis=1)))
        else:
            empirical_set_sizes_by_age.append(np.nan)
        if len(prediction_sets_per_cat0)>0:
            empirical_set_sizes_by_age_per_cat.append(np.mean(np.sum(prediction_sets_per_cat0, axis=1)))
        else:
            empirical_set_sizes_by_age_per_cat.append(np.nan)


    compute_coverage_histograms(bin_edges, empirical_set_sizes_by_age, 
                                "Average set size with global quantile.",
                                'CP_set_size_quantile{0:.2f}.png'.format(1-alpha), ylim=70)
    compute_coverage_histograms(bin_edges, empirical_set_sizes_by_age_per_cat, 
                                "Average set size with per-class quantile.",
                                'CP_set_size_per_class_quantile{0:.2f}.png'.format(1-alpha), ylim=70)



    # Plot some examples
    img_idx = 0
    plot_cnt = 0
    plot_max_cnt = 50
    with torch.no_grad():
        for dataset_name, dataloader in dataset_pbar:
            dataset_pbar.set_description(f"Dataset evaluation '{dataset_name}'")
            for image_tensors, gender_labels, age_labels in dataloader:
                age_labels = age_labels.numpy().tolist()
                preds = model(image_tensors.to(device))
                # age_preds =  preds[age_head_idx].sigmoid()
                age_preds = preds[age_head_idx]
                age_preds = torch.nn.functional.softmax(age_preds, dim=1).cpu().numpy().tolist()
                age_preds = np.array(age_preds)
                prediction_sets_per_cat = age_preds >= (1-qhat_per_cat)
                for img, age_label, age_pred in zip(image_tensors, age_labels, prediction_sets_per_cat):
                    if cal_val_idx[img_idx]==False: # validation image
                        img = img.cpu().numpy()*255
                        pred_set = [ii for ii, val in enumerate(age_pred) if val]
                        out_path = "pred_sample_{0:05d}.png".format(plot_cnt)
                        plot_sample(img.transpose((1,2,0)), np.argmax(age_label), pred_set, out_path)
                        
                        if plot_cnt==plot_max_cnt:
                            break
                        plot_cnt += 1
                    img_idx += 1
                if plot_cnt==plot_max_cnt:
                    break
            if plot_cnt==plot_max_cnt:
                break          
                    

def compute_quantile(cal_smx, cal_labels, alpha):
    # 1: get conformal scores. n = calib_Y.shape[0]
    cal_nb = len(cal_smx)
    cal_scores = 1-cal_smx[np.arange(cal_nb),np.argmax(cal_labels, axis=1)]
    
    # 2: get adjusted quantile
    q_level = np.ceil((cal_nb+1)*(1-alpha))/cal_nb
    qhat_global = np.quantile(cal_scores, q_level, interpolation='higher')

    # compute qhat per category
    qhat_per_cat = []
    cat_ids = [ii for ii in range(cal_smx.shape[1])]
    for cat_id in cat_ids:
        cal_scores_cat = cal_scores[np.argmax(cal_labels, axis=1)==cat_id]
        if len(cal_scores_cat)==0:
            qhat_cat = 0.0
        else:
            cal_nb_cat = len(cal_scores_cat)
            q_level_cat = np.minimum(0.98, np.ceil((cal_nb_cat+1)*(1-alpha))/cal_nb_cat)
            qhat_cat = np.quantile(cal_scores_cat, q_level_cat, interpolation='higher')
        qhat_per_cat.append(qhat_cat)
    qhat_per_cat = np.array(qhat_per_cat)
    qhat_per_cat[qhat_per_cat==0.0] = np.max(qhat_per_cat)

    return qhat_global, qhat_per_cat

def compute_coverage_histograms(bin_edges, hist, title, out_path, ylim=1):
    fig, ax = plt.subplots()
    plt.bar(bin_edges, hist, width=0.7, align='center', color='b')
    plt.grid()
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Coverage")
    ax.set_ylim(0,ylim)
    plt.savefig(out_path)

def plot_sample(img, label, prediction_set, out_path):

    # resize to 680*680 for lisibility, and set to 3 channels
    img_resize = cv2.resize(img, (680,680), interpolation=cv2.INTER_LINEAR)
    if len(img_resize.shape)<3:
        img_resize = cv2.cvtColor(img_resize,cv2.COLOR_GRAY2BGR)

    txt1 = "GT = {}".format(label)
    txt2 = "{}".format(prediction_set)
    img_resize = cv2.putText(img_resize, txt1, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
    img_resize = cv2.putText(img_resize, txt2, (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0,0,255), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, img_resize) 



if __name__ == "__main__":
    # Get args
    parser = argparse.ArgumentParser("parser")
    parser.add_argument("--config_file", required=True, help="Path to input config.")
    parser.add_argument("--ckpt_file", required=True, help="Path to input trained model.")
    args = parser.parse_args()

    # Get config
    config = get_config()
    config.set_config(args.config_file)

    # Sanity check
    model_file = args.ckpt_file
    assert os.path.isfile(model_file), f"Model file {model_file} not found."
    assert 'age' in config.model.heads_params.keys(), "age head not found in the model."
    assert 'BCE' in config.model.heads_params.age.loss, "age head must a classifier, not a regressor."

    conformal_calibration(config, model_file)