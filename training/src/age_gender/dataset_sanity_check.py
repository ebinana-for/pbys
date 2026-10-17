import os
import pandas as pd
from tqdm import tqdm
import torch
import cv2
import numpy as np
from train_multi_heads import load_model
from src.age_gender.dataset.get_dataloader import get_train_loader, get_eval_loader
from utils.config import get_config
from src.utils.data_augmentations import DataAugPipeline

import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(model_config, model_path):
    model = load_model(model_config)
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k.removeprefix('module.'): v for k, v in pretrained_dict.items()}
    model_dict = model.state_dict()
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.to(device)
    model.eval()

    heads_param = []
    for key, val in model_config.model.heads_params.items():
        head = val.copy()
        head["name"] = key
        heads_param.append(head)
    age_head_idx = [elt["name"]=="age" for elt in heads_param].index(True)

    return model, age_head_idx


def run(config, model_config, model_path):
    model, age_head_idx = get_model(model_config, model_path)
    
    data_aug_list = []
    for name, parameters in model_config.input.data_augmentations["test"].items():
        data_aug_list.append(DataAugPipeline(name,**parameters))
    with torch.no_grad():
        for task, task_params in config.input.items():
            for name_of_dataset, dataset_params in task_params.items():
                annotation_file = dataset_params.annot_path
                img_root_dir = dataset_params.img_path

                # Load annotations
                df_label = pd.read_csv(annotation_file)

                # Add a sanity column
                df_label["sanity"] = 1
                df_label["age_pred"] = -1

                for [index, row] in tqdm(df_label.iterrows(),desc=f"Processing {name_of_dataset} / {task}"):
                    img_path = os.path.join(img_root_dir, row['img_name'])
                    try:
                        img = cv2.imread(img_path)
                        # face box coords
                        xmin = int(row.face_x0)
                        ymin = int(row.face_y0)
                        xmax = int(row.face_x1)
                        ymax = int(row.face_y1)
                        box = [xmin,ymin,xmax,ymax]
                        if img is not None and len(img.shape)>1 and img.shape[0]>0 and img.shape[1]>0 and box!=[-1,-1,-1,-1]:
                            face_crop = img
                            for data_aug in data_aug_list:
                                face_crop = data_aug(face_crop, box)
                            # Convert to gray levels, one channel
                            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)[..., None]
                            face_crop = face_crop.transpose((2, 0, 1))
                            face_crop = face_crop[None, ...]
                            face_crop = np.ascontiguousarray(face_crop)
                            face_crop = face_crop/255.0  # normalization
                            input_tensor = torch.from_numpy(face_crop.astype(np.float32))
                            preds = model(input_tensor.to(device))
                            age_preds = preds[age_head_idx]
                            age_preds = torch.nn.functional.softmax(age_preds, dim=1).cpu().numpy().tolist()
                            age_pred_mean = np.sum(age_preds[0] * np.arange(0, len(age_preds[0]), 1))
                            df_label.loc[index,"age_pred"] = int(age_pred_mean+0.5)
                            # age_gt = int(row.age)
                        else:
                            # pdb.set_trace()
                            df_label.loc[index,"sanity"] = 0
                            print(f"Image {img_path} corrupted")

                    except: # Something wrong
                        df_label.loc[index,"sanity"] = 0
                        print(f"Image {img_path} is not inferencable by model")
                    
                # Write updated dataframe
                annotation_file_new = annotation_file[:-4] + "_sanity_check1" + annotation_file[-4:]
                df_label.to_csv(annotation_file_new)
                print(f"File {annotation_file_new} written.")


def load_annotations(annot_path):
        df = pd.read_csv(annot_path)

        # drop all rows that contain -1
        df = df.drop(df[df['age'] == -1].index)
        df = df.drop(df[df['gender'] == -1].index)
        df = df.drop(df[df['age'] == "-1"].index)
        df = df.drop(df[df['gender'] == "-1"].index)

        return df

if __name__ == "__main__":
    from src import PROJECT_DIR
    # # Get args
    # parser = argparse.ArgumentParser("parser")
    # parser.add_argument("--config_file", required=True, help="Path to input config")
    # args = parser.parse_args()

    config = get_config()
    model_config = get_config()
    config_file = PROJECT_DIR / "config/dataset_sanity_check_config.yaml"
    model_config_file = "/mnt/data/jdefretin/Forvia/saved_models/2024-12-30T09h58m52s/config.yaml"
    model_file = "/mnt/data/jdefretin/Forvia/saved_models/2024-12-30T09h58m52s/ckpt/epoch_95.pth"
    config.set_config(str(config_file))
    model_config.set_config(str(model_config_file))

    assert 'BCE' in model_config.model.heads_params.age.loss, "age head must a classifier, not a regressor."

    run(config, model_config, model_file)
