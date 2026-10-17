import os
import argparse
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from src.posture.dataset.posture_dataset import PostureDataset
import progressbar

import pdb


def main():
    parser = argparse.ArgumentParser("Posture video inference")
    parser.add_argument(
        '-c', "--model_config", required=False, 
        help="Path to input posture model config"
     )
    parser.add_argument(
        '-m', "--model_path", required=False, 
        help="Path to input posture model path"
     )
    parser.add_argument(
        '-o', "--out_path", required=False, 
        help="Path to store output video"
    )
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.model_config and args.model_path:
        from src.posture.train_posture import load_model
        from src.utils.config import get_config

        config_file = args.model_config
        config = get_config()
        config.set_config(str(config_file))
        trained_state_dict = torch.load(args.model_path)
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
            trained_state_dict, "module."
        )

        model = load_model(config)
        model.load_state_dict(trained_state_dict)
        model.to(device)
        model.eval()

    # Create dataloader
    # train_dataset = PostureDataset(config, task="TRAIN")
    # sequence_df_train = train_dataset.img_df.copy()
    # sequence_df_train['seq_name'] = sequence_df_train['file'].apply(lambda x: os.path.basename(os.path.dirname(x)))
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    eval_dataset = PostureDataset(config, task="TEST")
    sequence_df_eval = eval_dataset.img_df.copy()
    sequence_df_eval["seq_name"] = sequence_df_eval["file"].apply(
        lambda x: os.path.basename(os.path.dirname(x))
    )
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    infer(model, eval_loader, sequence_df_eval, device, out_path=args.out_path)


def infer(model, dataloader, sequence_df, device, out_path=None):

    if out_path:
        out_dir = out_path
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        _, _, _, seq_name0, _ = dataloader.dataset[0]
        out_file = seq_name0 + "_posture_viewer.mp4"
        fps = 23
        size = (2560 // 2, 1984 // 2)
        result = cv2.VideoWriter(
            os.path.join(out_dir, out_file), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
        )

    # Build label mapping
    num_positive_labels = len(dataloader.dataset.label_map)
    label_map = ["correctly_seated"] + [
        dataloader.dataset.label_map[key] for key in range(num_positive_labels)
    ]
    # label_map = {key + 1: val for key, val in dataloader.dataset.label_map.items()}
    # label_map[0] = "correctly_seated"

    dataloader_pbar = tqdm(
        dataloader, total=len(dataloader), colour="green", leave=None
    )
    with torch.no_grad():
        cnt = 0
        for (
            image_tensors,
            posture_labels,
            posture_kpts,
            seq_name,
            frame_idx,
        ) in dataloader_pbar:
            labels = np.squeeze(posture_labels.numpy())
            if model.n_inputs == 2:
                pred = model(image_tensors.to(device), posture_kpts.to(device))
            else:
                pred = model(image_tensors.to(device))
            pred = np.squeeze(torch.sigmoid(pred).cpu().numpy())
            pred_class = np.zeros((len(pred) + 1), dtype=np.float64)
            pred_class[1:] = (pred > 0.5).astype(np.float64)
            pred_class[0] = 1.0 if np.sum(pred_class[1:]) == 0 else 0.0

            # Load input image
            img_path = dataloader.dataset.img_df.iloc[cnt]["file"]
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

            # Write label and pred in the image
            pred_labels = [
                label_map[ii] for ii, val in enumerate(pred_class) if val == 1
            ]
            gt_labels = [label_map[ii] for ii, val in enumerate(labels) if val == 1]

            posx = size[0] // 2
            posy = 50
            img = cv2.putText(
                img,
                "Prediction",
                (posx, posy),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            # pdb.set_trace()
            for pred_label in pred_labels:
                posy += 50
                img = cv2.putText(
                    img,
                    pred_label,
                    (posx, posy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.4,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

            posx = 50
            posy = 50
            img = cv2.putText(
                img,
                "Ground Truth",
                (posx, posy),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            for gt_label in gt_labels:
                posy += 50
                img = cv2.putText(
                    img,
                    gt_label,
                    (posx, posy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.4,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

            img = cv2.putText(
                img,
                str(frame_idx.item()),
                (size[0] - 100, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            if out_path:
                if seq_name[0] != seq_name0:
                    result.release()
                    out_file = seq_name[0] + "_posture_viewer.mp4"
                    result = cv2.VideoWriter(
                        os.path.join(out_dir, out_file),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        size,
                    )
                    seq_name0 = seq_name[0]
                result.write(img)

            cnt += 1
            ##### DEBUG #####
            # if cnt==100:
            #     break
            #################

    if out_path:
        result.release()



if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
