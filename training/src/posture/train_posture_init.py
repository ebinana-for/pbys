"""Training script for posture classification models"""

import os
import sys
import argparse
import datetime
import logging
import tqdm
import cv2
import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from pprint import pformat
from src.posture.dataset.posture_dataset import PostureDataset
from src.posture.models.mobilenetv2 import (
    MobileNetV2Posture,
    MobileNetV2TrPosture,
    MobileNetV2MlpPosture,
    MobileNetV2PostureMultiHeads,
)
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from torch.utils.data import DataLoader
from src.utils.config import get_config
from src.utils.train_utils import Averager, get_lr
from src.posture.test_posture import eval
from src.utils.losses import Loss

logger = logging.getLogger(__name__)


def load_model(config):
    """
    Load the posture classification model based on the configuration.
    Args:
        config (Config): Configuration object containing model parameters.
    Returns:
        model (torch.nn.Module): The loaded posture classification model.
    """
    head_param = []
    for key, val in config.model.heads_params.items():
        head = val.copy()
        head["name"] = key
        head_param.append(head)

    model_weights = None
    if config.model.name == "mobilenet_v2":
        assert (
            len(head_param) == 1
        ), "You defined {} heads, but {} cannot handle more than 1.".format(
            len(head_param), config.model.name
        )
        model = MobileNetV2Posture(
            head_param,
            in_chans=config.model.input_channels,
            width_mult=config.model.width_multiplier,
        )
        model_weights = MobileNet_V2_Weights.IMAGENET1K_V1
        model_weights = MobileNet_V2_Weights.verify(model_weights)
    elif config.model.name == "mobilenet_v2_tr":
        assert (
            len(head_param) == 1
        ), "You defined {} heads, but {} cannot handle more than 1.".format(
            len(head_param), config.model.name
        )
        model = MobileNetV2TrPosture(
            head_param,
            in_chans=config.model.input_channels,
            width_mult=config.model.width_multiplier,
        )
        model_weights = MobileNet_V2_Weights.IMAGENET1K_V1
        model_weights = MobileNet_V2_Weights.verify(model_weights)
    elif config.model.name == "mobilenet_v2_mlp":
        assert (
            len(head_param) == 1
        ), "You defined {} heads, but {} cannot handle more than 1.".format(
            len(head_param), config.model.name
        )
        model = MobileNetV2MlpPosture(
            head_param,
            in_chans=config.model.input_channels,
            width_mult=config.model.width_multiplier,
        )
        model_weights = MobileNet_V2_Weights.IMAGENET1K_V1
        model_weights = MobileNet_V2_Weights.verify(model_weights)
    elif config.model.name == "mobilenet_v2_multi_head":
        model = MobileNetV2PostureMultiHeads(
            head_param,
            in_chans=config.model.input_channels,
            width_mult=config.model.width_multiplier,
        )
        model_weights = MobileNet_V2_Weights.IMAGENET1K_V1
        model_weights = MobileNet_V2_Weights.verify(model_weights)
    else:
        logger.error(f"Unkown model {config.model}")

    if (
        "pretrained_backbone" in config.model
        and config.model.pretrained_backbone is not None
    ):
        if os.path.isfile(config.model.pretrained_backbone):
            logger.info(
                f"Loading backbone weights from {config.model.pretrained_backbone}"
            )
            pretrained_dict = torch.load(config.model.pretrained_backbone)
            pretrained_backbone = {
                k.removeprefix("module."): v
                for k, v in pretrained_dict.items()
                if "heads" not in k
            }
            model_dict = model.state_dict()
            # overwrite entries in the existing state dict
            model_dict.update(pretrained_backbone)
            model.load_state_dict(model_dict)
        elif config.model.pretrained_backbone == "imagenet":
            if model_weights is not None:
                pretrained_dict = model_weights.get_state_dict(progress=True)
                pretrained_backbone = {
                    k.removeprefix("module."): v
                    for k, v in pretrained_dict.items()
                    if "classifier" not in k
                }
                model_dict = model.state_dict()
                # overwrite entries in the existing state dict
                model_dict.update(pretrained_backbone)
                model.load_state_dict(model_dict)
            else:
                logger.warning(
                    "You requested {} weights, but they are not implemented \
                               for {}".format(
                        config.model.pretrained_backbone, config.model.name
                    )
                )
        else:
            logger.error(
                "Unkown model weights {}".format(config.model.pretrained_backbone)
            )
    return model


def train(config):
    """
    Main training function for posture classification models.
    Args:
        config (Config): Configuration object containing all parameters.
    Returns:
        None
    This function performs the following steps:
    1. Loads the training and evaluation datasets using the PostureDataset class.
    2. Loads the posture classification model based on the configuration.
    3. Sets up the loss function and optimizer.
    4. Runs the training loop for the specified number of epochs.
    5. Evaluates the model on the test dataset after each epoch.
    6. Saves the model checkpoints after each epoch.
    7. Logs training and evaluation metrics to TensorBoard.
    """
    logger.info("Starting training with config: {}".format(config))
    # GPU settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Using device {device}")

    # Data Part
    train_dataset = PostureDataset(config, task="TRAIN")
    test_dataset = PostureDataset(config, task="TEST")
    train_loader = DataLoader(
        train_dataset, batch_size=config.input.batch_size, shuffle=True
    )
    eval_loader = DataLoader(
        test_dataset, batch_size=config.input.batch_size, shuffle=False
    )

    # Model configuration
    model = load_model(config)
    print(model)

    # setup loss
    heads_list = model.head_param
    loss = Loss(config, heads_list)

    # data parallel for multi-GPU
    model_n_inputs = model.n_inputs
    model = torch.nn.DataParallel(model).to(device)

    loss_avg = Averager()

    # Setup optimizer
    scheduler = None
    if config.train.optimizer.upper() == "ADAM":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.train.lr, betas=(config.train.beta1, 0.999)
        )
    elif config.train.optimizer.upper() == "ADAMW":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.train.lr, betas=(config.train.beta1, 0.999)
        )
    elif config.train.optimizer.upper() == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=config.train.lr, momentum=0.9
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
    elif config.train.optimizer.upper() == "ADADELTA":
        optimizer = torch.optim.Adadelta(
            model.parameters(),
            lr=config.train.lr,
            rho=config.train.rho,
            eps=config.train.eps,
        )

    # save ckpt path
    output_dir = os.path.join(config.general.train_dir, "ckpt")
    os.makedirs(output_dir, exist_ok=True)

    epoch = 0

    # initialize TB
    writer = SummaryWriter(config.general.train_dir, filename_suffix="train")
    writer.add_text(
        "config",
        "```yaml\n" + pformat(config, indent=1, sort_dicts=False) + "\n```",
        epoch,
    )

    with torch.no_grad():
        model.eval()
        eval(
            eval_loader,
            model,
            heads_list,
            loss,
            writer,
            epoch,
            device,
            config,
            model_n_inputs=model_n_inputs,
            normalize_cm=True,
        )

    # Train Loop
    while True:
        if epoch == config.train.num_epoch:
            logger.info(f"Maximum number of epoch reached ({epoch}). End of training")
            sys.exit()
        model.train()
        logger.info(f"Start epoch {epoch}/{config.train.num_epoch}")
        train_pbar = tqdm.tqdm(
            train_loader, total=len(train_loader), colour="red", leave=None
        )

        for image_tensors, posture_labels, posture_kpts, *_ in train_pbar:
            posture_labels = posture_labels.to(device)
            if model_n_inputs == 2:
                pred = model(image_tensors.to(device), posture_kpts.to(device))
            else:
                pred = model(image_tensors.to(device))

            # Create a list of pred and gt according to the number of heads
            pred_list = []
            label_list = []
            if len(heads_list) > 1:
                for elt in heads_list:
                    # _, elt = next(iter(head.items()))
                    indices = torch.tensor(elt["labels"]).to(device)
                    pred_head = torch.index_select(pred, 1, indices - 1)
                    pred_list.append(pred_head)

                    label_head = torch.index_select(posture_labels, 1, indices)
                    label_head = label_head.to(device)
                    label_list.append(label_head)
            else:
                pred_list = [pred]
                # Remove 1st column (negative labels, unused for loss)
                label_list = [posture_labels[:, 1:]]

            cost = loss.compute(pred_list, label_list, epoch=epoch)
            model.zero_grad()
            cost.backward()
            optimizer.step()
            loss_avg.add(cost)

        # test procedure
        model.eval()
        with torch.no_grad():
            eval(
                eval_loader,
                model,
                heads_list,
                loss,
                writer,
                epoch,
                device,
                config,
                model_n_inputs=model_n_inputs,
                normalize_cm=True,
            )

            # Display some training images in tensorboard
            images_to_show = image_tensors[: config.eval.nb_imgs_in_tensorboard]

            pred = torch.sigmoid(pred)
            preds_to_show = [
                (torch.where(vec > 0.5)[0].cpu().numpy() + 1).tolist()
                for vec in torch.unbind(
                    pred[: config.eval.nb_imgs_in_tensorboard], dim=0
                )
            ]
            preds_to_show = [[0] if len(elt) == 0 else elt for elt in preds_to_show]
            gt_to_show = [
                torch.where(vec >= 0.5)[0].cpu().numpy().tolist()
                for vec in torch.unbind(
                    posture_labels[: config.eval.nb_imgs_in_tensorboard], dim=0
                )
            ]

            lst_img_final = []
            for image, pred, gt in zip(images_to_show, preds_to_show, gt_to_show):
                image = (np.array(image.cpu()).transpose((1, 2, 0))) * 255
                image = np.ascontiguousarray(image, dtype=np.uint8)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                color = (0, 255, 0)
                image_rgb = cv2.putText(
                    image_rgb,
                    "GT: " + str(gt),
                    (10, 30),
                    cv2.FONT_HERSHEY_PLAIN,
                    1.7,
                    color,
                    2,
                )
                image_rgb = cv2.putText(
                    image_rgb,
                    "Pred: " + str(pred),
                    (10, 50),
                    cv2.FONT_HERSHEY_PLAIN,
                    1.7,
                    color,
                    2,
                )
                image_rgb = (
                    torch.from_numpy(image_rgb.transpose((2, 0, 1))).float() / 255.0
                )
                lst_img_final.append(image_rgb)
            grid = torchvision.utils.make_grid(lst_img_final, nrow=3)
            writer.add_image("train_images", grid, epoch)

            """ Write metrics to tensorboard """
            writer.add_scalar("loss/train_loss", loss_avg.val(), epoch)
            lr = get_lr(optimizer)
            writer.add_scalar("learning_rate/lr", lr, epoch)

        # save model
        output_file_ckpt = os.path.join(output_dir, f"epoch_{epoch}.pth")
        torch.save(model.state_dict(), output_file_ckpt)

        logger.info(
            f"[{epoch}/{config.train.num_epoch}] Train loss: {loss_avg.val():0.5f}"
        )
        if scheduler is not None:
            scheduler.step(epoch)
        loss_avg.reset()
        epoch += 1


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
    # Get args
    parser = argparse.ArgumentParser("parser")
    parser.add_argument("--config_file", required=True, help="Path to input config")
    args = parser.parse_args()

    # Get config
    config = get_config()
    config.set_config(args.config_file)
    date_str = ""
    if config.general.get("create_timestamp_folder", True):
        date_str = datetime.datetime.now().strftime("%Y-%m-%dT%Hh%Mm%Ss")

    output_dir = os.path.join(config.general.train_dir, date_str)
    config.general.set_param("train_dir", output_dir)
    os.makedirs(config.general.train_dir, exist_ok=True)

    out_config_file = os.path.join(config.general.train_dir, "config.yaml")
    config.to_yaml(out_config_file)

    logging.info("Config written successfully in file config.yaml")

    train(config)
