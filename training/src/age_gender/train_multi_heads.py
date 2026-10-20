"""Training script for age/gender estimation models"""

import os

os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
import argparse
import datetime
import logging
import tqdm
import torch
import cv2
import torchinfo
import torchvision
from torch.utils.tensorboard import SummaryWriter
from pprint import pformat

from src.age_gender.models.mobilenetv2 import MobileNetV2MultiHeads
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from src.age_gender.models.mobilenetv3 import MobileNetV3MultiHeads
from src.age_gender.models.edgevit.edgevit import (
    edgevit_xxs_multi_heads,
    edgevit_xs_multi_heads,
    edgevit_s_multi_heads,
)
from src.age_gender.models.efficientvit.build import (
    EfficientViT_M0_multi_heads,
    EfficientViT_M1_multi_heads,
    EfficientViT_M2_multi_heads,
)

from src.age_gender.dataset.get_dataloader import get_train_loader, get_eval_loader
from src.utils.config import get_config
from src.utils.train_utils import Averager, get_lr
from src.age_gender.test_multi_heads import eval
from src.utils.losses import Loss

import random
import numpy as np

import hashlib

cv2.setNumThreads(0)
cv2.setLogLevel(2)

__author__ = "Probayes"

def params_hash(model):
    m = hashlib.md5()
    for p in model.parameters():
        m.update(p.detach().cpu().contiguous().numpy().tobytes())
    return m.hexdigest()[:8]

def batch_hash(*tensors):
    m = hashlib.md5()
    for tensor in tensors:
        m.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return m.hexdigest()[:8]

DEBUG_HASH = bool(os.environ.get("DEBUG_HASH"))
DEBUG_HASH_EVERY = max(1, int(os.environ.get("DEBUG_HASH_EVERY", "50")))

def set_seed(seed=42):
    """Fixe les graines aléatoires pour garantir la reproductibilité."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # <- AJOUT CRUCIAL
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # torch.backends.cudnn.enabled = False
    
    torch.use_deterministic_algorithms(True)  # <- GARANTIT LE MODE STRICT


def load_model(config):
    """Load the model based on the configuration
    Args:
        config (Config): configuration object
    Returns:
        model (torch.nn.Module): the loaded model
    """
    heads_param = []
    for key, val in config.model.heads_params.items():
        head = val.copy()
        head["name"] = key
        heads_param.append(head)
    model_weights = None
    if config.model.name == "mobilenet_v2_multi_heads":
        model = MobileNetV2MultiHeads(
            heads_param,
            in_chans=config.model.input_channels,
            width_mult=config.model.width_multiplier,
        )
        model_weights = MobileNet_V2_Weights.IMAGENET1K_V1
        model_weights = MobileNet_V2_Weights.verify(model_weights)
    elif config.model.name == "mobilenet_v3_multi_heads":
        model = MobileNetV3MultiHeads(
            heads_param,
            in_chans=config.model.input_channels,
            width_mult=config.model.width_multiplier,
        )
    elif config.model.name == "edgevit_xxs_multi_heads":
        model = edgevit_xxs_multi_heads(
            heads_param,
            img_size=(config.input.input_height, config.input.input_width),
            in_chans=config.model.input_channels,
            pretrained=False,
        )
    elif config.model.name == "edgevit_xs_multi_heads":
        model = edgevit_xs_multi_heads(
            heads_param,
            img_size=(config.input.input_height, config.input.input_width),
            in_chans=config.model.input_channels,
            pretrained=False,
        )
    elif config.model.name == "edgevit_s_multi_heads":
        model = edgevit_s_multi_heads(
            heads_param,
            img_size=(config.input.input_height, config.input.input_width),
            in_chans=config.model.input_channels,
            pretrained=False,
        )
    elif config.model.name == "efficientvit_xxs_multi_heads":
        model = EfficientViT_M0_multi_heads(
            heads_param,
            img_size=(config.input.input_height, config.input.input_width),
            in_chans=config.model.input_channels,
            pretrained=False,
        )
    elif config.model.name == "efficientvit_xs_multi_heads":
        model = EfficientViT_M1_multi_heads(
            heads_param,
            img_size=(config.input.input_height, config.input.input_width),
            in_chans=config.model.input_channels,
            pretrained=False,
        )
    elif config.model.name == "efficientvit_s_multi_heads":
        model = EfficientViT_M2_multi_heads(
            heads_param,
            img_size=(config.input.input_height, config.input.input_width),
            in_chans=config.model.input_channels,
            pretrained=False,
        )
    else:
        logging.error(f"Unkown model {config.model}")

    if (
        "pretrained_backbone" in config.model
        and config.model.pretrained_backbone is not None
    ):
        if os.path.isfile(config.model.pretrained_backbone):
            logging.info(
                f"Loading backbone weights from {config.model.pretrained_backbone}"
            )
            pretrained_dict = torch.load(config.model.pretrained_backbone)
            pretrained_backbone = {
                k.removeprefix("module."): v
                for k, v in pretrained_dict.items()
                if "heads" not in k
            }
            model_dict = model.state_dict()
            model_dict.update(
                pretrained_backbone
            )  # overwrite entries in the existing state dict
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
                model_dict.update(
                    pretrained_backbone
                )  # overwrite entries in the existing state dict
                model.load_state_dict(model_dict)
            else:
                logging.warning(
                    "You requested {} weights, but they are not implemented for {}".format(
                        config.model.pretrained_backbone, config.model.name
                    )
                )
        else:
            logging.error(
                "Unkown model weights {}".format(config.model.pretrained_backbone)
            )
    return model


def train(config):
    """Main training function
    Args:
        config (Config): configuration object
    Returns:
        None
    """

    seed_value = getattr(config.general, "seed", 42) if hasattr(config, "general") else 42
    set_seed(seed_value)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info("Starting training...")
    logging.info(f"Using config file {config}")

    # GPU settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    # Data Part
    train_loader = get_train_loader(config)
    eval_loader = get_eval_loader(config)

    # Model configuration
    model = load_model(config)

    # Summarize the given PyTorch model
    logging.info(
        torchinfo.summary(
            model.to(device),
            [
                (
                    1,
                    config.model.input_channels,
                    config.input.input_height,
                    config.input.input_width,
                )
            ],
            dtypes=[torch.cuda.FloatTensor],
            depth=1,
        )
    )

    # setup loss
    heads_list = model.heads_list
    loss = Loss(config, heads_list)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    if config.model.checkpoint:
        model.load_state_dict(torch.load(config.model.checkpoint))

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

    # test procedure
    with torch.no_grad():
        model.eval()
        current_accuracy = eval(
            eval_loader, model, heads_list, writer, epoch, device, config
        )

    log_str = "Current "
    for elt, val in zip(heads_list, current_accuracy):
        if elt["name"] == "gender":
            log_str += "gender acc. = {}".format(val)
        elif elt["name"] == "age":
            log_str += " / age mae = {}".format(val)

    logging.info(log_str)

    print("cudnn.benchmark =", torch.backends.cudnn.benchmark,
      "| deterministic =", torch.backends.cudnn.deterministic,
      "| det_algos =", torch.are_deterministic_algorithms_enabled(),
      "| devices =", torch.cuda.device_count(),
      "| torch =", torch.__version__, "| cudnn =", torch.backends.cudnn.version(),
      flush=True)

    # Train Loop
    while True:
        if epoch == config.train.num_epoch:
            logging.info(f"Maximum number of epoch reached ({epoch}). End of training")
            sys.exit()
        model.train()
        logging.info(f"Start epoch {epoch}/{config.train.num_epoch}")
        for dataset in train_loader.datasets.values():
            if hasattr(dataset, 'set_epoch'):
                dataset.set_epoch(epoch)
        train_pbar = tqdm.tqdm(
            train_loader, total=len(train_loader), colour="red", leave=None, 
            desc=f"Epoch {epoch}/{config.train.num_epoch}"
        )

        for step, (image_tensors, gender_labels, age_labels, *_) in enumerate(train_pbar):
            gender_labels = gender_labels.to(device)
            age_labels = age_labels.to(device)
            preds = model(image_tensors.to(device))

            label_list = []
            for elt in heads_list:
                if elt["name"] == "gender":
                    label_list.append(gender_labels)
                elif elt["name"] == "age":
                    label_list.append(age_labels)

            # cost = criterion(preds, gender_labels)
            cost = loss.compute(preds, label_list, epoch=epoch)
            model.zero_grad()
            cost.backward()
            optimizer.step()
            loss_avg.add(cost)

            if DEBUG_HASH and step % DEBUG_HASH_EVERY == 0:
                print(f"HASH train epoch={epoch} step={step} "
                      f"loss={cost.item():.8f} "
                      f"batch={batch_hash(image_tensors, gender_labels, age_labels)} "
                      f"params={params_hash(model)}", flush=True)
        epoch += 1
        model.eval()

        # test procedure
        with torch.no_grad():
            current_accuracy = eval(
                eval_loader, model, heads_list, writer, epoch, device, config
            )

            log_str = "Current "
            for elt, val in zip(heads_list, current_accuracy):
                if elt["name"] == "gender":
                    log_str += "gender acc. = {}".format(val)
                elif elt["name"] == "age":
                    log_str += " / age mae = {}".format(val)
            logging.info(log_str)

            grid = torchvision.utils.make_grid(
                image_tensors[: config.eval.nb_imgs_in_tensorboard], nrow=3
            )
            writer.add_image("train_images", grid, epoch)

            """ Write metrics to tensorboard """
            writer.add_scalar("loss/train_loss", loss_avg.val(), epoch)
            lr = get_lr(optimizer)
            writer.add_scalar("learning_rate/lr", lr, epoch)

        # save model
        output_file_ckpt = os.path.join(output_dir, f"epoch_{epoch}.pth")
        torch.save(model.state_dict(), output_file_ckpt)

        log_str = "Current "
        for elt, val in zip(heads_list, current_accuracy):
            if elt["name"] == "gender":
                log_str += "gender acc. = {}".format(val)
            elif elt["name"] == "age":
                log_str += " / age mae = {}".format(val)
        logging.info(log_str)

        logging.info(
            f"[{epoch}/{config.train.num_epoch}] Train loss: {loss_avg.val():0.5f}"
        )
        if scheduler is not None:
            scheduler.step(epoch)
        loss_avg.reset()


if __name__ == "__main__":
    # Get args
    parser = argparse.ArgumentParser("parser")
    parser.add_argument("--config_file", required=True, help="Path to input config")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    args = parser.parse_args()

    # Get config
    config = get_config()
    config.set_config(args.config_file)
    if args.seed is not None:
        if hasattr(config, "general"):
            if hasattr(config.general, "seed"):
                config.general.set_param("seed", args.seed)
            else:
                config.general.add_param("seed", args.seed)
        else:
            config.add_param("seed", args.seed)

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

