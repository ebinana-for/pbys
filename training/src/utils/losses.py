"""Losses module for Forvia OMS model training"""

import logging
import torch


class BCECDLoss(torch.nn.Module):
    """
    Reference:
        [1] Liu et al. "C-Adapter: Adapting Deep Classifiers for Efficient Conformal
        Prediction Sets". arXiv:2410.09408, 2024.
    This loss is a combination of Binary Cross Entropy (BCE) and Conformal Discriminative
    Loss (CD). It is designed to be used with a model that outputs logits for binary
    classification tasks, and it applies a temperature parameter to control the sharpness
    of the sigmoid function used in the BCE loss.
    The BCE loss is computed using the logits directly, while the CD loss is computed
    using the predicted class labels (obtained by taking the argmax of the target tensor).
    The final loss is a weighted sum of the BCE and CD losses, where the weights are
    controlled by the `lambda_cd` and `lambda_bce` parameters.
    Args:
        lambda_cd (float): Weight applied to the CD part of the loss.
        lambda_bce (float): Weight applied to the BCE part of the loss.
        epsilon (float): Temperature parameter that controls the sharpness of the
            sigmoid function. Smaller values create sharper boundaries.
        reduction (str): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default is 'none'.
    """
    def __init__(self, lambda_cd=0.5, lambda_bce=0.5, epsilon=1e-4, reduction="none"):

        super(BCECDLoss, self).__init__()
        from torch_cp.torchcp.classification.loss.cd import CDLoss
        from torch_cp.torchcp.classification.predictor.split import (
            SplitPredictor as Predictor,
        )
        from torch_cp.torchcp.classification.score.thr import THR

        predictor = Predictor(THR(score_type="softmax"))
        self.lambda_cd = lambda_cd
        self.lambda_bce = lambda_bce
        self.cd_loss = CDLoss(predictor, epsilon, reduce=reduction)
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, input, target):
        loss0 = self.bce_loss.forward(input, target)
        loss1 = self.cd_loss.forward(input, torch.argmax(target, dim=1))
        return self.lambda_bce * loss0 - self.lambda_cd * loss1


class Loss:
    """
    Loss module for Forvia OMS model training.
    This module supports multiple loss functions including Binary Cross Entropy (BCE),
    Mean Squared Error (MSE), and a combination of BCE and Conformal Discriminative Loss
    (BCECD). It also supports Online Hard Example Mining (OHEM) and focal loss strategies.
    Args:
        config (object): Configuration object containing loss parameters.
        head_param (list): List of dictionaries containing head parameters including
            loss type, loss strategy, and loss weight.
    Attributes:
        criterion (list): List of loss functions corresponding to each head.
        loss_strategy (list): List of loss strategies for each head.
        loss_weight (list): List of weights for each loss function.
        ohem_start_epoch (list): List of epochs to start OHEM for each head.
        ohem_keep_ratio (list): List of ratios to keep during OHEM for each head.
    Example usage:
        loss_module = Loss(config, head_param)
        loss_value = loss_module.compute(pred_list, gt_list, epoch)
    """

    def __init__(self, config, head_param):
        default_ohem_keep_ratio = 0.75
        default_ohem_start_epoch = 3
        self.criterion = []
        self.loss_strategy = []
        self.loss_weight = []
        self.ohem_start_epoch = []
        self.ohem_keep_ratio = []

        # Parse heads config
        for elt in head_param:
            # key, elt = next(iter(elt0.items()))
            if elt["loss"] == "BCE":
                if elt["loss_strategy"] in ["ohem", "focal", "none"]:
                    self.criterion.append(torch.nn.BCELoss(reduction="none"))
                    self.loss_strategy.append(elt["loss_strategy"])
                    loss_weight = elt["loss_weight"] if "loss_weight" in elt else 1.0
                    self.loss_weight.append(loss_weight)
                else:
                    logging.error(
                        "Invalid loss strategy for BCE: {}".format(elt["loss_strategy"])
                    )
            elif elt["loss"] == "MSE":
                if elt["loss_strategy"] in ["ohem", "none"]:
                    self.criterion.append(torch.nn.MSELoss(reduction="none"))
                    self.loss_weight.append(elt["loss_weight"])
                    self.loss_strategy.append(elt["loss_strategy"])
                else:
                    logging.error(
                        "Invalid loss strategy for MSE: {}".format(elt["loss_strategy"])
                    )
            elif (
                elt["loss"] == "BCECD"
            ):  # mix of BCE and Conformal Discriminative Loss (C-adapter, from [1])
                if elt["loss_strategy"] in ["none"]:
                    lambda_cd = config.losses.BCECD.lambda_cd
                    lambda_bce = config.losses.BCECD.lambda_bce
                    epsilon = config.losses.BCECD.epsilon
                    self.criterion.append(
                        BCECDLoss(
                            lambda_cd=lambda_cd,
                            lambda_bce=lambda_bce,
                            epsilon=epsilon,
                            reduction="none",
                        )
                    )
                    self.loss_strategy.append(elt["loss_strategy"])
                    self.loss_weight.append(elt["loss_weight"])
                else:
                    logging.error(
                        "Invalid loss strategy for BCE-CD: {}".format(
                            elt["loss_strategy"]
                        )
                    )
            else:
                logging.error("Unkown loss {}".format(elt["loss"]))

            if "ohem_start_epoch" in elt.keys():
                self.ohem_start_epoch.append(elt["ohem_start_epoch"])
            else:
                self.ohem_start_epoch.append(default_ohem_start_epoch)

            if "ohem_keep_ratio" in elt.keys():
                self.ohem_keep_ratio.append(elt["ohem_keep_ratio"])
            else:
                self.ohem_keep_ratio.append(default_ohem_keep_ratio)

    def compute(self, pred_list, gt_list, epoch=0):
        loss = float(0.0)
        for (
            criterion,
            strategy,
            weight,
            ohem_start_epoch,
            ohem_keep_ratio,
            pred,
            gt,
        ) in zip(
            self.criterion,
            self.loss_strategy,
            self.loss_weight,
            self.ohem_start_epoch,
            self.ohem_keep_ratio,
            pred_list,
            gt_list,
        ):
            if criterion.__class__ == torch.nn.BCELoss:
                pred = torch.sigmoid(pred)

            loss0 = criterion(pred, gt.float())
            if strategy == "ohem":
                loss0 = torch.sum(loss0, dim=1)
                if epoch >= ohem_start_epoch:
                    loss0_med = torch.quantile(loss0, 1.0 - ohem_keep_ratio)
                    loss0[loss0 < loss0_med] = 0.0
                    keep_nb = (loss0 > 0.0).sum()
                else:
                    keep_nb = len(loss0)
                loss0 = weight * loss0.sum() / keep_nb
            elif strategy == "focal":
                gamma = 1.0  # See https://pytorch.org/vision/stable/_modules/torchvision/ops/focal_loss.html
                p_t = pred * gt + (1 - pred) * (1 - gt)
                loss0 = loss0 * ((1 - p_t) ** gamma)
                loss0 = torch.sum(loss0, dim=1)
                loss0 = weight * loss0.mean()
            elif strategy == "none":
                loss0 = torch.sum(loss0, dim=1)
                loss0 = weight * loss0.mean()
            else:
                logging.error("Unkown loss strategy {}".format(criterion[2]))

            loss += loss0
        return loss
