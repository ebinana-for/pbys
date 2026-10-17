"""Evaluation script for posture classification models"""
import logging
from tqdm import tqdm
import torch
import torchvision
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.utils.train_utils import Averager
import cv2


def recur_count(list_in):
    if not list_in:  # keep going until list is empty
        return 0
    else:
        # add length of list element 0 and move to next element
        return recur_count(list_in[1:]) + len(list_in[0]) 


def eval(eval_loader,
         model,
         heads_list,
         loss,
         writer,
         epoch,
         device,
         config,
         model_n_inputs=2,
         normalize_cm=False
         ):
    """Evaluation function for posture classification models.
    Args:
        eval_loader : DataLoader for the evaluation dataset.
        model : The posture classification model to evaluate.
        heads_list: List of heads in the model, each containing information about the loss and labels.
        loss : Loss function used for evaluation.
        writer : TensorBoard writer for logging results.
        epoch : Current epoch number for logging.
        device : Device to run the evaluation on (CPU or GPU).
        config : Configuration object containing model and evaluation parameters.
        model_n_inputs : Number of inputs to the model (default is 2).
        normalize_cm : Whether to normalize the confusion matrix (default is False).
    Returns:
        None
    This function performs the following steps:
    1. Initializes the model in evaluation mode.
    2. Iterates over the evaluation DataLoader to get image tensors and labels.
    3. Computes predictions using the model and compares them with ground truth labels.
    4. Computes the loss and updates the confusion matrix.
    5. Logs the evaluation results to TensorBoard.
    6. Displays the confusion matrix and some sample images with predictions.
    7. Returns the evaluation results.
    Note:
        This function assumes that the model uses Binary Cross-Entropy (BCE) loss.
        It also assumes that the labels are one-hot encoded and that the first column
        corresponds to negative labels (not used for loss computation).
    """

    model.eval()
    loss_avg = Averager()
    logging.info("-" * 80)
    dataloader_pbar = tqdm(
        eval_loader,
        total=len(eval_loader),
        colour="green",
        leave=None
    )
       
    nb_imgs = 0
    out_dim = len(config.input.positive_labels)
    cm = np.zeros((out_dim+1, out_dim+1))
    cm_labels = np.array([ii for ii in range(out_dim+1)])
    # test loop
    # Works only for BCE loss
    for _, elt in enumerate(heads_list):
        # _, elt = next(iter(elt0.items()))
        assert elt["loss"] == 'BCE', "Evaluation works only for BCE loss."
    with torch.no_grad():
        for image_tensors, posture_labels, posture_kpts, *_ in dataloader_pbar:
            posture_labels = posture_labels.to(device)
            # one_hot to class
            posture_labels_pos = [torch.where(vec >= 0.5)[0].cpu().numpy().tolist()
                                  for vec in torch.unbind(posture_labels, dim=0)]
            if model_n_inputs == 2:
                pred = model(image_tensors.to(device), posture_kpts.to(device))
            else:
                pred = model(image_tensors.to(device))

            # Compute loss ; create a list of pred and gt according 
            # to the number of heads
            pred_list = []
            label_list = []
            if len(heads_list) > 1:
                for elt in heads_list:
                    # _, elt = next(iter(head.items()))
                    indices = torch.tensor(elt['labels']).to(device)
                    pred_head = torch.index_select(pred, 1, indices-1)
                    pred_list.append(pred_head)
                    
                    label_head = torch.index_select(posture_labels, 1, indices)
                    label_head = label_head.to(device)
                    label_list.append(label_head)
            else:
                pred_list = [pred]
                # Remove 1st column (negative labels, unused for loss)
                label_list = [posture_labels[:, 1:]] 

            cost = loss.compute(pred_list, label_list, epoch=epoch)
            loss_avg.add(cost)
            
            pred = torch.sigmoid(pred)
            preds = [(torch.where(vec > 0.5)[0].cpu().numpy()+1).tolist()
                     for vec in torch.unbind(pred, dim=0)]
            preds_cm = [[0] if len(elt) == 0 else elt for elt in preds]

            y_true_list = []
            y_pred_list = []
            for y_true, y_pred in zip(posture_labels_pos, preds_cm):
                # duplicate pred-label pairs for confusion matrix
                y_true_list += [yt for yt in y_true for yp in y_pred] 
                y_pred_list += [yp for yt in y_true for yp in y_pred]

            cm += confusion_matrix(y_true_list, y_pred_list, labels=cm_labels)

            if nb_imgs == 0:  # Display some images in tensorboard
                images_to_show = image_tensors[: config.eval.nb_imgs_in_tensorboard]
                preds_to_show = preds_cm[: config.eval.nb_imgs_in_tensorboard]
                gt_to_show = posture_labels_pos[: config.eval.nb_imgs_in_tensorboard]
                
            nb_imgs += image_tensors.size(0)

    mean_accuracy = np.trace(cm) / nb_imgs
    if normalize_cm:
        norm = cm.sum(axis=1)[:, np.newaxis]
        norm[norm == 0] = 1.0
        cm = cm.astype('float') / norm
        cm = np.nan_to_num(cm, copy=True)

    display_labels = ['correctly_seated'] + \
                     [eval_loader.dataset.label_map[key] 
                      for key in range(len(config.input.positive_labels))]
    cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    text_kw = {'fontsize': 'small'}
    cm_disp.plot(xticks_rotation="vertical",
                 values_format=".2f",
                 text_kw=text_kw)
    cm_disp.figure_.tight_layout()
   
    lst_img_final = []
    for image, pred, gt in zip(images_to_show, preds_to_show, gt_to_show):
        image = (np.array(image.cpu()).transpose((1, 2, 0))) * 255
        image = np.ascontiguousarray(image, dtype=np.uint8)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        color = (0, 255, 0)
        image_rgb = cv2.putText(image_rgb, 'GT: '+str(gt),
                                (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.7, color, 2)
        image_rgb = cv2.putText(image_rgb, 'Pred: '+str(pred),
                                (10, 50), cv2.FONT_HERSHEY_PLAIN, 1.7, color, 2)
        image_rgb = torch.from_numpy(image_rgb.transpose((2, 0, 1))).float()/255.0
        lst_img_final.append(image_rgb)

    grid = torchvision.utils.make_grid(lst_img_final, nrow=3)

    logging.info(
            f"[{epoch}/{config.train.num_epoch}] Eval loss: {loss_avg.val():0.5f}"
        )
    """ Write metrics to tensorboard """
    writer.add_scalar("loss/eval_loss", loss_avg.val(), epoch)
    writer.add_scalar("loss/eval_mean_accuracy", mean_accuracy, epoch)
    writer.add_figure("eval/confusion_matrix", cm_disp.figure_, epoch)
    writer.add_image("eval_images", grid, epoch)
    
    return

