import cv2
import numpy as np
import os
import pickle
import torch
from tqdm import tqdm
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smp_utils
from skimage.measure import label, regionprops, regionprops_table
from skimage import morphology
from sklearn.model_selection import train_test_split

from backend.augmentation import get_preprocessing
from backend.dataset import HuBMAPDataset,prepare_image_map_annotation
from backend.metric import iou
from backend.utils.io import import_configuration
from backend.utils.seed import seed_everything 

def get_vessels(mask):
    #mask = image.astype(bool)
    label_img = label(mask)
    regions = regionprops(label_img)

    # get items/vessels
    label_items = []
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        zero = np.zeros(mask.shape)
        zero[minr:maxr, minc:maxc] = 1
        label_item = (mask*zero).astype(bool)
        label_items.append(label_item)
    return  label_items

def compute_iou(labels, y_pred):
    """
    Computes the IoU for instance labels and predictions.

    Args:
        labels (np array): Labels.
        y_pred (np array): predictions

    Returns:
        np array: IoU matrix, of size true_objects x pred_objects.
    """

    true_objects = len(labels)
    pred_objects = len(y_pred)

    iou_arr = np.zeros((true_objects,pred_objects))

    for it in range(true_objects):
        for ip in range(pred_objects):
            iou_arr[it,ip] = iou(labels[it],y_pred[ip])
    return iou_arr

def precision_at(threshold, iou):
    """
    Computes the precision at a given threshold.

    Args:
        threshold (float): Threshold.
        iou (np array [n_truths x n_preds]): IoU matrix.

    Returns:
        int: Number of true positives,
        int: Number of false positives,
        int: Number of false negatives.
    """
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) >= 1  # Correct objects
    false_negatives = np.sum(matches, axis=1) == 0  # Missed objects
    false_positives = np.sum(matches, axis=0) == 0  # Extra objects
    tp, fp, fn = (
        np.sum(true_positives),
        np.sum(false_positives),
        np.sum(false_negatives),
    )
    return tp, fp, fn

def iou_map(truths, preds, verbose=1):
    """
    Computes the metric for the competition.
    Masks contain the segmented pixels where each object has one value associated,
    and 0 is the background.

    Args:
        truths (list of masks): Ground truths.
        preds (list of masks): Predictions.
        verbose (int, optional): Whether to print infos. Defaults to 0.

    Returns:
        float: mAP.
    """
    ious = [compute_iou(truth, pred) for truth, pred in zip(truths, preds)]

    if verbose:
        print("Thresh\tTP\tFP\tFN\tPrec.")

    prec = []
    for t in np.arange(0.6, 1.0, 0.05):
        tps, fps, fns = 0, 0, 0
        for iou in ious:
            tp, fp, fn = precision_at(t, iou)
            tps += tp
            fps += fp
            fns += fn

        p = tps / (tps + fps + fns)
        prec.append(p)

        if verbose:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tps, fps, fns, p))

    if verbose:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))

    return np.mean(prec)

config_path = 'config.20230610_train_v01'
config = import_configuration(config_path)

base_dir = config.base_dir
tag = config.tag
version = config.version
seed = config.seed
model_class = config.model_class
model_args = config.model_args
encoder = config.encoder
encoder_weights = config.encoder_weights
classes = config.classes
pr_threshold = config.pr_threshold

model = model_class(**model_args)
preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)
image_map,annotations,golden_ids,silver_ids = prepare_image_map_annotation(base_dir)

valid_image_keys = pickle.load(open(os.path.join(tag,'valid_image_map_{:s}.pkl'.format(version)),'rb'))
valid_image_map = {k:image_map[k] for k in valid_image_keys}
dataset = HuBMAPDataset(valid_image_map,annotations,classes,preprocessing=get_preprocessing(preprocessing_fn))
model.load_state_dict(torch.load(tag+'/best_model_{:s}.pth'.format(version),map_location=torch.device('cpu')))

output_dir = os.path.join(tag,'display/')
os.makedirs(output_dir,exist_ok=True)
truths,preds = [],[]
with torch.no_grad():
    for i in tqdm(range(len(dataset))):
        key,image,gt = dataset[i]
        if gt.sum() == 0: continue
        image = torch.tensor(image).reshape((1,*image.shape))
        pr = (model(image) > pr_threshold).numpy()[0,0,:]
        pr_instances = get_vessels(pr)
        gt_instances = get_vessels(gt[0,:])
        truths.append(gt_instances)
        preds.append(pr_instances)
print(iou_map(truths,preds))
