import numpy as np

def iou(gt,pr,threshold=0.6,eps=1e-7):
    intersection = np.sum(gt * pr)
    union = np.sum(gt) + np.sum(pr) - intersection + eps
    iou = intersection / union 
    return iou
