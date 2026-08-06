import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.ndimage import label

def calculate_auroc(ground_truth: np.ndarray, predictions: np.ndarray) -> float:
    if np.all(ground_truth == 0) or np.all(ground_truth == 1):
        return 0.0
    return roc_auc_score(ground_truth.flatten(), predictions.flatten())

def calculate_pro(
        ground_truth: np.ndarray,
        anomaly_map: np.ndarray,
        max_fpr: float = 0.3
) -> float:
    #calculate the per-region overlap
    #fpr = false positive rate
    gt_binary = (ground_truth > 0.5).astype(np.uint8) #just making sure to binarize the mask

    labeled_gt, num_features = label(gt_binary) #find the connected components
    if num_features == 0:
        return 0.0

    thresholds = np.linspace(anomaly_map.min(), anomaly_map.max(), 100) #list of thresholds to check against

    fpr_list = []
    pro_list = []

    for thresh in thresholds:
        pred_binary = (anomaly_map > thresh).astype(np.uint8)

        #calculate the false positive rate on background pixels
        background_mask = (gt_binary == 0)
        if background_mask.sum() > 0:
            fpr = (pred_binary[background_mask] == 1).sum() / background_mask.sum()
        else:
            fpr = 0.0

        if fpr <= max_fpr:
            #we limit to only thresholds where FPR is below max_fpr so that its not flooded by worthless thresholds
            fpr_list.append(fpr)
            #calculate PRO for each connected component
            pro_sum = 0.0
            for i in range(1, num_features+1):
                component_mask = (labeled_gt == i)
                intersection = (pred_binary[component_mask] == 1).sum()
                union = component_mask.sum()
                pro_sum += (intersection / (union + 1e-8)) #add 1e-8 to denom to avoid divide by 0
            pro_list.append(pro_sum / num_features) #add the average per-region overlap to the list

    if not fpr_list:
        return 0.0

    #convert to arrays (fpr is normalized because we capped it
    fpr_array = np.array(fpr_list) / max_fpr
    pro_array = np.array(pro_list)

    #sort by fpr to ensure that integration works right
    sort_idx = np.argsort(fpr_array)
    fpr_array = fpr_array[sort_idx]
    pro_array = pro_array[sort_idx]

    #add origin so that graphing like works
    fpr_array = np.concatenate([[0], fpr_array])
    pro_array = np.concatenate([[0], pro_array])

    return np.trapezoid(pro_array, fpr_array).item()