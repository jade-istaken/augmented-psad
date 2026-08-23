import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
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

    if ground_truth.ndim == 4:
        ground_truth = ground_truth.squeeze(1)
    if anomaly_map.ndim == 4:
        anomaly_map = anomaly_map.squeeze(1)

    N = ground_truth.shape[0]
    labeled_components = []
    num_features_per_image = []
    total_components = 0

    for i in range(N):
        gt_img = ground_truth[i]
        if gt_img.max() > 0.5:
            gt_binary = (gt_img > 0.5).astype(np.uint8)
            labeled, num_feats = label(gt_binary)
            labeled_components.append(labeled)
            num_features_per_image.append(num_feats)
            total_components += num_feats
        else:
            labeled_components.append(None)
            num_features_per_image.append(0)

    if total_components == 0:
        return 0.0

    thresholds = np.linspace(anomaly_map.min(), anomaly_map.max(), 100) #list of thresholds to check against

    fpr_list = []
    pro_list = []

    background_mask = (ground_truth <= 0.5)
    total_bg_pixels = background_mask.sum()

    for thresh in thresholds:
        pred_binary = (anomaly_map > thresh).astype(np.uint8)

        #calculate the false positive rate on background pixels
        if total_bg_pixels > 0:
            fpr = (pred_binary[background_mask] == 1).sum() / total_bg_pixels
        else:
            fpr = 0.0

        pro_sum = 0.0
        for i in range(N):
            if num_features_per_image[i] > 0:
                labeled = labeled_components[i]
                pred_img = pred_binary[i]

                labeled_flat = labeled.ravel()
                pred_flat = pred_img.ravel()

                # np.bincount computes the sum of pred_flat for each unique label in labeled_flat
                intersections = np.bincount(labeled_flat, weights=pred_flat, minlength=num_features_per_image[i] + 1)
                unions = np.bincount(labeled_flat, minlength=num_features_per_image[i] + 1)

                # ignore background (index 0)
                intersections = intersections[1:]
                unions = unions[1:]

                pro_sum += np.sum(intersections / (unions + 1e-8))

        pro_list.append(pro_sum / total_components) #add the average per-region overlap to the list

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

def compute_metrics(gt_labels: np.ndarray, scores: np.ndarray, gt_maps: np.ndarray | None = None, pred_maps: np.ndarray | None = None):
    metrics = {}

    metrics["Image AUROC"] = calculate_auroc(gt_labels, scores)
    metrics["Image AUPRC"] = average_precision_score(gt_labels, scores)

    precisions, recalls, thresholds = precision_recall_curve(gt_labels, scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    metrics["Image F1-Max"] = np.max(f1_scores)

    if gt_maps is not None and pred_maps is not None:
        if gt_maps.ndim == 4:
            gt_maps = gt_maps.squeeze(1)
        if pred_maps.ndim == 4:
            pred_maps = pred_maps.squeeze(1)
        preds_flat = pred_maps.reshape(len(pred_maps), -1)  # [N,H,W] -> [N * H * W]
        gt_flat = gt_maps.reshape(len(gt_maps), -1)
        has_anomaly = gt_flat.max(axis=1)>0
        if has_anomaly.any():
            #metrics["Pixel PRO"] = calculate_pro(gt_maps, pred_maps)
            gt_binary_flat = (gt_flat > 0).astype(np.uint8)
            metrics["Pixel AUROC"] = roc_auc_score(gt_binary_flat.flatten(), preds_flat.flatten())

            precisions_pixels, recalls_pixels, _ = precision_recall_curve(gt_binary_flat.flatten(), preds_flat.flatten())
            f1_scores_pixels = 2* (precisions_pixels * recalls_pixels) / (precisions_pixels + recalls_pixels + 1e-8)
            metrics["Pixel F1-Max"] = np.max(f1_scores_pixels)
        else:
            metrics["Pixel AUROC"] = 0.0
            metrics["Pixel F1-Max"] = 0.0

    return metrics