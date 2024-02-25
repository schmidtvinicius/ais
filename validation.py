import numpy as np
import pandas as pd
from sklearn.metrics import auc


def get_results_from_file(file_path):
    """
    Get results from a file.
    """
    return pd.read_csv(file_path)


def calculate_roc_auc_components(data, step_size = 0.0001):
    """
    Calculate the AUC metric, true positive rate (tpr) and false positive rate (fpr).
    """

    score = data["anomaly_score"].values.astype(float)
    label = data["label"].values.astype(int)

    # Sort scores and corresponding labels
    sorted_indices = np.argsort(score)
    sorted_score = score[sorted_indices]
    
    # normalize the score from 0 to 1
    
    sorted_score = (sorted_score - sorted_score.min()) / (sorted_score.max() - sorted_score.min())
    sorted_label = label[sorted_indices]

    # Initialize variables
    tprs = []
    fprs = []

    # Iterate through each score as a threshold
    for threshold in np.arange(0, 1.01, step_size):

        # Compute true positive rate (tpr) and false positive rate (fpr)
        tp = np.sum((sorted_score >= threshold) & (sorted_label == 1))
        fp = np.sum((sorted_score >= threshold) & (sorted_label == 0))
        fn = np.sum((sorted_score < threshold) & (sorted_label == 1))
        tn = np.sum((sorted_score < threshold) & (sorted_label == 0))

        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)

        tprs.append(tpr)
        fprs.append(fpr)

    # Calculate AUC
    auc_value = auc(fprs, tprs)

    return auc_value, fprs, tprs
