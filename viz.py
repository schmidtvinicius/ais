import matplotlib.pyplot as plt

from validation import calculate_roc_auc_components, get_results_from_file


def plot_roc_auc(fprs, tprs, auc_metric, n, r):
    # Plot ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(fprs, tprs, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_metric:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve (n={n}, r={r})')
    plt.legend(loc='lower right')
    plt.show()


def plot_roc_auc_from_file(file_path, n, r):
    data = get_results_from_file(file_path)
    auc_value, fprs, tprs = calculate_roc_auc_components(data)
    plot_roc_auc(fprs, tprs, auc_value, n, r)


def plot_auc_optimization(n, rs, auc_values):
    plt.figure(figsize=(8, 8))
    plt.plot(rs, auc_values, color='darkorange', lw=2, label=f'AUC')
    plt.xlabel('r')
    plt.ylabel('AUC')
    plt.title(f'AUC Optimization (n={n})')
    plt.legend(loc='lower right')
    plt.show()


def plot_best_aucs_per_n(n_values, best_aucs):
    plt.figure(figsize=(8, 8))
    plt.plot(n_values, best_aucs, color='darkorange', lw=2, label=f'AUC')
    plt.xlabel('n')
    plt.ylabel('AUC')
    plt.title(f'Best AUCs per n')
    plt.legend(loc='lower right')
    plt.show()
