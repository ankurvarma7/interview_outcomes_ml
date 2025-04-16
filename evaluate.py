import numpy as np
from scipy.stats import pearsonr


# Evaluates the predictions of a model using Mean Absolute Relative Error and Pearsons Correlation
def evaluate(y_true, y_pred):
    max_true = np.max(y_true)
    relative_errors = np.abs(y_true - y_pred) / max_true
    mean_relative_error = np.mean(relative_errors)

    pearson_r, p_value = pearsonr(y_true, y_pred)
    print(f"Mean Absolute Relative Error: {mean_relative_error:.4f}")
    print(f"Pearson: {pearson_r:.4f} (p-value: {p_value:.4g})")
