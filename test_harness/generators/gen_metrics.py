"""Generate golden data for metrics tests."""

import json
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


def generate():
    cases = []

    # Regression metrics
    y_true_reg = [3.0, -0.5, 2.0, 7.0]
    y_pred_reg = [2.5, 0.0, 2.0, 8.0]

    cases.append(
        {
            "name": "regression_basic",
            "y_true": y_true_reg,
            "y_pred": y_pred_reg,
            "mse": float(mean_squared_error(y_true_reg, y_pred_reg)),
            "mae": float(mean_absolute_error(y_true_reg, y_pred_reg)),
            "r2": float(r2_score(y_true_reg, y_pred_reg)),
        }
    )

    # Another regression case
    y_true_reg2 = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_pred_reg2 = [1.1, 2.2, 2.8, 4.1, 4.9]

    cases.append(
        {
            "name": "regression_close",
            "y_true": y_true_reg2,
            "y_pred": y_pred_reg2,
            "mse": float(mean_squared_error(y_true_reg2, y_pred_reg2)),
            "mae": float(mean_absolute_error(y_true_reg2, y_pred_reg2)),
            "r2": float(r2_score(y_true_reg2, y_pred_reg2)),
        }
    )

    # Binary classification metrics
    y_true_bin = [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    y_pred_bin = [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]

    y_true_int = [int(v) for v in y_true_bin]
    y_pred_int = [int(v) for v in y_pred_bin]

    cases.append(
        {
            "name": "binary_classification",
            "y_true": y_true_bin,
            "y_pred": y_pred_bin,
            "accuracy": float(accuracy_score(y_true_int, y_pred_int)),
            "confusion_matrix": confusion_matrix(y_true_int, y_pred_int).tolist(),
            "precision": precision_score(
                y_true_int, y_pred_int, average=None, zero_division=0.0
            ).tolist(),
            "recall": recall_score(
                y_true_int, y_pred_int, average=None, zero_division=0.0
            ).tolist(),
            "f1": f1_score(
                y_true_int, y_pred_int, average=None, zero_division=0.0
            ).tolist(),
        }
    )

    # Multiclass classification
    y_true_multi = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0]
    y_pred_multi = [0.0, 2.0, 1.0, 0.0, 0.0, 2.0, 0.0, 1.0, 2.0]

    y_true_int_m = [int(v) for v in y_true_multi]
    y_pred_int_m = [int(v) for v in y_pred_multi]

    cases.append(
        {
            "name": "multiclass_classification",
            "y_true": y_true_multi,
            "y_pred": y_pred_multi,
            "accuracy": float(accuracy_score(y_true_int_m, y_pred_int_m)),
            "confusion_matrix": confusion_matrix(y_true_int_m, y_pred_int_m).tolist(),
            "precision": precision_score(
                y_true_int_m, y_pred_int_m, average=None, zero_division=0.0
            ).tolist(),
            "recall": recall_score(
                y_true_int_m, y_pred_int_m, average=None, zero_division=0.0
            ).tolist(),
            "f1": f1_score(
                y_true_int_m, y_pred_int_m, average=None, zero_division=0.0
            ).tolist(),
        }
    )

    return cases
