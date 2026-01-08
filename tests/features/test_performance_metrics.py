import numpy as np
import pandas as pd
import pytest
from src.features import performance_metrics


@pytest.mark.parametrize("thresh", (None, 0.2, -0.1, 0.0, 1.0))
def test_probability_to_label(get_X_y_data_imputed, thresh):
    _, y, *_ = get_X_y_data_imputed
    preds = np.hstack([y.reshape(-1, 1), np.array([0, 1, 1, 0, 0]).reshape(-1, 1)])

    output = performance_metrics.probability_to_label(preds, y, thresh)

    if thresh is not None:
        assert ((preds[:, 1] >= thresh) == output).all()

    assert len(output) == preds.shape[0]
    assert isinstance(output, np.ndarray)


@pytest.mark.skip(reason="too slow")
@pytest.mark.parametrize("thresh", (None, 0.2))
def test_compute_model_performance(get_X_y_data_imputed, thresh):
    X, y, *_ = get_X_y_data_imputed
    preds = np.hstack([y.reshape(-1, 1), np.array([0, 1, 1, 0, 0]).reshape(-1, 1)])

    output = performance_metrics.compute_model_performance(preds, y, X, y, threshold=thresh)

    assert [
        "balanced_accuracy",
        "f1_score",
        "sensitivity",
        "specificity",
        "pos_pred_value",
        "neg_pred_value",
        "average_precision",
        "roc_auc",
        "pr_auc",
        "confusion_matrix",
        "mcc",
    ] == list(output[0].keys())
    assert len(output) == 2
