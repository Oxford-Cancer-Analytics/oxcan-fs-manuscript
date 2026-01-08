import numpy as np
from src.figure_generation import compute_roc_curve


def test_compute_roc_curve(get_X_y_data_imputed):
    _, y, *_ = get_X_y_data_imputed
    fpr_steps = np.linspace(0, 1, 101)
    preds = np.array([0, 1, 1, 0, 0])

    output = compute_roc_curve([preds], [y], fpr_steps)

    assert [
        "mean_tprs",
        "mean_auc",
        "interp_tprs",
        "interp_thresh",
        "sens_spec_99",
        "sens_spec_95",
        "sens_spec_90",
        "sens_spec_85",
        "sens_spec_80",
        "roc_auc",
    ] == list(output.keys())
