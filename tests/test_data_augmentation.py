import numpy as np
import pandas as pd
from src.data_augmentation import augment_cancer_patients


def test_up_sampling():
    """Data augmentation makes correct number of patients."""
    X = np.array(
        [  # columns are proteins
            [1, 10, 100, 1000],  # patients in rows
            [2, 11, 101, 1001],
            [3, 12, 102, 1002],
            [5, 50, 500, 5000],
            [6, 51, 501, 5001],
            [7, 52, 502, 5002],
            [1, 10, 100, 1000],
            [2, 11, 101, 1001],
            [3, 12, 102, 1002],
            [5, 50, 500, 5000],
            [6, 51, 501, 5001],
            [7, 52, 502, 5002],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1])
    X_aug, y_aug = augment_cancer_patients(pd.DataFrame(X), y, final_ratio_cancer=0.4, final_total=100)
    assert len(y_aug) == 100, "Not correct augmentation"
    assert X_aug.shape == (100, 4), "Not correct augmentation"


def test_down_sampling():
    """Data augmentation makes correct number of patients."""
    X = np.random.random(size=(100, 5))
    y = np.random.binomial(1, 0.5, size=(100,))
    X_aug, y_aug = augment_cancer_patients(X, y, final_ratio_cancer=1, final_total=25)
    assert len(y_aug) < 100, "Not correct augmentation"
