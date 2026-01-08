import numpy as np
import pandas as pd
import pytest
from src.features.missingness import detect_mnar


def test_detect_mnar(get_classwise_imputation_df):
    target = np.random.choice([0, 1], size=get_classwise_imputation_df.shape[0], p=[0.4, 0.6])
    get_classwise_imputation_df["target"] = target
    output = detect_mnar(get_classwise_imputation_df)

    cols = []
    for out in output:
        for col in out.columns:
            if col not in cols:
                cols.append(col)

    assert len(output) == 2
    assert all("target" in df.columns for df in output)
    assert sorted(cols) == sorted(get_classwise_imputation_df.columns)
