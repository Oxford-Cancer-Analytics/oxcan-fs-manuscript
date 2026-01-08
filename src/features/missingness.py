from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.sandbox.stats.multicomp import multipletests


def detect_mnar(
    data: pd.DataFrame,
    alpha: float = 0.05,
    return_target: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """MNAR detection via a chi-square test comparing class-wise missingness.

    Parameters
    ----------
    data
        The unimputed data.
    alpha, optional
        Significance level, by default 0.05
    return_target, optional
        Whether to return the target column.

    Returns
    -------
        Two dataframes, the first with the mnar-specific features, the second without.
    """
    class_labels = data.target
    combos = list(combinations(class_labels.unique(), 2))

    mnar_features = []
    # Create contingency tables for each feature
    for i, feature in enumerate(data.drop(columns=["target"]).columns):
        table = pd.Series(np.where(np.isnan(data[feature]), 0, 1))
        contingency_table = pd.crosstab(table, class_labels)

        # Merge value counts, for each feature, for each class label into a contingency table
        uncorrected_p_values = []

        # compare one label to another
        for combo in combos:
            contingency_table_combo = contingency_table[[*combo]]
            try:
                _, p_value, *_ = chi2_contingency(contingency_table_combo)
            except ValueError:
                # If no missing values, continue to next combination
                continue
            uncorrected_p_values.append(p_value)

        if not uncorrected_p_values:
            continue

        _, p_values, *_ = multipletests(uncorrected_p_values, method="fdr_bh")
        if (np.array(p_values) < alpha).any():
            feature = data.iloc[:, i].name
            mnar_features.append(feature)

    mnar_df = data[mnar_features]
    other_df = pd.DataFrame(data[np.setdiff1d(data.columns, mnar_features)]).drop(columns=["target"])

    if return_target:
        mnar_df = pd.concat([mnar_df, data.target], axis=1)
        other_df = pd.concat([other_df, mnar_df.target], axis=1)

    return mnar_df, other_df
