from typing import cast

import pandas as pd
from src._types import BaseEstimator


class TestFeatureElimination:
    def test_rfe(self, get_binomial_X_y_data, feature_class, pipeline):
        X, y, protein_list = get_binomial_X_y_data
        pipe_df = pd.DataFrame(X, columns=protein_list)
        pipe_df["target"] = y

        result = feature_class.elimination.recursive_feature_elimination(
            pipe_df,
            estimator=pipeline[-1],
            feature_range=range(4, 3, -1),
        )

        assert len(result) > 0
        assert result.shape[1] == 22
