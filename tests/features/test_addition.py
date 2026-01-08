import pandas as pd
import pytest
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier


class TestFeatureAddition:
    @pytest.mark.skip(reason="too slow")
    @pytest.mark.parametrize("method", ("permutation_importance", "tree_importance"))
    def test_rfa_feat_imp(self, get_binomial_X_y_data, feature_class, pipeline, method):
        X, y, proteins = get_binomial_X_y_data
        df = pd.DataFrame(X, columns=proteins)
        df["target"] = y
        df["label"] = y.copy()

        feature_class.selection.config.data.feature_selection.cross_validation_splits = 2
        feature_class.selection.config.data.feature_selection.addition.importance_selection = method
        result, importances = feature_class.addition.recursive_feature_addition(
            df,
            model_pipeline=pipeline,
        )

        assert isinstance(result, pd.DataFrame)
        assert isinstance(importances, pd.DataFrame)

    @pytest.mark.skip(reason="too slow")
    @pytest.mark.parametrize("pipeline", (RandomForestClassifier(), xgb.XGBClassifier()), indirect=True)
    def test_rfa_model_perf(self, get_binomial_X_y_data, feature_class, pipeline):
        X, y, proteins = get_binomial_X_y_data
        df = pd.DataFrame(X, columns=proteins)
        df["target"] = y

        feature_class.selection.config.data.feature_selection.cross_validation_splits = 2
        result, importances = feature_class.addition.recursive_feature_addition(
            df, model_pipeline=pipeline, method="model_perf"
        )

        assert isinstance(result, pd.DataFrame)
        assert isinstance(importances, type(None))
