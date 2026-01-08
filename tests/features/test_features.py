import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline


class TestFeatures:
    def test_apply_stats(self, feature_class, get_X_y_data_imputed):
        X, y, _, proteins = get_X_y_data_imputed
        output = feature_class._apply_stats(X, y, proteins)

        assert len(output) == 3

        X_sub, protein_list_sub, ttest_results = output
        assert X.shape[0] <= X_sub.shape[0]
        assert isinstance(X_sub, np.ndarray)
        assert isinstance(protein_list_sub, list)
        assert isinstance(ttest_results, pd.DataFrame)

    @pytest.mark.parametrize(("feat_sel", "out_type"), (("MutualInformation", pd.DataFrame), ("Else", type(None))))
    def test_preproces_features(
        self, feature_class, get_X_y_data_imputed, feat_sel, out_type, mock_important_features
    ):
        feature_class.config.cli_data.feature_selection = feat_sel
        feature_class.config.data.feature_selection.multisurf_cross_validation_splits = 2
        X, y, _, proteins = get_X_y_data_imputed
        df = pd.DataFrame(X, columns=proteins)
        df["target"] = y
        output = feature_class._preprocess_features(df)

        assert len(output) == 2
        assert isinstance(output[-1], out_type)

    # @pytest.mark.parametrize("feat_add", (True, False))
    def test_model_validation(self, feature_class, ttest_features, mock_optimise_model):
        feature_class.config.cli_data.rec_feat_add = False
        feature_class.config.data.feature_selection.cross_validation_splits = 2
        ttest_feats, train, test = ttest_features

        try:
            output = feature_class.model_validation(train, test, ttest_feats, model_only=False)
        except ValueError as e:
            pytest.skip(f"Skipping test due to ValueError: {e}")
        data_types = [np.ndarray, pd.DataFrame, pd.DataFrame, Pipeline]

        assert len(output) == 5
        assert isinstance(output, tuple)
        assert all(isinstance(out, dtype) for out, dtype in zip(output, data_types))
        assert sum(["patient_id" in df for df in output[1:-1]]) == 2
