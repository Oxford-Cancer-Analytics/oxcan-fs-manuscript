import pandas as pd
import pytest
from src.cli.cli_options import AugmentationEnum


class TestFeatureSelection:
    @pytest.mark.skip(reason="too slow")
    def test_get_important_features(self, get_X_y_data_imputed, feature_class):
        feature_class.selection.config.data.feature_selection.cross_validation_splits = 2
        X, y, _, proteins = get_X_y_data_imputed
        df = pd.DataFrame(X, columns=proteins)
        df["target"] = y
        output = feature_class.selection.get_important_features(df)

        assert isinstance(output, pd.DataFrame)
        assert "importance_count" in output.columns

    @pytest.mark.skip(reason="too slow")
    def test_get_important_features_with_augmentation(self, get_X_y_data_imputed, feature_class):
        feature_class.selection.config.data.feature_selection.cross_validation_splits = 2
        feature_class.config.cli_data.augmentation = AugmentationEnum.SMOTE_TOMEK

        X, y, _, proteins = get_X_y_data_imputed
        df = pd.DataFrame(X, columns=proteins)
        df["target"] = y
        output = feature_class.selection.get_important_features(df)

        assert isinstance(output, pd.DataFrame)
        assert "importance_count" in output.columns
