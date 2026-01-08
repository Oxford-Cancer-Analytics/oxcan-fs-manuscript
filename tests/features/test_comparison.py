import numpy as np
import pandas as pd
import pytest


class TestModelComparison:
    def test_extract_data(self, model_comparison):
        assert model_comparison.dataframes
        assert model_comparison.model_names

    def test_get_significant_features(self, model_comparison):
        # Initially sig_figs should be empty
        assert model_comparison.sig_figs == []

        output = model_comparison.get_significant_features()

        # After calling get_significant_features, sig_figs should be populated
        assert model_comparison.sig_figs
        assert all([isinstance(out, pd.Series) for out in output])
        assert all([out.to_numpy().tolist() == sorted(out.to_numpy(), reverse=True) for out in output])

    def test_compare(self, model_comparison):
        output = model_comparison.compare()

        assert isinstance(output, pd.DataFrame)
        assert "avg" in output.columns

    def test_compare_features_keyerror(self, model_comparison):
        data = model_comparison.get_significant_features()
        key_to_error = "test"
        features = data[0].index.to_list() + [key_to_error]

        output = model_comparison._compare_features(features, data)
        assert key_to_error in output.index
        assert np.array_equal(np.array([0, 0, 0]), output.loc[key_to_error].to_numpy())
