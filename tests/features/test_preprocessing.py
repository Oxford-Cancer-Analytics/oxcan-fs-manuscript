import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal
from xgboost import XGBRegressor


class TestNormalize:
    @pytest.mark.parametrize("normalizer", ("log2", "top_l", "quantile_sample"), indirect=True)
    def test_normalizer(self, normalizer, get_classwise_imputation_df):
        X, y = get_classwise_imputation_df.drop(columns=["target"]), get_classwise_imputation_df.target
        fit_params = {}

        if normalizer.method == "top_l":
            fit_params |= {"L": 2}
            X = X.fillna(0)

        output = normalizer.fit_transform(X, y, **fit_params)

        assert isinstance(output, pd.DataFrame)

        if normalizer.method == "log2":
            assert_almost_equal((2**output).to_numpy(), X.to_numpy())
        elif normalizer.method == "top_l":
            sample_median = X.apply(lambda x: np.median(x.nlargest(fit_params["L"])), axis=0)
            est_pop_mean = 1 / X.shape[0] * sample_median.sum()
            output = output.fillna(0)

            assert_almost_equal((output * (1 / est_pop_mean * sample_median)).to_numpy(), X.to_numpy())
        else:
            stacked_na = X.stack(dropna=False)
            na_coords = [list(x) for x in stacked_na.index[stacked_na.isna()]]

            for row, column in na_coords:
                assert np.isnan(output.loc[row, column])

    def test_set_params(self, normalizer):
        assert not all(hasattr(normalizer, param) for param in ["fit_params", "test"])

        output = normalizer.set_params(test=True)
        assert isinstance(output, type(normalizer))
        assert "test" in normalizer.fit_params


class TestImputer:
    @pytest.mark.parametrize("imputer", (("qrilc", ["1", "3"]), ("minprob", ["1", "3"])), indirect=True)
    def test_imputer_mnar(self, imputer, get_classwise_imputation_df, mock_imputer):
        X, y = get_classwise_imputation_df.drop(columns=["target"]), get_classwise_imputation_df.target

        for row, col in zip([3, 0], [*imputer.features]):
            X.loc[row, int(col)] = np.nan

        fit_params = {"tune_sigma": 0.5}

        output = imputer.fit_transform(X, y, **fit_params)

        if imputer.method == "minprob":
            with pytest.raises(ValueError, match="Lower bound must be positive."):
                imputer.mnar._minprob(X, y, lower_bound=-1)

        assert isinstance(output, pd.DataFrame)
        if len(imputer.features) > 1 and not np.isnan(output[imputer.features]):
            assert output[imputer.features].notna().all().all()

    def test_imputer_mnar_all(self, imputer):
        assert all(hasattr(imputer.mnar, method) for method in ["_qrilc", "_minprob"])

    def test_imputer_mar_all(self, imputer):
        assert all(
            hasattr(imputer.mar, method) for method in ["_knn", "_mice", "_miss_forest", "_mode", "_median", "_mean"]
        )

    @pytest.mark.parametrize(
        "imputer",
        (("knn", []), ("mean", []), ("median", []), ("mode", []), ("miss_forest", []), ("mice", []), (None, [])),
        indirect=True,
    )
    def test_imputer_mar(self, imputer, get_classwise_imputation_df):
        X, y = get_classwise_imputation_df.drop(columns=["target"]), get_classwise_imputation_df.target
        fit_params = {}

        if imputer.method == "mice":
            fit_params |= {"estimator": XGBRegressor(n_estimators=10)}

        output = imputer.fit_transform(X, y, **fit_params)

        assert isinstance(output, pd.DataFrame)

        if imputer.method is None:
            assert output.equals(X)
        else:
            assert output.notna().all().all()

    def test_set_params(self, imputer):
        assert not all(hasattr(imputer, param) for param in ["fit_params", "tune_sigma"])

        output = imputer.set_params(tune_sigma=True)
        assert isinstance(output, type(imputer))
        assert "tune_sigma" in imputer.fit_params
