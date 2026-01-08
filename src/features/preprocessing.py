from __future__ import annotations

import logging
import re
from typing import Any
from typing import cast
from typing import Generic
from typing import Literal
from typing import TypeAlias
from typing import TypedDict
from typing import TypeVar

import numpy as np
import pandas as pd
from missingpy.missforest import MissForest
from numpy.typing import NDArray
from scipy.stats import mode
from scipy.stats import norm
from scipy.stats import truncnorm
from sklearn.base import BaseEstimator
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from typing_extensions import Self
from typing_extensions import Unpack
from xgboost import XGBRegressor

from .._types import np_array
from .missingness import detect_mnar

T = TypeVar("T", pd.DataFrame, np_array)

logger = logging.getLogger(__name__)


class Normalize(Generic[T]):
    """Implementation of various normalization methods.

    Parameters
    ----------
    method, {"log2", "top_l", "quantile_sample"}, optional
        The method of normalization to use, by default "log2".
    """

    def __init__(self, method: Literal["log2", "top_l", "quantile_sample"] = "log2") -> None:
        self.method = method

        self.X: T
        self.y: pd.Series[float] | np_array | None
        self.feature_names_in_: list[str] | NDArray[Any]

        self.fit_params: dict[str, int] = {}

    def fit(self, X: T, y: pd.Series[float] | np_array | None = None, **fit_params: int) -> Self:
        """Sklearn-like interface for fitting.

        Parameters
        ----------
        X
            The data to normalize.
        y, optional
            The target, not used, by default None
        **fit_params
            Extra parameters to pass to the fit.

        Returns
        -------
            Fitted transformer.
        """
        self.X = X
        self.y = y
        self.feature_names_in_ = X.columns.tolist() if isinstance(X, pd.DataFrame) else np.arange(X.shape[1])
        self.fit_params |= fit_params

        return self

    def transform(self, X: T) -> pd.DataFrame:
        """Scales data by the specified transormation method.

        Parameters
        ----------
        X
            The data to normalize.

        Returns
        -------
            Transformed data.
        """
        # Check is fitted
        assert all(hasattr(self, attr) for attr in ["X", "y", "fit_params", "feature_names_in_"])

        if self.method == "log2":
            x_T = self.log2(X)
        elif self.method == "quantile_sample":
            x_T = self.quantile_sample(X)
        else:
            x_T = self.top_l_ordered_statistic(X, L=self.fit_params.get("L", 10))

        return x_T

    def fit_transform(self, X: T, y: pd.Series[float] | np_array | None = None, **fit_params: int) -> pd.DataFrame:
        """Fits to the data, then transforms it.

        Parameters
        ----------
        X
            The data to transform.
        y, optional
            The target, not used, by default None
        **fit_params
            Extra parameters to pass to the fit.

        Returns
        -------
            Transformed data.
        """
        np.random.seed(0)
        return self.fit(X, y, **fit_params).transform(X)

    def log2(self, X: T) -> pd.DataFrame:
        """Log2 transformation of data.

        Parameters
        ----------
        X
            A dataframe of unnormalized data.

        Returns
        -------
            The log2 normalized data.
        """
        return pd.DataFrame(np.log2(X), columns=self.feature_names_in_)

    def top_l_ordered_statistic(self, X: T, L: int = 10) -> pd.DataFrame:
        """Top-L ordered normalization method.

        MNAR-specific global normalization step which takes the top `L` values for each
        sample and applies the median. Scaling coefficients are calculated by the inverse
        of the population median times the sample medians.

        Parameters
        ----------
        X
            Data of unnormalized data.
        L, optional
            The number of top values to include for the median calculation, by default 10

        Returns
        -------
            The top-l normalized data.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)  # type: ignore
            assert isinstance(X, pd.DataFrame)

        n_samples = X.shape[0]  # Number of samples

        # Calculate the median of the top-L values for each sample
        sample_medians = X.apply(lambda x: np.median(x.nlargest(L)), axis=0)

        # Calculate the population median (µ0)
        population_median = 1 / n_samples * sample_medians.sum()

        # Calculate the scaling coefficients for each sample
        scaling_coefficients = 1 / population_median * sample_medians

        return X / scaling_coefficients

    def quantile_sample(self, X: T) -> pd.DataFrame:
        """Quantile normalization for each sample.

        Proteins are ranked in each sample from lowest to largest. The average is
        then calculated and re-substituted back into the original format. Distributions
        of each sample are equalized.

        Parameters
        ----------
        X
            A dataframe of unnormalized data.

        Returns
        -------
            The quantile normalized data.
        """
        raw_data = X.copy()
        if not isinstance(raw_data, pd.DataFrame):
            raw_data = pd.DataFrame(raw_data, columns=self.feature_names_in_)

        proteins = []
        values = []
        for row in range(raw_data.shape[0]):
            row_data = raw_data.iloc[row, :].sort_values()
            proteins.append(row_data.index)
            values.append(row_data.reset_index(drop=True))

        ranked_df = pd.concat(values, axis=1)
        row_means = ranked_df.mean(axis=1)

        quantile_normalized_data = []

        for i, protein in enumerate(proteins):
            ranked_proteins = pd.Series(row_means.to_numpy(), index=protein).sort_index()

            # Check for nans and add them back
            current_sample = raw_data.iloc[i]
            if current_sample.isna().any():
                na_idx = current_sample.loc[current_sample.isna()].index
                ranked_proteins[na_idx] = np.nan
            elif ranked_proteins.isna().any() and not current_sample.isna().any():
                na_idx = ranked_proteins.loc[ranked_proteins.isna()].index
                ranked_proteins[na_idx] = current_sample.loc[na_idx]

            quantile_normalized_data.append(ranked_proteins)

        # Return columns to original order
        return pd.concat(quantile_normalized_data, axis=1).T[raw_data.columns]

    def set_params(self, **params: Any) -> Self:
        """Sets fitting parameters.

        Parameters
        ----------
        **params
            A dictionary of parameters to set on calling the fit method.

        Returns
        -------
            Itself.
        """
        if not hasattr(self, "fit_params"):
            self.fit_params = {}

        for key, value in params.items():
            self.fit_params[key] = value

        return self

    def __repr__(self) -> str:
        cls_name = type(self).__name__
        args = [f"{k}={v}" for k, v in self.__dict__.items() if k not in ["feature_names_in_", "X", "y"]]
        return f"{cls_name}({', '.join(args)})"


MNARMethods: TypeAlias = Literal["qrilc", "minprob"]
MARMethods: TypeAlias = Literal["knn", "mean", "median", "mode", "miss_forest", "mice", None]
Methods: TypeAlias = MNARMethods | MARMethods


class Imputer(Generic[T]):
    """Implementation of various imputation methods.

    Parameters
    ----------
    method
        The imputation method to use.
    strategy
        The missingness-type to use.
    features, optional
        A list of features based on the strategy, by default [].
        Not used anymore.
    """

    def __init__(
        self,
        method: Methods,
        strategy: str,
        features: list[int] = [],
    ) -> None:
        self.mnar = ImputeMNAR()
        self.mar = ImputeMAR()

        self.method = method
        self.features = features
        self.strategy = strategy

        self.X: T
        self.y: pd.Series[float] | None

        self.fit_params: dict[str, Any] = {}

    def fit(self, X: T, y: pd.Series[float] | None = None, **fit_params: Any) -> Self:
        """Sklearn-like interface fitting function.

        Parameters
        ----------
        X
            The unimputed data to fit.
        y, optional
            The target, not used, by default None
        **fit_params
            Extra parameters to pass to the fit.

        Returns
        -------
            Fitted imputer.
        """
        if self.strategy != "mar":
            full_data = pd.DataFrame(
                np.hstack((X, np.array(y).reshape(-1, 1))),
                columns=[*[str(col) for col in range(X.shape[1])], "target"],
            )

        if self.strategy == "mixed":
            assert y is not None
            # Only look for mnar_features if doing mnar
            if self.method in ["qrilc", "minprob"]:
                mnar_data, _ = detect_mnar(full_data, return_target=False)
                self.features = [full_data.columns.get_loc(col) for col in mnar_data.columns]

        if self.strategy == "mnar":
            # all features assumed MNAR
            mnar_data = full_data.drop(columns=["target"]).columns
            self.features = [full_data.drop(columns=["target"]).columns.get_loc(col) for col in mnar_data]

        if self.strategy == "mar":
            # all features assumed to not be MNAR; MAR
            self.features = []

        self.fit_params |= fit_params
        self.X = X
        self.y = y

        return self

    def transform(self, X: T) -> pd.DataFrame:
        """Imputes the data by the specified imputation method.

        Parameters
        ----------
        X
            The data to impute.

        Returns
        -------
            Imputed data.
        """
        # Check is fitted
        assert all(hasattr(self, attr) for attr in ["X", "y", "fit_params", "features"])

        fp = self.fit_params
        X_is_array = isinstance(X, np.ndarray)
        X_ = pd.DataFrame(X) if X_is_array else X

        if self.method is not None:
            logger.info(f"Imputing data using {self.method} with parameters {self.fit_params}")

        if self.method in ["qrilc", "minprob"]:
            # No features as determined by the test
            if not self.features:
                return X_

            mnar_potential_params = ["tune_sigma", "quantile", "protein_na", "lower_bound"]
            mnar_params: dict[str, float | None] = {
                param: fp.get(param, None) for param in mnar_potential_params if fp.get(param, None) is not None
            }
            X_sub = X_[self.features] if X_is_array else X_.iloc[:, self.features]
            x_I = self.mnar(cast(MNARMethods, self.method), X_sub, **mnar_params)
            X_.update(x_I)
        else:
            estimator = fp.get("estimator", XGBRegressor(n_estimators=500))
            mar_params = {key: value for key, value in fp.items() if key != "estimator"}
            X_ = self.mar(cast(MARMethods, self.method), X_, estimator=estimator, **mar_params)

        if self.method is not None:
            logger.info(f"Completed imputation of data using {self.method} with parameters {self.fit_params}")

        return X_

    def fit_transform(self, X: T, y: pd.Series[float] | None = None, **fit_params: Any) -> pd.DataFrame:
        """Fits to the data, then transforms it.

        Parameters
        ----------
        X
            The data to transform.
        y, optional
            The target, not used, by default None
        **fit_params
            Extra parameters to pass to the fit.

        Returns
        -------
            Imputed data.
        """
        np.random.seed(0)
        return self.fit(X, y, **fit_params).transform(X)

    def set_params(self, **params: Any) -> Self:
        """Sets fitting parameters.

        Parameters
        ----------
        **params
            A dictionary of parameters to set on calling the fit method.

        Returns
        -------
            Itself.
        """
        if not hasattr(self, "fit_params"):
            self.fit_params = {}

        methods = {
            "qrilc": ["tune_sigma"],
            "minprob": ["tune_sigma", "quantile", "protein_na", "lower_bound"],
            "knn": list(KNNImputer().get_params().keys()),
            "mean": [],
            "median": [],
            "mode": [],
            "miss_forest": list(MissForest().get_params().keys()),
            "mice": list(IterativeImputer._parameter_constraints.keys()),
            None: [],
        }

        for key, value in params.items():
            if key not in methods[self.method]:
                continue

            self.fit_params[key] = value

        return self

    def __repr__(self) -> str:
        cls_name = type(self).__name__
        args = [f"{k}={v}" for k, v in self.__dict__.items() if k not in ["mnar", "mar", "features", "X", "y"]]

        if "XGBRegressor" in args[-1]:
            args[-1] = re.sub(r"XGBRegressor\(.*\)", "XGBRegressor(...)", args[-1].replace("\n", ""))

        return f"{cls_name}({', '.join(args)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(self, Imputer) or not isinstance(other, Imputer):
            return NotImplemented

        return (self.method == other.method) and (self.fit_params == other.fit_params)  # type: ignore


class ImputeMNAR:
    """Imputation methods specifically for Missing Not At Random."""

    def __call__(self, method: MNARMethods, /, data: pd.DataFrame, **kwargs: float | None) -> pd.DataFrame:
        """Indirect calling of imputation methods by specifying the `method`.

        Parameters
        ----------
        method
            The imputation method to perform.
        data
            The unimputed data.
        **kwargs
            Extra arguments to pass to imputation functions

        Returns
        -------
            Imputed data.
        """
        imputed_data = data

        # Get the method parameters to pass only specific ones to specific methods
        method_code = eval(f"self._{method}").__code__
        method_args = method_code.co_varnames[1 : method_code.co_argcount]
        kwargs_none = {key: value for key, value in kwargs.items() if key in method_args and value is not None}

        if method == "qrilc":
            imputed_data = self._qrilc(data, **kwargs_none)
        elif method == "minprob":
            imputed_data = self._minprob(data, **kwargs_none)

        imputed_data.columns = data.columns
        return imputed_data

    def __repr__(self) -> str:
        cls_name = type(self).__name__
        args = [f"{k}={v}" for k, v in self.__dict__.items()]
        return f"{cls_name}({', '.join(args)})"

    def _qrilc(self, data: pd.DataFrame, tune_sigma: float = 1) -> pd.DataFrame:
        """Imputes left-censored missing data.

        Missing data is randomly drawn from a truncated distribution.
        imputeLCMD: https://github.com/cran/imputeLCMD

        Parameters
        ----------
        data
            A dataframe containing left-censored missing data. Log2 imputed.
        tune_sigma, optional
            Used to tune the standard deviation if the complete data
            distribution is not Gaussian, by default 1.

        Returns
        -------
        A complete dataframe.
        """
        n_samples, n_features = data.shape
        data_imputed = data.copy()

        upper_q = 0.99
        err = 0.001
        for i in range(n_samples):
            current_sample = data.iloc[i, :]

            # Calculate the proportion of NAs in a sample
            prop_na = current_sample.isna().sum() / current_sample.size

            # If 0 NaN's continue. If all NaN's, can't do quantile sampling.
            # If only one value, then there will be Inf/-Inf when imputing
            if prop_na in [0, 1] or current_sample.notna().sum() == 1:
                continue

            # Inverse cumulative distribution function of N(0, 1)
            q_norm = norm.ppf(np.arange(prop_na + err, upper_q + err, (upper_q - prop_na) / (upper_q * 100)))

            # Compute quantiles with same sample size as q_norm
            q_current_sample = np.quantile(
                current_sample.loc[~current_sample.isna()], np.arange(err, upper_q + err * 2, 1e-2)
            )

            # Check that sample sizes are consistent
            if q_norm.size != q_current_sample.size:
                diff = max(q_norm.size, q_current_sample.size) - min(q_norm.size, q_current_sample.size)
                if q_norm.size > q_current_sample.size:
                    q_norm = q_norm[:-diff]
                else:
                    q_current_sample = q_current_sample[:-diff]

            # Basic linear model
            lm = np.polyfit(q_norm, q_current_sample, 1)
            sd_cdd, mean_cdd = lm
            sd_cdd = abs(sd_cdd)

            # Tune sd if not N(0, 1)
            sigma = sd_cdd * tune_sigma
            upper_bound = float(norm.ppf(prop_na + err, loc=mean_cdd, scale=sd_cdd))

            if upper_bound < 0:
                upper_bound = abs(upper_bound)

            # Can't have a nan upper bound, so set to max of current sample
            if np.isnan(upper_bound):
                upper_bound = current_sample.max()

            # Left-censored data so lower bound is 0
            data_to_impute = pd.Series(
                self._truncated_norm(n_features, mean_cdd, sigma, 0, upper_bound),
                index=current_sample.index,
            )

            current_sample_imputed = current_sample.copy()

            na_idx = current_sample.loc[current_sample.isna()].index
            current_sample_imputed.loc[na_idx] = data_to_impute.loc[na_idx]
            data_imputed.iloc[i, :] = current_sample_imputed

        return data_imputed

    def _truncated_norm(self, n: int, mu: float, sigma: float, lower: float, upper: float) -> np_array:
        """Gibbs sampler for truncated multivariate normal distribution

        rnorm_trunc: https://github.com/WandeRum/GSimp

        Parameters
        ----------
        n
            The number of features.
        mu
            The mean.
        sigma
            The standard deviation.
        lower
            The lower bound.
        upper
            The upper bound.

        Returns
        -------
            Imputed data.
        """
        cdf_norm_lower = norm.cdf(lower, loc=mu, scale=sigma)
        cdf_norm_upper = norm.cdf(upper, loc=mu, scale=sigma)

        # In the rare case where all are equal, then cdf_x is np.nan
        # The current_sample values are 0 and np.nan
        if lower == upper == mu == sigma:
            cdf_norm_lower = 0
            cdf_norm_upper = 0

        uni_dist = np.random.uniform(low=cdf_norm_lower, high=cdf_norm_upper, size=n)

        return norm.ppf(uni_dist, loc=mu, scale=sigma)

    def _minprob(
        self,
        data: pd.DataFrame,
        quantile: float = 0.01,
        tune_sigma: float = 1,
        protein_na: float = 0.5,
        lower_bound: float = 0,
    ) -> pd.DataFrame:
        """Imputes missing values drawn from a gaussian distribution.

        Parameters
        ----------
        data
            A dataframe containing missing data.
        quantile, optional
            The quantile corresponding to the minimum value, by default .01
        tune_sigma, optional
            Coefficient that controls the standard deviation, by default 1
        protein_na, optional
            The percentage of protein standard deviation to filter, by default .5
        lower_bound, optional
            The lower bound an imputed value can be, by default 0

        Returns
        -------
        A complete dataframe

        Raises
        ------
        ValueError
            Negative values for protein intensity don't make sense.
        """
        if lower_bound < 0:
            raise ValueError("Lower bound must be positive.")

        n_samples, n_features = data.shape

        # If not enough features, likely to encounter an all NaN sample
        # <= 4 features can be imputed with MAR
        if n_features <= 4:
            return data

        data_imputed = data.copy()
        data_imputed.columns = list(range(len(data_imputed.columns)))

        # Get min values sample-wise based on the q-th quantile
        # Slightly differs from the R quantile function even though the method/type is the same
        min_samples = data_imputed.apply(np.nanpercentile, axis=1, raw=True, result_type=None, **{"q": quantile})

        # Percentage of observed values for each protein
        count_nas = n_samples - data_imputed.isna().sum()
        count_nas = count_nas / n_samples

        data_filtered = data_imputed.loc[:, count_nas > protein_na]

        prot_sd = data_filtered.apply(np.nanstd, axis=0)
        temp_sd = np.nanmedian(prot_sd) * tune_sigma

        for i in range(n_samples):
            mean = min_samples.iloc[i]
            # Some samples have all NaN values so continue and impute in MAR
            if np.isnan(mean):
                continue

            imputed_data = truncnorm(lower_bound, mean + temp_sd, loc=mean, scale=temp_sd).rvs(n_features)

            current_sample = data_imputed.iloc[i]
            na_idx = current_sample.isna().loc[current_sample.isna()].index
            data_imputed.iloc[i, na_idx] = imputed_data[na_idx]

        data_imputed.columns = data.columns
        return data_imputed


class MissForestParams(TypedDict, total=False):
    """Typing class for MissForest parameters."""

    pass


class MICEParams(TypedDict, total=False):
    """Typing class for MICE parameters."""

    pass


class KNNParams(TypedDict, total=False):
    """Typing class for KNN parameters."""

    pass


class ImputeMAR:
    """Imputation methods specifically for Missing At Random.

    This type of randomness is ignorable and is all the missing values which are
    validated to not be not missing at random.
    """

    def __call__(
        self,
        method: MARMethods,
        /,
        data: pd.DataFrame,
        estimator: BaseEstimator | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Indirect calling of imputation methods by specifying the `method`.

        Parameters
        ----------
        method
            The imputation method to perform.
        data
            The unimputed data.
        estimator, optional
            The estimator to use for imputation, by default None.
            Only used for MICE.
        **kwargs
            Extra parameters to pass to the relevant imputation method.

        Returns
        -------
            Imputed data.
        """
        if method == "knn":
            imputed_data = self._knn(data, **kwargs)
        elif method == "mean":
            imputed_data = self._mean(data)
        elif method == "median":
            imputed_data = self._median(data)
        elif method == "mode":
            imputed_data = self._mode(data)
        elif method == "miss_forest":
            imputed_data = self._miss_forest(data, **kwargs)
        elif method == "mice":
            estimator_ = XGBRegressor(n_estimators=500) if estimator is None else estimator
            imputed_data = self._mice(data, estimator_, **kwargs)
        else:
            imputed_data = data

        return imputed_data

    def __repr__(self) -> str:
        cls_name = type(self).__name__
        args = [f"{k}={v}" for k, v in self.__dict__.items()]
        return f"{cls_name}({', '.join(args)})"

    def _miss_forest(self, data: pd.DataFrame, **kwargs: Unpack[MissForestParams]) -> pd.DataFrame:
        """Imputes missing data with the RandomForest algorithm.

        Parameters
        ----------
        data
            A dataframe containing missing data.
        *args, **kwargs
            Extra arguments to pass to the MissForest imputer.

        Returns
        -------
            A complete dataframe.
        """
        kwargs = MissForestParams(
            **{
                key: value for key, value in kwargs.items() if key in MissForest._parameter_constraints.keys()
            }  # type: ignore
        )

        imputer = MissForest(verbose=2, **kwargs)

        return pd.DataFrame(imputer.fit_transform(data), columns=data.columns, index=data.index)

    def _knn(self, data: pd.DataFrame, **kwargs: Unpack[KNNParams]) -> pd.DataFrame:
        """Imputes missing data with uniform weighting.

        Parameters
        ----------
        data
            A dataframe containing missing data.
        *args, **kwargs
            Extra arguments to pass to the KNN imputer.

        Returns
        -------
            A complete dataframe.
        """
        kwargs = KNNParams(
            **{
                key: value for key, value in kwargs.items() if key in KNNImputer._parameter_constraints.keys()
            }  # type: ignore
        )

        # keep_empty_features will set all nan columns to 0
        imputer = KNNImputer(**kwargs, keep_empty_features=True)

        return pd.DataFrame(imputer.fit_transform(data), columns=data.columns, index=data.index)

    def _mice(
        self,
        data: pd.DataFrame,
        estimator: BaseEstimator = XGBRegressor(n_estimators=500),
        **kwargs: Unpack[MICEParams],
    ) -> pd.DataFrame:
        """Imputes missing data with the MICE algorithm.

        Parameters
        ----------
        data
            A dataframe containing missing data.
        estimator, optional
            The estimator to use for imputation, by default XGBRegressor(n_estimators=500)
        **kwargs
            Extra arguments to pass to the MICE imputer.

        Returns
        -------
            A complete dataframe.
        """
        kwargs = MICEParams(
            **{
                key: value for key, value in kwargs.items() if key in IterativeImputer._parameter_constraints.keys()
            }  # type: ignore
        )

        # No *args as estimator is the only positional arg for IterativeImputer
        imputer = IterativeImputer(estimator, **kwargs, keep_empty_features=False)

        return pd.DataFrame(cast(np_array, imputer.fit_transform(data)), columns=data.columns, index=data.index)

    def _mean(self, data: pd.DataFrame) -> pd.DataFrame:
        data_imputed = data.copy()

        return data_imputed.apply(lambda x: x.fillna(x.mean()), axis=0)

    def _median(self, data: pd.DataFrame) -> pd.DataFrame:
        data_imputed = data.copy()

        return data_imputed.apply(lambda x: x.fillna(x.median()), axis=0)

    def _mode(self, data: pd.DataFrame) -> pd.DataFrame:
        data_imputed = data.copy()

        return data_imputed.apply(lambda x: x.fillna(mode(x, nan_policy="omit", keepdims=True).mode[0]), axis=0)
