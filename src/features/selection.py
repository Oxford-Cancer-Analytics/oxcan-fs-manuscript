from __future__ import annotations

import logging
from typing import cast
from typing import TYPE_CHECKING
from typing import TypeAlias

import numpy as np
import pandas as pd
from BorutaShap import BorutaShap
from joblib import delayed
from joblib import Parallel
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import ShuffleSplit
from skrebate import MultiSURF
from xgboost import XGBClassifier

from .._types import np_array
from ..data_augmentation import apply_augmentation
from .models import UnivariateFeature

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..cli.toml_parser import TomlParser
    from .features import Features

ReliefTypes: TypeAlias = MultiSURF | UnivariateFeature


def effective_dimension_pca(X: pd.DataFrame) -> float:
    """Compute the effective dimension of a dataset using PCA eigenvalues.

    Parameters
    ----------
    X
        Pandas DataFrame of numerical features.

    Returns
    -------
        The estimated effective dimension of the dataset.
    """
    # Compute the covariance matrix
    cov_matrix = np.cov(X, rowvar=False)

    # Compute the eigenvalues
    eigenvalues = np.linalg.eigvals(cov_matrix)

    # Compute the absolute values of the eigenvalues
    all_abs_eigenvalues = np.abs(eigenvalues)

    # To avoid having to take the limit x * log x when x = 0 numerically
    abs_eigenvalues = [a for a in all_abs_eigenvalues if a != 0]

    # Normalize by the sum of all eigenvalues' absolute values
    weights = abs_eigenvalues / np.sum(abs_eigenvalues)

    # Compute the entropy
    entropy = -np.sum(weights * np.log(weights))

    # Effective dimension as the exponent of the entropy
    effective_dim = np.exp(entropy)

    return effective_dim


def calculate_mutual_info_cv_parallel(
    X: pd.DataFrame, y: np_array, random_state: int, n_splits: int, n_repeats: int
) -> pd.Series[float]:
    """
    Calculate mutual information for each feature using cross-validated parallel computation.

    Parameters
    ----------
    X
        The feature matrix.
    y
        The target variable.
    random_state
        Random seed for reproducibility.
    n_splits
        Number of splits in each iteration of cross-validation.
    n_repeats
        Number of times cross-validation is repeated.

    Returns
    -------
        A Series containing the mean mutual information scores for each feature.
    """
    # Initialize RepeatedStratifiedKFold
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    # Define a function for parallel computation of mutual information for each fold
    def compute_mutual_info(
        train_index: list[int], test_index: list[int], X: pd.DataFrame, y: np_array
    ) -> pd.DataFrame:
        X_train, _ = X.iloc[train_index], X.iloc[test_index]
        y_train, _ = y[train_index], y[test_index]

        # Replace missing values with zeros in the training set
        X_train_filled = X_train.fillna(0)

        # Compute mutual information on the filled training set
        importances_fold = mutual_info_classif(X_train_filled, y_train, random_state=random_state)

        # Create a DataFrame of feature importances incorporating missingness
        mutual_info_fold_df = pd.DataFrame(
            importances_fold * (1 - X_train.isnull().sum() / X_train.shape[0]), index=X.columns
        )

        return mutual_info_fold_df

    with Parallel(n_jobs=-2) as parallel:
        # Parallelize the loop over folds
        importances_list = cast(
            list[pd.DataFrame],
            parallel(
                delayed(compute_mutual_info)(train_index, test_index, X, y)
                for train_index, test_index in cv.split(X, y)
            ),
        )

    # Concatenate the individual DataFrames into one
    mutual_info_df = pd.concat(importances_list, axis=1).mean(axis=1)

    return mutual_info_df


def permute_dataframe(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """This function takes a DataFrame and returns a permuted DataFrame

    The values in each column have been randomly permuted.

    Parameters
    ----------
    df
        The original DataFrame to be permuted.
    seed, optional
        The random seed used, by default 0.

    Returns
    -------
        A DataFrame with the same structure as the original but with
        values in each column permuted.
    """
    # Use the apply method to permute each column independently
    np.random.seed(seed)
    permuted_df = df.apply(
        np.random.permutation,
    )
    return permuted_df


class FeatureSelection:
    """Models for feature selection.

    Parameters
    ----------
    config
        The full config options.
    features
        The features class.
    """

    def __init__(self, config: TomlParser, features: Features) -> None:
        self.config = config
        self.features = features

    def get_important_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Important features from relief based algorithms or Mutual Information.

        Parameters
        ----------
        df
            Training dataset.

        Returns
        -------
            Important features summary.
        """
        available_models = {
            "UnivariateFeature": UnivariateFeature,
            "MultiSURF": MultiSURF,
        }

        user_selected_model = self.config.cli_data.feature_selection
        number_of_mi_features = self.config.data.feature_selection.number_of_mi_features
        number_of_features = self.config.data.feature_selection.number_of_features

        logger.info(f"Using {user_selected_model} feature preprocessing model.")

        y = df["target"].to_numpy()
        X = df.drop(columns=["target"])
        features = X.columns

        effective_dim = effective_dimension_pca(X.fillna(0))
        logger.info(f"Effective dimension dataset:{effective_dim}")
        if number_of_mi_features == "effective_dim":
            number_of_mi_features = max(int(np.ceil(effective_dim)), number_of_features)

        if len(features) <= self.config.data.feature_selection.number_of_features:
            number_of_mi_features = 0

        if user_selected_model == "BorutaShap":

            xgb_model = XGBClassifier(random_state=self.config.data.random_state)
            boruta_shap = BorutaShap(model=xgb_model, importance_measure="shap", classification=True)
            boruta_shap.fit(X=X, y=y, n_trials=200, random_state=self.config.data.random_state)

            return (
                pd.DataFrame(
                    {"feature_importance": boruta_shap.X_feature_import},
                    index=boruta_shap.Subset(tentative=True).columns,
                )
                .reindex(X.columns)
                .dropna()
            )

        if user_selected_model == "MutualInformation":
            # Using Mutual Information
            if self.config.cli_data.prepare_data:
                logger.info("Replacing NAs with zeros for mutual informaton")

            mutual_info_df = calculate_mutual_info_cv_parallel(
                X,
                y,
                random_state=self.config.data.random_state,
                n_splits=2,
                n_repeats=25,
            )

            if isinstance(number_of_mi_features, int) and number_of_mi_features > 1:
                logger.info(f"Selecting top {number_of_mi_features} mutual informaton features")
                best_features = mutual_info_df.nlargest(number_of_mi_features)
            elif isinstance(number_of_mi_features, float) and number_of_mi_features >= 0.5:
                logger.info(
                    f"Since number_of_mi_features {number_of_mi_features} >= 0.5 we use it as MI noise quantile"
                )
                logger.info(f"Selecting top features above {100*number_of_mi_features} th quantile of noise")
                X_perm = permute_dataframe(X.fillna(0), seed=self.config.data.random_state)
                importances_permuted = mutual_info_classif(X_perm, y, random_state=self.config.data.random_state)
                mutual_info_df_permuted = pd.Series(
                    importances_permuted * (1 - X_perm.isnull().sum() / X_perm.shape[0]), index=features
                )
                # for now we choose a Quantile as the threshold
                mutual_info_threshold = mutual_info_df_permuted.quantile(number_of_mi_features)
            else:
                logger.info(f"Since number_of_mi_features {number_of_mi_features} < 0.5 we use it as MI threshold")
                logger.info(f"Selecting top features above {number_of_mi_features} MI score")
                mutual_info_threshold = number_of_mi_features
                best_features = mutual_info_df.loc[mutual_info_df >= mutual_info_threshold]
                logger.info(
                    f"There are {best_features.shape[0]} features above the MI threshold {mutual_info_threshold}"
                )

            return (
                pd.DataFrame({"feature_importance": best_features.values}, index=best_features.index)
                .reindex(X.columns)
                .dropna()
            )

        # Using other models
        feature_selection_model = available_models.get(user_selected_model, UnivariateFeature)

        model = feature_selection_model(n_features_to_select=X.shape[1], n_jobs=-2)

        augment_flag = self.config.cli_data.augmentation

        if augment_flag != "none":  # augment the data
            X, y = apply_augmentation(
                X,
                y,
                augment_flag,
                self.config.data.augmentation.smote.new_cancer_ratio,
                self.config.data.augmentation.smote.new_total_samples,
                self.config.data.random_state,
            )

        # Cross-validation
        importance_list = self.relief_cross_validation(model, X, y)

        # Summary
        full_result = pd.DataFrame(importance_list, columns=features)

        # Important features list
        threshold = 0
        best_features = (full_result > threshold).sum(axis=0)

        return pd.DataFrame(best_features, columns=["importance_count"])

    def relief_cross_validation(self, model: ReliefTypes, X: pd.DataFrame, y: np_array) -> list[list[float]]:
        """Cross validation for Relief models.

        Parameters
        ----------
        model
            Skrebate model.
        X
            Train protein signal.
        y
            Train cancer label.

        Returns
        -------
            List of feature importances.
        """
        importance_list = []
        n_splits = self.config.data.feature_selection.multisurf_cross_validation_splits
        kf = ShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=self.config.data.random_state)
        logger.info(f"Running k={n_splits} fold cross validation.")
        for k, (train_index, _) in enumerate(kf.split(X)):
            logger.info(f"Fold {k}")
            X_train, y_train = X.iloc[train_index], y[train_index]
            _ = model.fit(X_train.to_numpy(), y_train)
            importance_list.append(model.feature_importances_)

        return importance_list
