from __future__ import annotations

import logging
from collections.abc import Hashable
from typing import Any
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.model_selection import ShuffleSplit

from .._types import BaseEstimator
from .._types import np_array
from .performance_metrics import youdens_j

if TYPE_CHECKING:
    from ..cli.toml_parser import TomlParser
    from .features import Features


logger = logging.getLogger(__name__)


class FeatureElimination:
    """Feature elimination methods.

    Parameters
    ----------
    config
        The full config options.
    features
        The features class.

    Attributes
    ----------
    filter_number
        Number of filters calculated from multisurf cross validation splits.
    metric_keys
        List of metric keys used for evaluation.
    """

    def __init__(self, config: TomlParser, features: Features) -> None:
        self.config = config
        self.features = features

        _fs = config.data.feature_selection
        self.filter_number = int(_fs.multisurf_cross_validation_splits / 2)

        self.metric_keys: list[str]

    # Normal RFE
    def recursive_feature_elimination(
        self,
        pipe_df: pd.DataFrame,
        *,
        estimator: type[BaseEstimator],
        feature_range: range = range(2, 3),
    ) -> pd.DataFrame:
        """Call the sklearn RFE.

        Generate optimum features for a range of n features.

        Parameters
        ----------
        pipe_df
            Protein data and labels.
        estimator
            Model class.
        feature_range, optional
            range of features to choose, by default range(2,3).

        Returns
        -------
            Number of features, mean roc auc, std roc auc, features.
            chosen dict
        """
        logger.info("Recursive Feature Elimination.")
        roc_auc_summary = [
            self._rfe_cross_validation(pipe_df, n_features_to_select, estimator=estimator)
            for n_features_to_select in feature_range
        ]

        # summarize cross valiation results here
        metric_keys = (
            np.array([[f"{calc}_{key}" for calc in ["mean", "std"]] for key in self.metric_keys]).flatten().tolist()
        )
        result = pd.DataFrame(
            roc_auc_summary,
            columns=[
                "n_features_to_select",
                *metric_keys,
                "features_chosen",
            ],
        )
        logger.info("Completed recursive feature elimination")
        return result

    def map_feature_importance(
        self,
        support: np_array,
        feature_importances: np_array,
    ) -> list[float]:
        """Maps feature importances to support values of different shapes.

        Parameters
        ----------
        support
            The support mask of features selected.
        feature_importances
            The feature importanes of all features.

        Returns
        -------
            Each feature importance value of all features, not just
            selected features.
        """
        assert len(feature_importances) == sum(support)
        res = []
        enum_iter: list[int] = []
        for i, mask in enumerate(support):
            if i in enum_iter:
                continue
            if mask:
                res.append(feature_importances[len(enum_iter)])
                enum_iter.append(i)
            else:
                res.append(0.0)
        return res

    def _rfe_cross_validation(
        self,
        pipe_df: pd.DataFrame,
        n_features_to_select: int,
        estimator: type[BaseEstimator],
    ) -> list[float | dict[Hashable, Any]]:
        """Cross validation step, can use Ray to parallelize.

        Parameters
        ----------
        pipe_df
            Protein data and labels.
        n_features_to_select
            N features to choose.
        estimator
            Model class.

        Returns
        -------
            Cross validation metrics and metric names.
        """
        n_splits = self.config.data.feature_selection.cross_validation_splits  # override for debugging
        kf = ShuffleSplit(
            n_splits=n_splits,
            test_size=0.3,
            random_state=self.config.data.random_state,
        )
        support_list, metrics_list = [], []
        X, y = pipe_df.drop("target", axis=1).to_numpy(), pipe_df["target"].to_numpy()
        protein_list = pipe_df.drop("target", axis=1).columns.tolist()
        feature_importances: dict[str, list[float]] = {protein: [] for protein in protein_list}
        for k, (train_index, test_index) in enumerate(kf.split(X)):
            logger.info(f"fold: {k}")
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # sklearn recursive feature elimination
            rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select)  # type: ignore
            rfe.fit(X_train, y_train)

            # capture proteins chosen and feature importances at this fold
            support_list.append(rfe.support_)

            if hasattr(rfe.estimator_, "feature_importances_"):
                importances = rfe.estimator_.feature_importances_  # type: ignore
            else:
                # Check for coef_ and normalise values between 0-1
                coef = rfe.estimator_.coef_  # type: ignore
                # Normalize between [0,1]
                importances = ((coef - np.min(coef)) / (np.max(coef) - np.min(coef))).flatten()
            mapped_fi = self.map_feature_importance(rfe.support_, importances)

            # This assumes that all the features are in the same order as
            # protein_list
            for protein, fi in zip(feature_importances, mapped_fi):
                feature_importances[protein].append(fi)

            # capture all metrics
            metrics, _ = self.features.model_metrics.compute_model_performance(
                rfe.predict_proba(X_test),
                y_test,
                threshold=youdens_j(y_test, rfe.predict_proba(X_test)),
            )
            metrics_list.append(metrics)

        logger.info(f"Cross validation for features to select {n_features_to_select} complete.")

        # convert list of proteins chosen per fold into a dict
        average_selected_features = pd.DataFrame(support_list, columns=protein_list).mean()._set_name("count")
        average_features_importances = (
            pd.DataFrame(feature_importances, columns=protein_list).mean()._set_name("features_importances")
        )
        averaged_features = pd.concat([average_selected_features, average_features_importances], axis=1)
        average_selected_features_dict: dict[Hashable, Any] = averaged_features[
            averaged_features["count"] > 0
        ].to_dict()

        self.metric_keys = list(metrics_list[0].keys())[:-1]
        return [
            n_features_to_select,
            *self._calculate_metrics(metrics_list),
            average_selected_features_dict,
        ]

    def _calculate_metrics(self, metrics: list[dict[str, float | np_array]]) -> list[float]:
        keys = list(metrics[0].keys())[:-1]
        metric_values: list[tuple[float, float]] = [
            (
                np.mean([metric[key] for metric in metrics]),  # type: ignore
                np.std([metric[key] for metric in metrics]),  # type: ignore
            )
            for key in keys
        ]
        return [metric for values in metric_values for metric in values]
