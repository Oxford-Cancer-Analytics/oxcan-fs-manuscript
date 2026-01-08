from __future__ import annotations

import logging
import time
from copy import deepcopy
from pickle import PickleError
from typing import cast
from typing import Literal
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import shap
from joblib import delayed
from joblib import Parallel
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from src.cli.cli_options import AugmentationEnum

from .._types import BaseEstimator
from .._types import np_array
from ..data_augmentation import apply_augmentation
from ..figure_generation import compute_roc_curve
from .shap import EjectTree


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..cli.toml_parser import TomlParser


class FeatureAddition:
    """Feature addition methods.

    Parameters
    ----------
    config
        The full config options.
    """

    def __init__(self, config: TomlParser) -> None:
        self.config = config

    def recursive_feature_addition(
        self,
        pipe_df: pd.DataFrame,
        *,
        model_pipeline: Pipeline,
        method: Literal["feature_importance", "model_performance"] = "feature_importance",
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Apply recursive feature addition.

        Parameters
        ----------
        pipe_df
            Protein and cancer signals.
        model_pipeline
            Pipeline which only has an optimised model.
        method
            Optional, default to "feature_importance".

        Returns
        -------
            Dictionaries of RFA and grid search results.
        """
        logger.info("Recursive Feature Addition.")
        logger.info(f"NOTE: using RFA method {method}")

        if "label" in pipe_df:
            pipe_df.drop("label", axis=1, inplace=True)

        # run the RFA methods
        if method == "feature_importance":
            rfa_results, feature_importance = self.rfa_feature_importance(
                pipe_df,
                model_pipeline,
            )
        else:  # should be "model_perf"
            rfa_results = self.rfa_model_performance(pipe_df, model_pipeline)
            feature_importance = None

        return rfa_results, feature_importance

    def rfa_feature_importance(
        self,
        df: pd.DataFrame,
        model_pipeline: Pipeline,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Funtion to run RFA following the order of feature importances.

        Parameters
        ----------
        df
            ml_ready_data and labels.
        model_pipeline
            pipeline with best parameters found in optimisation.

        Returns
        -------
        result_df
            Dataframe of features, roc_auc mean and std.
        """
        n_folds = self.config.data.feature_selection.cross_validation_splits
        n_repeats = self.config.data.feature_selection.cross_validation_repeats
        kf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=self.config.data.random_state)
        num_feat = self.config.data.feature_selection.number_of_features
        X, y = df.drop("target", axis=1), df["target"].to_numpy()

        import_mean = self._get_feature_importance(X, y, model_pipeline)

        # combine features and mean importance
        import_feat = pd.DataFrame({"features": X.columns, "importance": import_mean}).reset_index(drop=True)

        if num_feat > len(import_feat):
            num_feat = len(import_feat)  # to avoid throwing error

        # get the feature names along with feature
        # importance and roc scores sorted in descending order
        sub_imp_feat = import_feat.sort_values(by=["importance"], ascending=False).iloc[:num_feat]

        import_feat = import_feat[import_feat.features.isin(sub_imp_feat.features)]

        # run the RFA on cross validation with the mean importances
        result_metrics: dict[str, list[float]] = {
            "mean_roc_auc": [],
            "std_roc_auc": [],
            "stab_roc_auc": [],
            "mean_mcc": [],
            "std_mcc": [],
            "stab_mcc": [],
            "mean_sens@99spec": [],
            "mean_sens@95spec": [],
            "mean_sens@90spec": [],
            "mean_sens@85spec": [],
            "mean_sens@80spec": [],
        }
        for i in range(1, len(sub_imp_feat) + 1):
            # Select top i features in original order
            last_feature = sub_imp_feat.iloc[i - 1].features
            last_feature_imp = sub_imp_feat.iloc[i - 1].importance

            features = import_feat[import_feat["importance"] >= last_feature_imp]["features"]
            select_X = X[features].to_numpy()

            logger.info(
                f"Feature ({last_feature}) num {i} out of {len(sub_imp_feat)} with importance {last_feature_imp}."
            )

            with Parallel(n_jobs=-2) as parallel:
                metrics = cast(
                    list[tuple[float, np_array, float]],
                    parallel(
                        delayed(self._cross_validation)(
                            train_index,
                            val_index,
                            pd.DataFrame(select_X),
                            y,
                            model_pipeline,
                        )
                        for train_index, val_index in kf.split(select_X, y)
                    ),
                )

            roc_auc, interp_tprs, mcc = (list(s) for s in zip(*metrics))

            fprs = np.array([1, 5, 10, 15, 20])
            tprs = np.concatenate([tpr[fprs] for tpr in interp_tprs]).reshape(-1, len(fprs))

            for key, value in zip(
                result_metrics.keys(),
                [
                    np.mean(roc_auc),
                    np.std(roc_auc),
                    np.mean(roc_auc) / np.std(roc_auc),
                    np.mean(mcc),
                    np.std(mcc),
                    np.mean(mcc) / np.std(mcc),
                    *np.mean(tprs, axis=0),
                ],
            ):
                result_metrics[key].append(float(value))

            del [select_X]  # type: ignore

        result_df = pd.DataFrame({"features": sub_imp_feat.features, **result_metrics}).reset_index(drop=True)

        return result_df, import_feat[import_feat.features.isin(sub_imp_feat.features)]

    def rfa_model_performance(self, df: pd.DataFrame, model_pipeline: Pipeline) -> pd.DataFrame:
        """Funtion to run RFA following the model performance.

        Parameters
        ----------
        df
            training and label data from ml_ready_data.
        model_pipeline
            with the best parameters found in optimisation.

        Returns
        -------
        result_df
            Dataframe of features, mean and std auc roc.
        """
        # suppress xgboost stack trace verbosity
        if model_pipeline.steps[-1][-1].__module__ == "xgboost.sklearn":
            import xgboost as xgb

            xgb.set_config(verbosity=0)

        n_folds = self.config.data.feature_selection.cross_validation_splits
        n_repeats = self.config.data.feature_selection.cross_validation_repeats
        X, y = df.drop("target", axis=1), df["target"].to_numpy()

        # create the list of features to iterate on
        lookup_feat = X.columns.tolist()

        selected_feat = []
        roc_auc_mean = []
        roc_auc_std = []
        mcc_mean = []
        mcc_std = []

        # num_feat = len(lookup_feat)  # take from config
        num_feat = self.config.data.feature_selection.number_of_features
        if num_feat > len(lookup_feat):
            num_feat = len(lookup_feat)  # to avoid throwing error

        logger.info("Recursive feature addition running...")

        # fit the model on the first feature with stratified kfold
        kf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=self.config.data.random_state)
        for i in range(1, num_feat + 1):
            logger.info(f"Finding feature at position {i} out of {num_feat}")
            temp_roc_auc_mean = []  # intermediate performace results
            temp_roc_auc_std = []

            temp_mcc_mean = []
            temp_mcc_std = []

            # iterate through the lookup list of features
            for j in range(0, len(lookup_feat)):
                temp_lookup_feat = deepcopy(lookup_feat)
                temp_lookup_feat.pop(j)

                # get subset of features while keeping initial order
                select_X = X.drop(temp_lookup_feat, axis=1)
                select_X = select_X.to_numpy()

                with Parallel(n_jobs=-2) as parallel:
                    metrics = cast(
                        list[tuple[float, np_array, float]],
                        parallel(
                            delayed(self._cross_validation)(
                                train_index,
                                val_index,
                                pd.DataFrame(select_X),
                                y,
                                model_pipeline,
                            )
                            for train_index, val_index in kf.split(select_X, y)
                        ),
                    )

                roc_auc, _, mcc_list = (list(s) for s in zip(*metrics))

                del [select_X]  # type: ignore

                # get the mean /std of roc auc for feat j
                temp_roc_auc_mean.append(np.mean(roc_auc))
                temp_roc_auc_std.append(np.std(roc_auc))

                # get the mean /std of mcc for feat j
                temp_mcc_mean.append(np.mean(mcc_list))
                temp_mcc_std.append(np.std(mcc_list))

            # select the feature that provides the best performance
            # in combination with previous features
            best_feat = lookup_feat[np.argmax(temp_mcc_mean)]
            best_feat_score = np.mean(temp_mcc_mean)
            logger.info(f"Feature num {i} selected is {best_feat}, with score {best_feat_score}")

            # add the feature to the list of important features
            selected_feat.append(best_feat)

            # remove the selected feature from the lookup list
            lookup_feat.pop(np.argmax(temp_mcc_mean))

            # save the performance MCC score
            mcc_mean.append(max(temp_mcc_mean))
            mcc_std.append(temp_mcc_std[np.argmax(temp_mcc_mean)])

            # save the performance AUC score
            roc_auc_mean.append(max(temp_roc_auc_mean))
            roc_auc_std.append(temp_mcc_std[np.argmax(temp_roc_auc_mean)])

        # combine performance results with feature names
        result_df = pd.DataFrame(
            {
                "features": selected_feat,
                "mean_roc_auc": roc_auc_mean,
                "std_roc_auc": roc_auc_std,
                "mean_mcc": mcc_mean,
                "std_mcc": mcc_std,
            }
        )

        return result_df

    def find_most_import_feat(self, feat_df: pd.DataFrame, metric: str = "mean_mcc") -> pd.DataFrame:
        """Funtion to find a cutoff to subset important features.

        Parameters
        ----------
        feat_df
            Sorted feature dataframe.
        metric, optional
            The metric used to select the feature set, by default "mean_mcc".

        Returns
        -------
            Subset of features.
        """
        feat_df = feat_df.set_index("features")

        # subset features to where the model reaches max performance
        max_metric_feat = feat_df.idxmax()[metric]  # mcc as test

        # get most important features
        most_imp_feat = feat_df.loc[:max_metric_feat].reset_index()

        return most_imp_feat

    def _compute_importance(
        self,
        train_index: np_array,
        X: pd.DataFrame,
        y: pd.DataFrame,
        model_pipeline: Pipeline,
        augment: AugmentationEnum,
        method: Literal["shap", "tree_importance", "mutual_information"],
    ) -> np_array:
        """Gathers an array of feature importances from a specific method.

        The method is specified as `method`.

        Parameters
        ----------
        train_index
            Indices for the training data.
        X
            The feature values.
        y
            The target values.
        model_pipeline
            The model as a `Pipeline` object.
        augment
            The type of data augmentation to apply. If "none", no augmentation is applied.
        method
            The method to compute feature importance. Options are "shap", "tree_importance", and "mutual_information".

        Returns
        -------
        An array with feature importances.
        """
        X_train, y_train = X.loc[train_index, :], y[train_index]  # type: ignore

        if augment != "none":
            X_train, y_train = apply_augmentation(
                X_train,
                y_train,
                augment,
                self.config.data.augmentation.smote.new_cancer_ratio,
                self.config.data.augmentation.smote.new_total_samples,
                self.config.data.random_state,
            )

        if method in ["shap", "tree_importance"]:
            model_pipeline.fit(X_train, y_train)

        # shap or eject_shap
        if "shap" in method:
            # Calculate SHAP values
            if "shap" == method:
                explainer = shap.TreeExplainer(model_pipeline[-1])
                shap_values = explainer.shap_values(X_train)
            else:
                now = time.time()
                shap_values = EjectTree(model_pipeline[-1]).shap_values(X_train)
                logger.info(f"Time taken for EjectShapley computation: {time.time() - now}")

            # Get the shap_values of the target class
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]
            elif shap_values.ndim > 2:  # type: ignore
                shap_values = shap_values[:, :, 1]  # type: ignore

            import_mean_temp = np.abs(shap_values).mean(axis=0)

        elif method == "tree_importance":
            # Tree based models
            import_mean_temp = model_pipeline[-1].feature_importances_  # type: ignore

        elif method == "mutual_information":
            # Model independent
            import_mean_temp = mutual_info_classif(X_train, y_train, random_state=self.config.data.random_state)

        return import_mean_temp

    def _get_feature_importance(self, X: pd.DataFrame, y: np_array, model_pipeline: Pipeline) -> np_array:
        """Gathers an array of feature importances from a specific method.

        The method is specified in the config as `importance_selection`.

        Parameters
        ----------
        X
            The feature values.
        y
            The target values.
        model_pipeline
            The model as a `Pipeline` object.

        Returns
        -------
            An array of feature importance values.
        """
        method = self.config.data.feature_selection.addition.importance_selection
        kf = RepeatedStratifiedKFold(
            n_splits=self.config.data.feature_selection.cross_validation_splits,
            n_repeats=self.config.data.feature_selection.cross_validation_repeats,
            random_state=self.config.data.random_state,
        )
        augment = self.config.cli_data.augmentation

        match method:
            case "shap" | "eject_shap" | "tree_importance" | "mutual_information":
                n_jobs = 1 if method == "eject_shap" else -2

                if n_jobs == 1:
                    logger.info(f"No parallelization as using {method}")

                # Parallel processing
                with Parallel(n_jobs=n_jobs) as parallel:
                    important_features = cast(
                        list[np_array],
                        parallel(
                            delayed(self._compute_importance)(train_index, X, y, model_pipeline, augment, method)
                            for train_index, _ in kf.split(X.to_numpy(), y)
                        ),
                    )

                import_mean = np.mean(important_features, axis=0)
            case "permutation_importance":
                model_pipeline.fit(X, y)
                import_mean = compute_permutation_importance(
                    X.to_numpy(),
                    y,
                    model_pipeline[-1],  # type: ignore
                    seed=self.config.data.random_state,
                    metric="average_precision",
                )

        return import_mean

    def _cross_validation(
        self,
        train_index: np_array,
        val_index: np_array,
        X: pd.DataFrame,
        y: np_array,
        model_pipeline: Pipeline,
    ) -> tuple[float, np_array, float]:
        """Runs one cross validation fold.

        Parameters
        ----------
        train_index
            indexes of the training set.
        val_index
            indexes of the validation set.
        X
            The feature values.
        y
            The target values.
        model_pipeline
            pipeline with the best parameters.

        Returns
        -------
            ROC AUC value.
        """
        X_train, X_val = X.loc[train_index, :], X.loc[val_index, :]  # type: ignore
        y_train, y_val = y[train_index], y[val_index]  # type: ignore

        if self.config.cli_data.augmentation != "none":
            X_train, y_train = apply_augmentation(
                X_train,
                y_train,
                self.config.cli_data.augmentation,
                self.config.data.augmentation.smote.new_cancer_ratio,
                self.config.data.augmentation.smote.new_total_samples,
                self.config.data.random_state,
            )
        # refit the entire pipeline on the training set
        refitted_pipeline = model_pipeline.fit(X_train, y_train)

        # evaluate the fitted model
        y_scores = refitted_pipeline.predict_proba(X_val)

        roc_val = roc_auc_score(y_val, y_scores[:, 1])
        roc_metrics = compute_roc_curve([y_scores[:, 1]], [y_val], np.linspace(0, 1, 101))
        interp_tprs = roc_metrics["interp_tprs"][0]

        y_pred = refitted_pipeline.predict(X_val)
        mcc = matthews_corrcoef(y_val, y_pred)

        return float(roc_val), interp_tprs, mcc


def compute_permutation_importance(
    X: np_array, y: np_array, fitted_model: type[BaseEstimator], seed: int = 42, metric: str = "average_precision"
) -> np_array:
    """Computes permutation importance on a pre-fitted model.

    Parameters
    ----------
    X
        Data on which to compute importance.
    y
        Labels on which to compute importance.
    fitted_model
        Fitted model object.
    seed, optional
        Random seed for permutation feature importance, by default 42.
    metric, optional
        The type of metric to use, by default 'average_precision'.

    Returns
    -------
        Array of feature importances.
    """
    try:
        logger.info("Parrallelising permutation feature importance run")
        perm_imp = permutation_importance(
            fitted_model,
            X,
            y,
            scoring=metric,
            n_jobs=-2,
            n_repeats=10,
            random_state=seed,
        )
    except (PickleError, TypeError):
        # Can't parallelise permutation feature importance
        logger.warning(
            """\
            Could not parallelise model.
            Permutation feature importance is calculated on single core. This might take some time.
        """
        )
        perm_imp = permutation_importance(
            fitted_model,
            X,
            y,
            scoring=metric,
            n_jobs=None,
            n_repeats=10,
            random_state=seed,
        )

    return perm_imp.importances_mean  # type: ignore
