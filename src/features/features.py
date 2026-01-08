from __future__ import annotations

import logging
from collections.abc import MutableMapping
from copy import deepcopy
from pathlib import Path
from typing import Any
from typing import cast
from typing import Literal
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .._types import np_array
from ..data_augmentation import apply_augmentation
from ..data_loader import load_data
from ..data_loader import load_pipeline_file
from ..data_loader import reassign_labels
from ..data_loader import save_deliverables
from ..data_preparation import DataPreparation
from ..figure_generation import plot_optuna_figures
from ..statistical_analysis import apply_t_test_minimal_features
from ..statistical_analysis import get_significant_features
from .addition import FeatureAddition
from .elimination import FeatureElimination
from .optimisation import ModelOptimisation
from .performance_metrics import bootstrap_performance_95ci
from .performance_metrics import calculate_mi_scores
from .performance_metrics import cluster_matrix
from .performance_metrics import cross_validation_performance_95ci
from .performance_metrics import ModelMetrics
from .performance_metrics import ROCThreshold
from .selection import FeatureSelection

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..cli.toml_parser import TomlParser


class Features:
    """Main entry class for FeatureX classes.

    Parameters
    ----------
    config
        The full config options.

    Attributes
    ----------
    selection
        Feature selection functionality handler.
    addition
        Feature addition functionality handler.
    elimination
        Feature elimination functionality handler.
    optimisation
        Model optimization functionality handler.
    data_preparation
        Data preparation functionality handler.
    model_metrics
        Model metrics calculation handler.
    feature_identifiers
        Dictionary storing feature identifier information.
    """

    def __init__(self, config: TomlParser) -> None:
        self.config = config

        self.selection = FeatureSelection(config, self)
        self.addition = FeatureAddition(config)
        self.elimination = FeatureElimination(config, self)
        self.optimisation = ModelOptimisation(config)
        self.data_preparation = DataPreparation(config)
        self.model_metrics = ModelMetrics(config)

        self.feature_identifiers: dict[str, list[Any]] = {}

        self.__thresh_names: dict[str, ROCThreshold] = {}

    # Function to evaluate model performance and create deliverables
    def evaluate_and_create_deliverables(
        self,
        fs_results: MutableMapping[str, pd.DataFrame | pd.Series[float] | Pipeline | None],
        selected_feats: pd.DataFrame,
        train_df: pd.DataFrame,
        holdout_df: pd.DataFrame,
        name: str,
        prefix: str,
    ) -> None:
        """Evaluate model performance and create deliverables.

        Parameters
        ----------
        fs_results
            A dictionary containing feature selection results, including feature importance.
        selected_feats
            DataFrame containing selected features for the model.
        train_df
            DataFrame representing the training dataset.
        holdout_df
            DataFrame representing the holdout/test dataset.
        name
            Name or identifier for the model being evaluated.
        prefix
            Prefix to be used in the output file paths for saving deliverables.
        """
        logger.info(f"Running evaluation and creating deliverables for {prefix} and {name}.")
        y_scores, X_valid, X_train, fitted_pipeline, grid_results = self.model_validation(
            train_df, holdout_df, selected_feats, model_only=True
        )
        logger.info(f"Completed model evaluation on {prefix} test set.")

        # Create model deliverables
        selected_features_with_importance = selected_feats.merge(
            cast(pd.DataFrame, fs_results["feature_importance"]).reset_index(drop=True), on="features"
        )

        # Ensure the optimal threshold is the same across semi-holdout and holdout
        if name in self.__thresh_names:
            threshold_class = self.__thresh_names[name].reset()
            optimal_threshold = threshold_class.youdens_j
        else:
            threshold_class = ROCThreshold(X_valid["target"].to_numpy(), y_scores)
            optimal_threshold = threshold_class.youdens_j
            self.__thresh_names[name] = threshold_class

        results, figures = self.model_metrics.create_model_deliverables(
            y_scores,
            X_valid,
            X_train,
            fitted_pipeline,
            selected_features_with_importance,
            optimal_threshold=optimal_threshold,
            ttest=True if hasattr(selected_feats, "p_value") else False,
        )

        thresholds = [*threshold_class]
        performance_metrics = self.model_metrics.evaluate_performance(
            valid_df=X_valid, y_scores=y_scores, thresholds=thresholds
        )
        save_deliverables(
            self.config, csv_dict=dict(performance_metrics), model_subfolder=f"{prefix}/{name}/performances/"
        )

        if self.optimisation.study:
            figures = plot_optuna_figures(self.optimisation.study, self.optimisation.scorers, figures)

        results["model_pipeline"] = fitted_pipeline  # type: ignore
        results["grid_results"] = grid_results

        # Adding CV for 95% confidence intervals of key metrics using full dataset
        sel_f = ["target"] + selected_feats.features.to_list()
        full_df = pd.concat([train_df[sel_f], holdout_df[sel_f]], axis=0).reset_index(drop=True)

        X_full = full_df.drop(columns=["target"])
        y_full = full_df["target"]

        results["CV_full_data_95CI"] = cross_validation_performance_95ci(
            model_object=fitted_pipeline,
            random_state=self.config.data.random_state,
            X=X_full,
            y=y_full,
            threshold=optimal_threshold,
        )

        # Adding Bootstrap for 95% confidence intervals of key metrics using holdout
        results["Bootstrapped_holdout_95CI"] = bootstrap_performance_95ci(
            model_object=fitted_pipeline,
            random_state=self.config.data.random_state,
            X_holdout=holdout_df[sel_f].drop(columns=["target"]),
            y_holdout=holdout_df[sel_f]["target"],
            threshold=optimal_threshold,
        )

        # Save all deliverables at once
        save_deliverables(self.config, csv_dict=results, figure_dict=figures, model_subfolder=f"{prefix}/{name}")
        logger.info("Completed performance analysis.")

    def run_pipeline(self) -> None:  # pragma: no cover
        """Runs the entire Feature pipeline."""
        fs_results: MutableMapping[str, pd.DataFrame | pd.Series[float] | Pipeline | None] = (
            {}
        )  # csv files of the RFS results
        fs_optimise: MutableMapping[str, Pipeline] = {}  # pickled fitted pipeline

        labelled_cohort, train, holdout, feature_identifiers = load_data(self.config)
        self.feature_identifiers = feature_identifiers

        if self.config.cli_data.prepare_data:
            # prepare dataframe for preprocessing optimisation
            X_sub, fs_results["feat_mi_importance"] = self._preprocess_features(
                train.drop(columns=["patient_id", "label", "batch"], errors="ignore")
            )

            pipeline, param_grid = self.optimisation.init_model(self.config.cli_data.model)

            # fs_results["model_pipeline"] is used for RFA but currently uses a
            # default model that has not been hyperparameter optimised
            fs_optimise["fitted_pipeline"] = fs_results["model_pipeline"] = self.data_preparation.get_best_pipeline(
                X_sub,
                pipeline,
                param_grid,
            )
            return
        elif self.config.cli_data.best_pipeline != Path(""):
            # Load best pipeline using intelligent mode detection (S3 or local)
            best_full_pipeline = load_pipeline_file(self.config, str(self.config.cli_data.best_pipeline))
            fs_optimise["fitted_pipeline"] = best_full_pipeline

            X_sub = train.drop(columns=["patient_id", "label", "batch"], errors="ignore")
            assert holdout is not None
            # if data has missing values perform imputation and batch correction
            if train.isna().any().any():
                logger.info("Imputation of training data")
                train = self._impute(train, best_full_pipeline)

                logger.info("Imputation of holdout data")
                holdout = self._impute(holdout, best_full_pipeline, transform_only=True)

                save_deliverables(
                    self.config,
                    csv_dict={
                        "train": train,
                        "holdout": holdout,
                    },
                    model_subfolder="imputed",
                )
                # reset output path if needed
            else:
                logger.info("No imputation needed")

            if self.config.cli_data.use_batch_corrected:
                logger.info("Batch correction of training vs holdout data separately")
                # batch correction on training data only
                train = self._batch_correct(train, labelled_cohort)
                # batch correction on holdout data only
                holdout = self._batch_correct(holdout, labelled_cohort)

                # save the batch corrected training vs holdout data
                save_deliverables(
                    self.config,
                    csv_dict={
                        "train": train,
                        "holdout": holdout,
                    },
                    model_subfolder="batch_corrected",
                )
                logger.info(f"Batch correction complete saved to S3 path: {self.config.cli_data.s3_output_path}")
            else:
                logger.info("No batch correction performed")

            logger.info("Running model optimisation")
            X_sub, fs_results["feature_mi_importance"], fs_results["model_pipeline"], fs_results["grid_results"] = (
                self._model_optimisation(train, optimize=self.config.data.optimization.optuna.before_feature_selection)
            )

            # subset important features from training and holdout data
            train = train[X_sub.columns.to_list() + ["patient_id", "label"]]
            holdout = holdout[X_sub.columns.to_list() + ["patient_id", "label"]]
        else:
            logger.info("Running model optimisation only")
            X_sub, fs_results["feature_mi_importance"], fs_results["model_pipeline"], fs_results["grid_results"] = (
                self._model_optimisation(train, optimize=self.config.data.optimization.optuna.before_feature_selection)
            )

            # subset important features from training and holdout data
            train = train[X_sub.columns.to_list() + ["patient_id", "label"]]
            if holdout is not None:
                holdout = holdout[X_sub.columns.to_list() + ["patient_id", "label"]]

        figures = plot_optuna_figures(self.optimisation.study, self.optimisation.scorers, dict())

        if self.config.cli_data.rec_feat_add:
            recursive_df, fs_results["feature_importance"] = self.addition.recursive_feature_addition(
                X_sub,
                model_pipeline=fs_results["model_pipeline"],
                method=self.config.data.feature_selection.addition.method,
            )

            if self.config.data.optimization.optuna.after_rfa:
                RFA_features = fs_results["feature_importance"]["features"].to_list()  # type: ignore
                RFA_features = RFA_features + ["target", "patient_id", "label"]
                (
                    X_sub,
                    fs_results["feature_mi_importance"],
                    fs_results["model_pipeline"],
                    fs_results["grid_results"],
                ) = self._model_optimisation(
                    train[RFA_features], optimize=self.config.data.optimization.optuna.after_rfa
                )

                RFA_figures = plot_optuna_figures(self.optimisation.study, self.optimisation.scorers, dict())

                figures |= RFA_figures

                recursive_df, fs_results["feature_importance"] = self.addition.recursive_feature_addition(
                    X_sub,
                    model_pipeline=fs_results["model_pipeline"],
                    method=self.config.data.feature_selection.addition.method,
                )

            logger.info("Preparing to save the deliverables.")
        else:
            recursive_df = self.elimination.recursive_feature_elimination(
                X_sub,
                estimator=fs_results["model_pipeline"],  # type: ignore
                feature_range=range(40, 0, -1),
            )

            logger.info("Applying t-test to reduce features for RFE.")

        obs = (
            self.config.data.feature_selection.cross_validation_splits
            * self.config.data.feature_selection.cross_validation_repeats
            if self.config.cli_data.rec_feat_add
            else self.config.data.feature_selection.multisurf_cross_validation_splits
        )

        # mi_score correlaton matrices
        if len(X_sub.columns) > 1000:
            logger.info(f"There are {len(X_sub.columns)} > 1000 features. Skipping Corr matrices")
        elif len(X_sub.columns) > 1:
            logger.info(f"There are {len(X_sub.columns)} features. Computing Corr matrices.")
            mi_score = calculate_mi_scores(X_sub)
            mi_score_cluster = cluster_matrix(mi_score)
            fs_results["all_mi_corr_data"] = mi_score.reset_index(names="features")
            fs_results["all_mi_corr_data_clustered"] = mi_score_cluster.reset_index(names="features")

            spearman_score = X_sub.corr(method="spearman")
            spearman_score_cluster = cluster_matrix(spearman_score)
            fs_results["spearman_corr_data"] = spearman_score.reset_index(names="features")
            fs_results["spearman_corr_data_clustered"] = spearman_score_cluster.reset_index(names="features")
        else:
            logger.info(f"There are {len(X_sub.columns)} <= 1 features. Skipping Corr matrices")

        selected_features_model = {}
        model_name = self.config.cli_data.model.value

        metrics_name = ["mean_roc_auc", "mean_mcc", "stab_mcc"]
        for metric_name in metrics_name:
            # fit metrics
            key = f"max_{metric_name}_features_{model_name}"
            selected_features_model[key] = self.addition.find_most_import_feat(recursive_df, metric=metric_name)

            if "stab" in metric_name:
                continue

            # Fit the t-test values
            all_ttest_key = f"all_ttest_{metric_name}_performance_features_{model_name}"
            metric = "_".join(metric_name.split("_")[1:])
            ttest_df = fs_results[all_ttest_key] = apply_t_test_minimal_features(
                recursive_df, metric=metric, observations=obs
            )

            # Filter features based on p-value and maximum index
            ttest_key = f"ttest_{metric_name}_performance_features_{model_name}"
            selected_features_model[ttest_key] = ttest_df[
                (ttest_df[f"p_value_{metric}"] <= self.config.data.stats.pvalue_thresh)
                & (ttest_df.index <= ttest_df.idxmax(numeric_only=True)[f"p_value_{metric}"])
            ]

        # save feature selection results
        save_deliverables(
            self.config,
            csv_dict=fs_results,
            figure_dict=figures,
            pickled_dict=fs_optimise,
            include_metadata=True,
            model_subfolder="validation",
        )

        logger.info("Completed feature selection.")

        if holdout is not None:
            # Create the semi df needed to perform the semi_holdout, using only train data
            split_holdout_semi = holdout.shape[0] / train.shape[0]
            train_semi_df, holdout_semi_df = cast(
                tuple[pd.DataFrame, pd.DataFrame],
                train_test_split(
                    train,
                    test_size=split_holdout_semi,
                    random_state=self.config.data.random_state,
                    stratify=reassign_labels(train.copy())["label"],
                ),
            )

            for name, selected_feats in selected_features_model.items():
                if "ttest" in name and selected_feats.empty:
                    logger.info(f"No ttest features selected for model {name}, all features above p > 0.05")
                    continue

                sel_f = ["patient_id", "label", "target"] + selected_feats.features.to_list()

                # Evaluate model performance on semi-holdout test set
                self.evaluate_and_create_deliverables(
                    fs_results,
                    selected_feats,
                    train_semi_df[sel_f].reset_index(drop=True),
                    holdout_semi_df[sel_f].reset_index(drop=True),
                    name,
                    prefix="validation/semi_holdout",
                )

                # Evaluate model performance on holdout test set
                self.evaluate_and_create_deliverables(
                    fs_results,
                    selected_feats,
                    train[sel_f],
                    holdout[sel_f],
                    name,
                    prefix="holdout",
                )

    def model_validation(
        self, train_set: pd.DataFrame, holdout: pd.DataFrame, selected_features: pd.DataFrame, model_only: bool
    ) -> tuple[np_array, pd.DataFrame, pd.DataFrame, Pipeline, pd.DataFrame]:
        """Evaluates the model accuracy with a holdout validation dataset.

        Parameters
        ----------
        train_set
            The training data.
        holdout
            The holdout validation data.
        selected_features
            The selected features to validate on.
        model_only
            To only optimise the model.

        Returns
        -------
            Evalutes the model on the holdout test set.
        """
        # prepare dataframe for preprocessing optimisation
        # Data has to be imputed for KDE to work
        if self.config.cli_data.augmentation != "none":
            X_preaugmented = train_set[selected_features.features.to_list() + ["patient_id", "label", "target"]]
            logger.info("Augmentation of training data prior to final model optimisation")
            # Only need to augment training for model evaluation
            X_augmented = self._augmentation(X_preaugmented, step="model_validation")
            train_set = X_augmented.drop(columns=["batch"], errors="ignore")

        y_train = train_set["target"]

        if self.config.cli_data.rec_feat_add:
            # subset columns to selected features for RFA
            # keep selected features in initial order
            exclude_feat = np.setdiff1d(train_set.columns, selected_features.features, assume_unique=True)
            keep_feat = np.setdiff1d(train_set.columns, exclude_feat, assume_unique=True)
            X_train = train_set[keep_feat].copy()
            X_valid = holdout[keep_feat].copy()
        else:
            # subset columns to selected features for RFE
            n = selected_features.n_features_to_select.to_numpy()[0]
            (feat_dict,) = selected_features.loc[selected_features.n_features_to_select == n].features_chosen
            feat_df = pd.DataFrame(feat_dict).sort_values(by="features_importances", ascending=False)
            subset_features = feat_df.iloc[:n].index.to_numpy().tolist()
            X_train = train_set.loc[:, subset_features]
            X_valid = holdout.loc[:, subset_features]

        # optimise model with selected features
        train_df = deepcopy(X_train)
        train_df["target"] = y_train
        optimize = self.config.data.optimization.optuna.before_validation
        model_pipeline, grid_results = self.optimisation.get_best_pipeline(
            train_df, model_type=self.config.cli_data.model, model_only=model_only, optimize=optimize
        )

        # fit the best model on training set
        fitted_pipeline = model_pipeline.fit(X_train, y_train)
        # predict outcome on holdout set
        y_scores = fitted_pipeline.predict_proba(X_valid)

        X_train["patient_id"], X_train["target"], X_train["label"] = (
            train_set["patient_id"],
            train_set["target"],
            train_set["label"],
        )
        X_valid["patient_id"], X_valid["target"], X_valid["label"] = (
            holdout["patient_id"],
            holdout["target"],
            holdout["label"],
        )

        return y_scores, X_valid, X_train, fitted_pipeline, grid_results

    def _preprocess_features(self, train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Filters data based on missingness

        Parameters
        ----------
        train
            training data containing target column

        Returns
        -------
        X_sub
            Data containing only features determined to be important in training data
            during model optimisation
        feature_mi_importance
            Data frame containing importance for each feature across all data
        """
        train_no_target = train.drop(columns=["target"])

        if self.config.cli_data.feature_selection in ("MutualInformation", "BorutaShap"):
            feature_mi_importance = self.selection.get_important_features(train).reset_index(names="features")
            X_sub = train.loc[:, feature_mi_importance.features.tolist()]
        else:
            important_features = self.selection.get_important_features(train)

            X_sub = train_no_target.loc[:, important_features.importance_count >= self.elimination.filter_number]
            feature_mi_importance = None

        logger.info(
            f"Feature preprocessing decreased protein count from {train_no_target.shape[1]} to {X_sub.shape[1]}"
        )
        if train_no_target.shape[1] - X_sub.shape[1] > 0:  # pragma: no cover
            logger.info(f"Selected features: {set(train_no_target.columns) & set(X_sub.columns)}")

        X_sub["target"] = train.target
        return X_sub, feature_mi_importance

    def _apply_stats(
        self,
        X: np_array,
        y: np_array,
        protein_list: list[str | int],
    ) -> tuple[np_array, list[str], pd.DataFrame]:
        """Apply statistical tests on the data.

        Parameters
        ----------
        X
            The full dataset.
        y
            The data labels.
        protein_list
            The list of protein features.

        Returns
        -------
            A subset of data and features.
        """
        logger.info("Statistical filtering.")
        ttest_results = get_significant_features(X, y, protein_list)

        X_sub = X[
            :, ttest_results.pvalues < self.config.data.stats.pvalue_thresh
        ]  # take subset of features from this test
        protein_list_sub = ttest_results[
            ttest_results.pvalues < self.config.data.stats.pvalue_thresh
        ].features.to_list()
        logger.info(f"Statistical filter decreased protein count from {X.shape} to {X_sub.shape}")

        return X_sub, protein_list_sub, ttest_results

    def _impute(
        self,
        data: pd.DataFrame,
        best_pipeline: Pipeline,
        transform_only: bool = False,
    ) -> pd.DataFrame:
        """Imputes data for full_data, training and holdout separately

        Parameters
        ----------
        data
            A dataframe containing missing values to be imputed
        best_pipeline
            The pipeline specified for model
            Includes imputation method to be used
        transform_only, optional
            To transform the dataset instead of fitting first, by default False.

        Returns
        -------
        imp_data
            Dataframe with missing values imputed
        """
        imp_data = deepcopy(data).drop(columns=["patient_id", "label", "target"])
        for _, step in best_pipeline.steps[:-1]:
            if transform_only:
                imp_data = step.transform(imp_data)
            else:
                imp_data = step.fit_transform(imp_data)

        imp_data.columns = data.drop(columns=["patient_id", "label", "target"]).columns
        imp_data.columns.name = None

        for key in ["patient_id", "target", "label"]:
            imp_data[key] = data[key]

        return imp_data

    def _batch_correct(
        self,
        data: pd.DataFrame,
        labelled_cohort: pd.DataFrame,
        batch: str = "plate",
    ) -> pd.DataFrame:
        """Batch correction of imputed data

        Parameters
        ----------
        data
            A dataframe containing no missing values to be batch corrected
        labelled_cohort
            Dataframe containing patient information including column used in batch correction.
        batch
            Column to use in batch correction.
            Default to "plate" but could be a different column to batch correct in future.

        Returns
        -------
        batch_corrected_data
            Dataframe with batch correction applied
        """
        from combat.pycombat import pycombat

        batch_data = labelled_cohort[["patient_id", batch]].merge(data, on="patient_id")
        batch_only = batch_data[batch]
        batch_corrected_data = pycombat(
            batch_data.drop(columns=["patient_id", "target", "label", batch]).T, batch_only
        ).T

        for key in ["patient_id", "target", "label"]:
            batch_corrected_data[key] = batch_data[key]

        return batch_corrected_data

    def _augmentation(
        self,
        data: pd.DataFrame,
        step: Literal["cross_validation", "model_validation"],
    ) -> pd.DataFrame:
        """Performs augmentation of data and returns correct dataframe

        Parameters
        ----------
        data
            The training data containing important features
        step
            The stage of preprocessing, tells function where to save data to S3.

        Returns
        -------
        augmented_data
            Augmented data frame of important features
        """
        patient_ids = pd.DataFrame(data["patient_id"]).reset_index()
        X_aug, y_aug = apply_augmentation(
            data.drop(columns=["patient_id", "label", "target"]),
            data["target"].to_numpy(),
            self.config.cli_data.augmentation,
            self.config.data.augmentation.smote.new_cancer_ratio,
            self.config.data.augmentation.smote.new_total_samples,
            self.config.data.random_state,
        )
        augmented_data = (
            pd.concat([X_aug.reset_index(), pd.Series(y_aug, name="target")], axis=1)
            .merge(patient_ids, on="index", how="left")
            .drop("index", axis=1)
        )
        augmented_data["label"] = data.label
        augmented_data.loc[augmented_data.patient_id.isna(), "patient_id"] = pd.Series(
            [f"AUG{i}" for i in range(augmented_data.patient_id.isna().sum())],
            index=augmented_data.loc[augmented_data.patient_id.isna()].index,
        )
        augmented_data.loc[augmented_data.label.isna(), "label"] = pd.Series(
            [f"AUG{i}" for i in range(augmented_data.label.isna().sum())],
            index=augmented_data.loc[augmented_data.label.isna()].index,
        )

        if step == "cross_validation":
            model_subfolder = "augmented_cv"
        elif step == "model_validation":
            model_subfolder = "augmented_model_validation"

        save_deliverables(
            self.config,
            csv_dict={
                "augmented_train": augmented_data,
            },
            model_subfolder=model_subfolder,
        )

        return augmented_data

    def _model_optimisation(
        self,
        data: pd.DataFrame,
        optimize: bool,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None, Pipeline, pd.DataFrame]:
        """Optimisation of model on a subset of important features

        Parameters
        ----------
        data
            training data to be used in model optimisation
        optimize
            A bolean flag to know if to use a default model or run optimization.

        Returns
        -------
        X_sub
            Training data containing only important features
        feat_mi_importance
            Results from mutual information importance
        results_be
            best estimator
        results_grid
            grid search results
        """
        # prepare dataframe for preprocessing optimisation
        X_sub, feat_mi_importance = self._preprocess_features(
            data.drop(columns=["patient_id", "label", "batch"], errors="ignore")
        )
        train = X_sub

        # prepare dataframe for preprocessing optimisation
        # Data has to be imputed for KDE or SMOTE to work
        if self.config.cli_data.augmentation != "none":
            X_preaugmented = data[X_sub.columns.to_list() + ["patient_id", "label"]]
            logger.info("Augmentation of training data prior to feature selection")

            # Only need to augment training for model evaluation
            X_augmented = self._augmentation(X_preaugmented, step="cross_validation")
            X_sub = X_augmented.drop(columns=["patient_id", "label", "batch"], errors="ignore")

            original_ratio = sum(X_preaugmented.target) / len(X_preaugmented.target)
            original_total = len(X_preaugmented.target)
            original_cancer = len(X_preaugmented.loc[X_preaugmented.target == 1].target)
            original_control = len(X_preaugmented.loc[X_preaugmented.target == 0].target)

            final_ratio = sum(X_augmented.target) / len(X_augmented.target)
            final_total = len(X_augmented.target)
            final_cancer = len(X_augmented.loc[X_augmented.target == 1].target)
            final_control = len(X_augmented.loc[X_augmented.target == 0].target)

            logger.info(f"Augmented data flag: using {self.config.cli_data.augmentation}")
            logger.info(f"Augmented data flag: changed cancer proportion from {original_ratio} to {final_ratio}")
            logger.info(f"Augmented data flag: changed total samples from {original_total} to {final_total}")
            logger.info(f"Augmented data flag: changed total cancer patients from {original_cancer} to {final_cancer}")
            logger.info(
                f"Augmented data flag: changed total control patients from {original_control} to {final_control}"
            )

        logger.info(f"Running {'an optuna' if optimize else 'a fixed'} model before feature selection.")
        results_be, results_grid = self.optimisation.get_best_pipeline(
            X_sub, model_type=self.config.cli_data.model, model_only=True, optimize=optimize
        )

        return train, feat_mi_importance, results_be, results_grid
