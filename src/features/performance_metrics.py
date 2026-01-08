from __future__ import annotations

import io
import logging
from collections.abc import Callable
from copy import deepcopy
from typing import Any
from typing import cast
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from joblib import delayed
from joblib import Parallel
from lime.explanation import Explanation
from lime.lime_tabular import LimeTabularExplainer
from scipy.cluster.hierarchy import leaves_list
from scipy.cluster.hierarchy import linkage
from shap import Explanation as ShapExplanation
from sklearn import metrics
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from typing_extensions import Self

from .._types import BaseEstimator
from .._types import np_array
from ..cli.toml_parser import TomlParser
from ..figure_generation import compute_roc_curve
from ..figure_generation import plot_feature_importance
from ..figure_generation import plot_feature_vs_x
from ..figure_generation import plot_roc_curve


logger = logging.getLogger(__name__)


class Explain:
    """Explainability class.

    Parameters
    ----------
    training
        The training data.
    """

    def __init__(self, training: pd.DataFrame) -> None:
        self.training = training.drop(columns=["patient_id", "label"], errors="ignore")

    def lime(
        self,
        test_instance: pd.Series[float],
        predict_fn: Callable[[Any], Any],
        *args: Any,
        **kwargs: Any,
    ) -> tuple[io.BytesIO, Explanation]:
        """Apply LIME explainer.

        Parameters
        ----------
        test_instance
            The instance to explain.
        predict_fn
            A fitted classifiers predict function.
        *args
            Any additional arguments to include for instance explanation.
        **kwargs
            Any additional arguments to include for instance explanation.

        Returns
        -------
            A figure and the explanation instance.
        """
        train, train_targets = self.training.drop(columns=["target"]), self.training.target
        explainer = LimeTabularExplainer(
            train.values,
            training_labels=train_targets,
            feature_names=train.columns.values.tolist(),
            discretize_continuous=True,
        )
        exp = explainer.explain_instance(
            test_instance.drop("target"),
            predict_fn=predict_fn,
            top_labels=1,
            *args,
            **kwargs,
        )

        fig = io.BytesIO()
        figure = exp.as_pyplot_figure(exp.available_labels()[0])

        figure.savefig(fig, format="png", bbox_inches="tight")

        return fig, exp

    def shap(
        self, model: BaseEstimator, testing: pd.DataFrame
    ) -> tuple[dict[str, dict[str, list[tuple[str, io.BytesIO]]]], list[pd.DataFrame]]:
        """Apply Shapley explainability.

        The training data is used as background data in the Shapley explainer.
        Shapley values foreground data uses the holdout test data. From this,
        several plots are created:
            - bar (global)
            - beeswarm (global)
            - force (local)
            - interactions

        Parameters
        ----------
        model
            The fitted model to explain.
        testing
            The holdout test data, including the target, to gather shapley values.

        Returns
        -------
            A mapping of image objects and shapley values for each case.
        """
        identifiers = ["full", *self.training["target"].unique().astype(str)]
        testing = testing.drop(columns=["label"], errors="ignore")
        shap_objs = {}
        shap_value_objs = []

        explainer = shap.TreeExplainer(
            model,
            data=self.training.drop(columns=["target"], axis=1),
            feature_perturbation="interventional",
            model_output="probability",
        )
        patient_ids = testing.patient_id
        for identifier in identifiers:
            if identifier != "full":
                holdout = testing.loc[testing["target"] == int(identifier)].reset_index(drop=True)
                identifier = f"class_{identifier}"
                patient_ids = holdout.patient_id
            else:
                holdout = testing.reset_index(drop=True).copy()

            X_test = holdout.drop(columns=["target", "patient_id"])
            shap_explainer = explainer(X_test)

            # Sklearn returns both labels
            if len(shap_explainer.shape) > 2:
                shap_explainer = cast(ShapExplanation, shap_explainer[:, :, 1])

            global_plots = [
                self._shap_plot(shap_explainer, global_plot, plot_type="global") for global_plot in ["bar", "beeswarm"]
            ]

            # Global force plot as html document
            img_data = io.BytesIO()
            temp_data = io.StringIO()
            force_plot = shap.plots.force(shap_explainer, show=False)
            shap.save_html(temp_data, force_plot)
            img_data.write(bytes(temp_data.getvalue(), "utf-8"))
            global_plots.append(("force_html", img_data))
            plt.clf()

            local_samples = [
                holdout.loc[holdout.target == target_value].sample(n=1) for target_value in holdout.target.unique()
            ]
            local_plots = [
                self._shap_plot(
                    shap_explainer,
                    "force",
                    plot_type="local",
                    local_sample=local_sample,
                    **{"matplotlib": True, "text_rotation": -45, "figsize": (20, 5)},
                )
                for local_sample in local_samples
            ]

            shap_values = pd.DataFrame(shap_explainer.values, columns=X_test.columns)
            shap_values["patient_id"] = patient_ids
            shap_objs[identifier] = {
                "global": global_plots,
                "local": local_plots,
            }
            shap_value_objs.append(shap_values)
            plt.close("all")

        return shap_objs, shap_value_objs

    def _shap_plot(
        self,
        explainer: ShapExplanation,
        plot: str,
        plot_type: Literal["global", "local"],
        local_sample: pd.DataFrame | None = None,
        **plot_kwargs: Any,
    ) -> tuple[str, io.BytesIO]:
        """Creates local and global shapley plots.

        HTML-based shapley plots are not included in this method.

        Parameters
        ----------
        explainer
            The SHAPley Explainer.
        plot
            The kind of shapley plot.
        plot_type, {"global", "local"}
            To use all samples or a single sample.
        local_sample, optional
            A single sample to use for local `plot_type`'s. This is ignored for
            `plot_type`=global, by default None.
        **plot_kwargs
            Any additional arguments to pass to the plotting function.

        Returns
        -------
            A tuple of the name of the plot and the image data as a BytesIO object.
        """
        img_data = io.BytesIO()

        plot_fn = eval(f"shap.plots.{plot}")

        if plot_type == "local":
            assert isinstance(local_sample, pd.DataFrame)
            local_pick = local_sample.index[0]
            local_target = local_sample.target.loc[local_pick]
            plot_fn(explainer[local_pick], show=False, **plot_kwargs)
        else:
            plot_fn(explainer, show=False, max_display=30)

        plt.savefig(img_data, format="png", bbox_inches="tight")
        plt.clf()

        if plot_type == "local":
            ret_val = (f"{plot}_{local_target}", img_data)
        else:
            ret_val = (f"{plot}", img_data)

        return ret_val


class ROCThreshold:
    """ROC Thresholding

    Parameters
    ----------
    y_true
        Ground truth labels.
    y_scores
        Predicted scores.
    """

    def __init__(self, y_true: np_array, y_scores: np_array) -> None:
        self.y_true = y_true
        self.y_scores = y_scores

        _, roc_data = plot_roc_curve([y_scores[:, 1]], [y_true])
        self._roc_data = roc_data

        self.__properties = [prop for prop, value in ROCThreshold.__dict__.items() if isinstance(value, property)]
        self.__idx = 0

    @property
    def youdens_j(self) -> float:
        """Computes the optimal threshold.

        Returns
        -------
            Threshold.
        """
        return youdens_j(y_true=self.y_true, y_scores=self.y_scores)

    @property
    def default(self) -> float:
        """Computes the default threshold.

        Returns
        -------
            Threshold.
        """
        return 0.5

    @property
    def sens_at_99_specificity(self) -> float:
        """Computes the threshold where sensitivity is at 99 specificity.

        Returns
        -------
            Threshold.
        """
        return self._roc_data["interp_thresh"][0][1]

    @property
    def sens_at_95_specificity(self) -> float:
        """Computes the threshold where sensitivity is at 95 specificity.

        Returns
        -------
            Threshold.
        """
        return self._roc_data["interp_thresh"][0][5]

    @property
    def sens_at_90_specificity(self) -> float:
        """Computes the threshold where sensitivity is at 90 specificity.

        Returns
        -------
            Threshold.
        """
        return self._roc_data["interp_thresh"][0][10]

    @property
    def sens_at_85_specificity(self) -> float:
        """Computes the threshold where sensitivity is at 85 specificity.

        Returns
        -------
            Threshold.
        """
        return self._roc_data["interp_thresh"][0][15]

    @property
    def sens_at_80_specificity(self) -> float:
        """Computes the threshold where sensitivity is at 80 specificity.

        Returns
        -------
            Threshold.
        """
        return self._roc_data["interp_thresh"][0][20]

    def reset(self) -> Self:
        """Resets the index count of the thresholds.

        Only used if the index count increases from 0.

        Returns
        -------
            Self.
        """
        self.__idx = 0

        return self

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> tuple[float, str]:
        try:
            prop = self.__properties[self.__idx]
            item = (eval(f"self.{prop}"), prop)
        except IndexError:
            raise StopIteration()

        self.__idx += 1
        return item


def youdens_j(y_true: np_array, y_scores: np_array) -> float:
    """Compute the optimal decision threshold based on the ROC curve.

    Parameters
    ----------
    y_true
        True binary labels.
    y_scores
        Predicted scores.

    Returns
    -------
        Optimal cut-off (threshold).
    """
    false_positive_rate, true_positive_rate, threshold = metrics.roc_curve(y_true, y_scores[:, 1])
    optimal_threshold = threshold[np.argmax(true_positive_rate - false_positive_rate)]
    logger.info(f"The optimal threshold was computed to be {optimal_threshold}.")

    return optimal_threshold


def probability_to_label(
    y_scores: np_array,
    y_true: np_array,
    threshold: float | None,
) -> np_array:
    """Convert probability scores into labels using a decision threshold.

    If threshold is not given, then y_scores and y_true must be the
    training sets.

    Parameters
    ----------
    y_scores
        Probability scores.
    y_true
        Ground truth labels.
    threshold
        The decision threshold.

    Returns
    -------
       Predicted labels.
    """
    if threshold is None:
        threshold = youdens_j(y_true, y_scores)

    predicted_labels = (y_scores[:, 1] >= threshold).astype(int)

    return predicted_labels


class ModelMetrics:
    """Model metrics.

    Parameters
    ----------
    config
        The full config options.
    """

    def __init__(self, config: TomlParser) -> None:
        self.config = config

    def compute_model_performance(
        self,
        y_scores: np_array,
        y_true: np_array,
        y_train_scores: np_array | None = None,
        y_train_true: np_array | None = None,
        threshold: float | None = None,
    ) -> tuple[dict[str, float | np_array], np_array]:
        """Compute model performance using probability scores and ground truth labels.

        Parameters
        ----------
        y_scores
            Probability scores.
        y_true
            Ground truth labels.
        y_train_scores, optional
            Probability training scores, by default None.
        y_train_true, optional
            Ground truth training labels, by default None.
        threshold, optional
            The decision threshold, by default None. If None, then `y_train_scores`
            and `y_train_true` must be specified.

        Returns
        -------
            Various performance metrics and predicted labels.

        Raises
        ------
        ValueError
            If binary and `y_train_scores` or `y_train_true` is not provided with `threshold`
            as None.
        """
        return compute_model_performance(
            y_scores=y_scores,
            y_true=y_true,
            y_train_scores=y_train_scores,
            y_train_true=y_train_true,
            threshold=threshold,
        )

    def create_model_deliverables(
        self,
        y_scores: np_array,
        valid_df: pd.DataFrame,
        train_df: pd.DataFrame,
        fitted_pipeline: Pipeline,
        selected_features: pd.DataFrame,
        optimal_threshold: float,
        ttest: bool = False,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, io.BytesIO]]:
        """Produces model deliverables.

        In the form of:
        - balanced accuracy, sensitivity, specificity
        - plots for ROC curve
        - plots Shapley explainer
        - plots Shapley explainer for individual observations

        Parameters
        ----------
        y_scores
            probability scores on holdout set.
        valid_df
            dataframe of holdout set, with patient_id and target columns.
        train_df
            dataframe of training set, with patient_id and target columns.
        fitted_pipeline
            optimised pipeline with selected features.
        selected_features
            subset of best features.
        optimal_threshold
            A threshold used to classify binary labels.
        ttest, optional
            Whether the data is the ttest subset, by default False.

        Returns
        -------
            Metrics based on the confusion matrix and figures.
        """
        return create_binary_deliverables(
            y_scores=y_scores,
            valid_df=valid_df,
            train_df=train_df,
            fitted_pipeline=fitted_pipeline,
            features_df=selected_features,
            optimal_threshold=optimal_threshold,
            ttest=ttest,
        )

    def evaluate_performance(
        self, valid_df: pd.DataFrame, y_scores: np_array, thresholds: list[tuple[float, str]]
    ) -> list[tuple[str, pd.DataFrame]]:
        """Performance evaluation at different thresholds.

        Parameters
        ----------
        valid_df
            Validation data, including the target.
        y_scores
            Predicted scores.
        thresholds
            A list of thresholds in the form (value, name).

        Returns
        -------
            A list of keys and performance dataframes according to the thresholds.
        """
        return evaluate_performance(valid_df=valid_df, y_scores=y_scores, thresholds=thresholds)


def combine_misclassifications(
    performance_metrics: pd.DataFrame,
    valid_df: pd.DataFrame,
    y_scores: np_array,
    threshold: float,
) -> pd.DataFrame:
    """Combines misclassifications of the validation dataset with performance metrics.

    Parameters
    ----------
    performance_metrics
        The performance metrics for the specific threshold.
    valid_df
        The validation data, including the target.
    y_scores
        The predicted scores of the validation data.
    threshold
        The threshold used for the predictions.

    Returns
    -------
        Misclassified patients combined with the performance metrics.
    """
    y_true_valid = valid_df["target"].to_numpy()
    predicted_labels = probability_to_label(y_scores, y_true_valid, threshold)

    # find label of misclassified observations
    misclassified = deepcopy(valid_df)
    misclassified["predicted"] = predicted_labels
    misclassified = misclassified.loc[misclassified["predicted"] != misclassified["target"]]

    # combine metrics
    misclassified_patients = misclassified[["patient_id", "label"]].sort_values("label").reset_index(drop=True)
    combined_df = pd.DataFrame(
        np.vstack(
            [
                performance_metrics,
                np.full(performance_metrics.shape[1], np.nan),
                misclassified_patients.T.reset_index().T,
            ]
        ),
        columns=performance_metrics.columns,
    )

    return combined_df


def evaluate_performance(
    valid_df: pd.DataFrame, y_scores: np_array, thresholds: list[tuple[float, str]]
) -> list[tuple[str, pd.DataFrame]]:
    """Performance evaluation at different thresholds.

    Parameters
    ----------
    valid_df
        Validation data, including the target.
    y_scores
        Predicted scores.
    thresholds
        A list of thresholds in the form (value, name).

    Returns
    -------
        A list of keys and performance dataframes according to the thresholds.
    """
    constant_metrics = {}
    y_true = valid_df["target"].to_numpy()
    _, roc_fig_data = plot_roc_curve([y_scores[:, 1]], [y_true])

    for sensitivity in [80, 85, 90, 95, 99]:
        constant_metrics[f"sen@{sensitivity}spec"] = roc_fig_data[f"sens_spec_{sensitivity}"][0]  # type: ignore

    results: list[tuple[str, pd.DataFrame]] = []
    for threshold, thresh_name in thresholds:
        metrics, _ = compute_model_performance(y_scores=y_scores, y_true=y_true, threshold=threshold)
        metrics |= constant_metrics

        # Use interpolated roc_auc rather than the one calculated with probabilities to match roc_auc figure
        metrics["roc_auc"] = roc_fig_data["mean_auc"]
        metrics["threshold"] = threshold

        metrics_df = pd.DataFrame.from_dict(dict(metrics), orient="index", columns=["value"])
        metrics_df.insert(0, "metric", metrics_df.index)
        metrics_df = metrics_df.reset_index(drop=True)

        metrics_and_misclassifications = combine_misclassifications(metrics_df, valid_df, y_scores, threshold)

        results.append((f"{thresh_name}_threshold", metrics_and_misclassifications))

    return results


def compute_model_performance(
    y_scores: np_array,
    y_true: np_array,
    y_train_scores: np_array | None = None,
    y_train_true: np_array | None = None,
    threshold: float | None = None,
) -> tuple[dict[str, float | np_array], np_array]:
    """Compute model performance using probability scores and ground truth labels.

    # This function is explicitly used for binary classification.

    Parameters
    ----------
    y_scores
        Probability scores.
    y_true
        Ground truth labels.
    y_train_scores, optional
        Probability training scores, by default None.
    y_train_true, optional
        Ground truth training labels, by default None.
    threshold, optional
        The decision threshold, by default None. If None, then `y_train_scores`
        and `y_train_true` must be specified.

    Returns
    -------
        Various performance metrics and predicted labels based on a threshold.

    Raises
    ------
    ValueError
        If  `y_train_scores` or `y_train_true` is not provided with `threshold`
        as None.
    """
    # Compute Youdens J Statistic if threshold is not provided
    if threshold is None:
        if (y_train_scores is None) or (y_train_true is None):
            raise ValueError(f"If {threshold=}, both y_train_scores and training labels must be specified.")
        threshold = youdens_j(y_train_true, y_train_scores)

    # Convert scores to crisp labels
    predicted_labels = probability_to_label(y_scores=y_scores, y_true=y_true, threshold=threshold)
    precision, recall, _ = metrics.precision_recall_curve(y_true, predicted_labels, pos_label=1)

    # Calculate all metrics, except roc_auc, at a given threshold
    # roc_auc must be calculated across all thresholds for probabilities!
    model_performance_dict = {
        "balanced_accuracy": float(metrics.balanced_accuracy_score(y_true, predicted_labels)),
        "f1_score": float(metrics.f1_score(y_true, predicted_labels)),
        "sensitivity": float(metrics.recall_score(y_true, predicted_labels, pos_label=1)),
        "specificity": float(metrics.recall_score(y_true, predicted_labels, pos_label=0)),
        "pos_pred_value": float(metrics.precision_score(y_true, predicted_labels, pos_label=1)),
        "neg_pred_value": float(metrics.precision_score(y_true, predicted_labels, pos_label=0)),
        "average_precision": float(metrics.average_precision_score(y_true, predicted_labels)),
        "roc_auc": float(metrics.roc_auc_score(y_true, y_scores[:, 1])),
        "pr_auc": float(metrics.auc(recall, precision)),
        "confusion_matrix": cast(np_array, metrics.confusion_matrix(y_true, predicted_labels)),
        "mcc": metrics.matthews_corrcoef(y_true, predicted_labels),
    }

    return model_performance_dict, predicted_labels


# MI corr for selected features
def calculate_mi_scores(df: pd.DataFrame, random_state: int = 0) -> pd.DataFrame:
    """Calculate mutual information scores between features in a DataFrame.

    Parameters
    ----------
    df
        Input DataFrame containing features.
    random_state, optional
        Random state for reproducibility, by default 0.

    Returns
    -------
        DataFrame with mutual information scores.
    """

    def compute_mi(i: int, j: int) -> tuple[int, int, float]:
        if i == j:
            return i, j, 1  # Assuming max MI with itself for clustering purpose
        mi = mutual_info_regression(df.iloc[:, [i]], df.iloc[:, j], random_state=random_state)
        return i, j, mi[0]

    n_features = df.shape[1]
    indices = [(i, j) for i in range(n_features) for j in range(i, n_features)]

    # Parallelize the computation of mutual information scores
    results = cast(
        list[tuple[int, int, float]],
        Parallel(n_jobs=-2)(delayed(compute_mi)(i, j) for i, j in indices),
    )

    # Initialize the mutual information scores matrix
    mi_scores = np.zeros((n_features, n_features))
    for i, j, mi in results:
        mi_scores[i, j] = np.sqrt(1 - np.exp(-2 * mi))
        mi_scores[j, i] = np.sqrt(1 - np.exp(-2 * mi))
        mi_scores[i, i] = 1

    # Create DataFrame from mutual information scores matrix
    mi_matrix = pd.DataFrame(mi_scores, index=df.columns, columns=df.columns)

    # Move target for easy comparison
    if "target" in mi_matrix.columns:
        new_index = mi_matrix.index.tolist()
        new_index.remove("target")
        new_index = ["target"] + new_index
        mi_matrix = mi_matrix.reindex(index=new_index, columns=new_index)

    return mi_matrix


def cluster_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Cluster df using hierarchical clustering ward linkage.

    Parameters
    ----------
    df
        Mutual information matrix containing scores between features.

    Returns
    -------
        DataFrame with clustered scores.
    """
    # Compute hierarchical clustering using Ward linkage
    linkage_matrix = linkage(df, method="ward")

    # Get the ordered indices of leaves in the hierarchical clustering tree
    ordered_indices = leaves_list(linkage_matrix)

    # Rearrange rows and columns of the DataFrame based on the clustering structure
    clustered_mi_matrix = df.iloc[ordered_indices, :].iloc[:, ordered_indices]

    # move target for easy comparison
    if "target" in df.columns:
        new_index = clustered_mi_matrix.index.tolist()
        new_index.remove("target")
        new_index = ["target"] + new_index
        clustered_mi_matrix = clustered_mi_matrix.reindex(index=new_index, columns=new_index)

    return clustered_mi_matrix


def create_binary_deliverables(
    y_scores: np_array,
    valid_df: pd.DataFrame,
    train_df: pd.DataFrame,
    fitted_pipeline: Pipeline,
    features_df: pd.DataFrame,
    optimal_threshold: float,
    ttest: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict[str, io.BytesIO]]:
    """Produces model performance metrics for binary classification.

    Parameters
    ----------
    y_scores
        Probability scores on validation data.
    valid_df
        Validation data.
    train_df
        Training data.
    fitted_pipeline
        A fitted model pipeline on the training data.
    features_df
        Selected feature dataframe for the current model.
    optimal_threshold
        A threshold used to classify binary labels.
    ttest, optional
        Whether the data is the ttest subset, by default False.

    Returns
    -------
        Model performance deliverables as metrics and figures.
    """
    results, figures = {}, {}

    y_true_valid = valid_df["target"].to_numpy()
    X_train = train_df.drop(["patient_id", "target", "label"], axis=1)

    # mi_score correlaton matrices
    if len(X_train.columns) > 1000:
        logger.info(f"There are {len(X_train.columns)} > 1000 features. Skipping Corr matrices.")
    elif len(X_train.columns) > 1:
        logger.info(f"There are {len(X_train.columns)} features. Computing Corr matrices.")
        mi_score = calculate_mi_scores(X_train)
        mi_score_cluster = cluster_matrix(mi_score)
        results["all_mi_corr_data"] = mi_score.reset_index(names="features")
        results["all_mi_corr_data_clustered"] = mi_score_cluster.reset_index(names="features")

        spearman_score = X_train.corr(method="spearman")
        spearman_score_cluster = cluster_matrix(spearman_score)
        results["spearman_corr_data"] = spearman_score.reset_index(names="features")
        results["spearman_corr_data_clustered"] = spearman_score_cluster.reset_index(names="features")
    else:
        logger.info(f"There are {len(X_train.columns)} <= 1 features. Skipping Corr matrices.")

    # bar plot of most important feature
    figures["figure_feature_importance"] = plot_feature_importance(
        features_df.features.to_list(),
        np.array(features_df.importance),
        min(len(features_df), 20),
    )

    # plot features vs roc-auc values
    figures["figure_features_vs_roc-auc"] = plot_feature_vs_x(features_df, y_label="ROC AUC")

    (performance,) = evaluate_performance(valid_df, y_scores, [(optimal_threshold, "youdens_j")])

    _, metrics_df = performance
    results["performance"] = metrics_df

    figures["figure_roc_curve"], roc_fig_data = plot_roc_curve([y_scores[:, 1]], [y_true_valid])

    # roc_fig_data needs to be a dataframe or series to be saved onto s3
    results["roc_fig_data"] = pd.concat(
        [
            pd.Series(roc_fig_data["mean_tprs"], name="mean_tprs"),
            pd.Series(roc_fig_data["interp_tprs"][0], name="interp_tprs"),  # type: ignore
        ],
        axis=1,
    )

    explain = Explain(train_df)
    shap_results, shap_values = explain.shap(fitted_pipeline[-1], valid_df)
    for idx, shap_key in enumerate(shap_results):
        results[f"shap_values_{shap_key}"] = shap_values[idx]

        for shap_explain in shap_results[shap_key]:
            figures |= {
                f"shap_{shap_key}_{shap_explain}_{plot}": img_obj
                for plot, img_obj in shap_results[shap_key][shap_explain]
            }

    if ttest:
        results_new: dict[str, pd.DataFrame] = {}
        figures_new: dict[str, io.BytesIO] = {}
        for result_dict, result_dict_new in zip([results, figures], [results_new, figures_new]):
            for key, value in result_dict.items():
                result_dict_new[f"{key}_ttest"] = value
        results = results_new
        figures = figures_new

    return results, figures


def train_and_evaluate_fold(
    train_index: list[int],
    test_index: list[int],
    X: pd.DataFrame,
    y: pd.DataFrame,
    model_object: Pipeline,
    threshold: float,
) -> tuple[np_array, np_array, float, float, float]:
    """Helper function to evaluate the folds for the 95% CI in parallel

    Parameters
    ----------
    train_index
        Indices of the training data.
    test_index
        Indices of the testing data.
    X
        Feature matrix.
    y
        Target variable.
    model_object
        Scikit-learn pipeline containing the model.
    threshold
        The optimal threshold to predict labels.

    Returns
    -------
        - Predicted probabilities for the positive class.
        - True labels for the test set.
        - Positive predictive value (precision) for the test set.
        - Negative predictive value for the test set.
        - Matthews correlation coefficient for the test set.
    """
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # fit the pipeline on the training set
    fitted_pipeline = model_object.fit(X_train, y_train)

    # get predictions for test set
    y_scores = fitted_pipeline.predict_proba(X_test)

    # convert test scores to crisp labels
    predicted_test_labels = probability_to_label(y_scores=y_scores, y_true=y_test.to_numpy(), threshold=threshold)

    # compute ppv and npv for each fold
    pos_pred_value = float(metrics.precision_score(y_test, predicted_test_labels, pos_label=1))
    neg_pred_value = float(metrics.precision_score(y_test, predicted_test_labels, pos_label=0))

    # compute mcc
    mcc_value = float(metrics.matthews_corrcoef(y_test, predicted_test_labels))

    return np.array(y_scores[:, 1]), np.array(y_test), pos_pred_value, neg_pred_value, mcc_value


def cross_validation_performance_95ci(
    model_object: Pipeline,
    random_state: int,
    X: pd.DataFrame,
    y: pd.Series[float],
    threshold: float,
) -> pd.DataFrame:
    """Compute model performance with confidence interval on cross validation.

    Parameters
    ----------
    model_object
        Pipeline with a minimum of baseline model.
    random_state
        Random state to be used.
    X
        ml_ready data.
    y
        Labels.
    threshold
        The optimal threshold calculated from the validation.

    Returns
    -------
        Dataframe of performance metrics and confidence intervals
    """
    logger.info("Computing 95% CI for several metrics using RSKF 4x25 folds on all_dataset.")

    # Number of folds and repeats
    kf = RepeatedStratifiedKFold(n_splits=4, n_repeats=25, random_state=random_state)

    with Parallel(n_jobs=-2) as parallel:
        # Parallelize the loop over folds
        final_results = cast(
            list[tuple[np_array, np_array, float, float, float]],
            parallel(
                delayed(train_and_evaluate_fold)(train_index, test_index, X, y, model_object, threshold)
                for train_index, test_index in kf.split(X, y)
            ),
        )

    # Unpack the results
    pos_probabilities, true_labels, pos_pred_value, neg_pred_value, mcc_value = (list(s) for s in zip(*final_results))

    # reuse compute_roc_curve function to compute
    # model performance on cross validation
    fpr_steps = np.linspace(0, 1, 101)
    results = compute_roc_curve(pos_probabilities, true_labels, fpr_steps)

    # Mean AUC and 95% CI
    mean_roc_auc = np.mean(np.array(results["roc_auc"]))
    ci_auc = np.quantile(results["roc_auc"], (0.025, 0.975))

    # Mean MCC and 95% CI
    mean_mcc = np.mean(np.array(mcc_value))
    ci_mcc = np.quantile(mcc_value, (0.025, 0.975))

    # Mean sensitivity and 95% CI for specificity=99%
    mean_sens_spec_99 = np.mean(results["sens_spec_99"])
    ci_sens_spec_99 = np.quantile(results["sens_spec_99"], (0.025, 0.975))

    # Mean sensitivity and 95% CI for specificity=90%
    mean_sens_spec_95 = np.mean(results["sens_spec_95"])
    ci_sens_spec_95 = np.quantile(results["sens_spec_95"], (0.025, 0.975))

    # Mean sensitivity and 95% CI for specificity=90%
    mean_sens_spec_90 = np.mean(results["sens_spec_90"])
    ci_sens_spec_90 = np.quantile(results["sens_spec_90"], (0.025, 0.975))

    # Mean PPV and 95% CI
    mean_pos_pred_value = np.mean(pos_pred_value)
    ci_pos_pred_value = np.quantile(pos_pred_value, (0.025, 0.975))

    # Mean NPV and 95% CI
    mean_neg_pred_value = np.mean(neg_pred_value)
    ci_neg_pred_value = np.quantile(neg_pred_value, (0.025, 0.975))

    return (
        pd.DataFrame(
            {
                "mean_auc": [float(mean_roc_auc), *ci_auc],
                "mean_mcc": [float(mean_mcc), *ci_mcc],
                "mean_sens_spec_99": [float(mean_sens_spec_99), *ci_sens_spec_99],
                "mean_sens_spec_95": [float(mean_sens_spec_95), *ci_sens_spec_95],
                "mean_sens_spec_90": [float(mean_sens_spec_90), *ci_sens_spec_90],
                "mean_pos_pred_value": [float(mean_pos_pred_value), *ci_pos_pred_value],
                "mean_neg_pred_value": [float(mean_neg_pred_value), *ci_neg_pred_value],
            },
            index=["Value", "Lower_bound", "Upper_bound"],
        )
        .T.loc[:, ["Lower_bound", "Value", "Upper_bound"]]
        .reset_index(names="")
    ).round(5)


def bootstrap_performance_95ci(
    model_object: Pipeline,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series[float],
    threshold: float,
    random_state: int,
    n_bootstraps: int = 1000,
) -> pd.DataFrame:
    """
    Compute model performance with confidence intervals using bootstrapping.

    Parameters
    ----------
    model_object
        Trained pipeline with the model.
    X_holdout
        Holdout feature matrix to evaluate.
    y_holdout
        True labels of the holdout set.
    threshold
        Fixed threshold used to convert probabilities to class labels.
    random_state
        Random seed for reproducibility.
    n_bootstraps, optional
        Number of bootstrap samples to generate, by default 1000.

    Returns
    -------
    DataFrame
        DataFrame of performance metrics and confidence intervals.
    """
    logger.info(f"Computing 95% CI for several metrics using {n_bootstraps} bootstraps on holdout set.")

    # Pre-generate all bootstrap indices with seeded RNG
    rng = np.random.default_rng(seed=random_state)
    n_samples = len(X_holdout)
    bootstrap_indices = [rng.choice(n_samples, size=n_samples, replace=True) for _ in range(n_bootstraps)]

    # Function that evaluates one bootstrap sample (takes pre-generated indices)
    def evaluate_bootstrap(indices: pd.Index) -> tuple[np_array, np_array, float, float, float]:
        X_sample = X_holdout.iloc[indices]
        y_sample = y_holdout.iloc[indices]

        y_proba = model_object.predict_proba(X_sample)
        y_pred = probability_to_label(y_proba, y_sample.to_numpy(), threshold)

        pos_pred_value = float(metrics.precision_score(y_sample, y_pred, pos_label=1))
        neg_pred_value = float(metrics.precision_score(y_sample, y_pred, pos_label=0))
        mcc_value = float(metrics.matthews_corrcoef(y_sample, y_pred))

        return np.array(y_proba[:, 1]), y_sample.to_numpy(), pos_pred_value, neg_pred_value, mcc_value

    # Parallel execution
    with Parallel(n_jobs=-2) as parallel:
        final_results = cast(
            list[tuple[np_array, np_array, float, float, float]],
            parallel(delayed(evaluate_bootstrap)(indices) for indices in bootstrap_indices),
        )

    # Unpack the results
    y_scores, true_labels, pos_pred_value, neg_pred_value, mcc_value = (list(s) for s in zip(*final_results))

    # Reuse compute_roc_curve function to compute model performance
    fpr_steps = np.linspace(0, 1, 101)
    results = compute_roc_curve(y_scores, true_labels, fpr_steps)

    # Mean AUC and 95% CI
    mean_roc_auc = np.mean(np.array(results["roc_auc"]))
    ci_auc = np.quantile(results["roc_auc"], (0.025, 0.975))

    # Mean MCC and 95% CI
    mean_mcc = np.mean(np.array(mcc_value))
    ci_mcc = np.quantile(mcc_value, (0.025, 0.975))

    # Mean sensitivity and 95% CI for specificity=99%
    mean_sens_spec_99 = np.mean(results["sens_spec_99"])
    ci_sens_spec_99 = np.quantile(results["sens_spec_99"], (0.025, 0.975))

    # Mean sensitivity and 95% CI for specificity=95%
    mean_sens_spec_95 = np.mean(results["sens_spec_95"])
    ci_sens_spec_95 = np.quantile(results["sens_spec_95"], (0.025, 0.975))

    # Mean sensitivity and 95% CI for specificity=90%
    mean_sens_spec_90 = np.mean(results["sens_spec_90"])
    ci_sens_spec_90 = np.quantile(results["sens_spec_90"], (0.025, 0.975))

    # Mean PPV and 95% CI
    mean_pos_pred_value = np.mean(pos_pred_value)
    ci_pos_pred_value = np.quantile(pos_pred_value, (0.025, 0.975))

    # Mean NPV and 95% CI
    mean_neg_pred_value = np.mean(neg_pred_value)
    ci_neg_pred_value = np.quantile(neg_pred_value, (0.025, 0.975))

    return (
        pd.DataFrame(
            {
                "mean_auc": [float(mean_roc_auc), *ci_auc],
                "mean_mcc": [float(mean_mcc), *ci_mcc],
                "mean_sens_spec_99": [float(mean_sens_spec_99), *ci_sens_spec_99],
                "mean_sens_spec_95": [float(mean_sens_spec_95), *ci_sens_spec_95],
                "mean_sens_spec_90": [float(mean_sens_spec_90), *ci_sens_spec_90],
                "mean_pos_pred_value": [float(mean_pos_pred_value), *ci_pos_pred_value],
                "mean_neg_pred_value": [float(mean_neg_pred_value), *ci_neg_pred_value],
            },
            index=["Value", "Lower_bound", "Upper_bound"],
        )
        .T.loc[:, ["Lower_bound", "Value", "Upper_bound"]]
        .reset_index(names="")
    ).round(5)
