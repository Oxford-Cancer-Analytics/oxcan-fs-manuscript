import io
from typing import cast
from typing import TypedDict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from optuna import Study
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_param_importances
from sklearn import metrics

from ._types import np_array
from .oxcan_colors import OXcanColors
from .statistical_analysis import ttest_posthoc


font = {  # TODO fonts to decide? - Will depend on journals
    # 'family': 'serif',
    # 'color':  'red',
    # 'weight': 'bold',
    "size": 10
}

matplotlib.use("Agg")


class ROCMetrics(TypedDict):
    """Metrics for computing the ROC Curve."""

    mean_tprs: np_array
    mean_auc: float
    interp_tprs: list[np_array]
    interp_thresh: list[np_array]
    sens_spec_99: list[float]
    sens_spec_95: list[float]
    sens_spec_90: list[float]
    sens_spec_85: list[float]
    sens_spec_80: list[float]
    roc_auc: list[float]


def compute_roc_curve(predictions: list[np_array], test_labels: list[np_array], fpr_steps: np_array) -> ROCMetrics:
    """Computes various ROCMetrics based on the ROC curve.

    Parameters
    ----------
    predictions
        Model probability predictions across folds or for overall validation.
    test_labels
        True labels on validation set or test sets across folds.
    fpr_steps
        FPR steps on x-Axis.

    Returns
    -------
        ROC performance metrics
    """
    # initialise lists of arrays for performance metrics
    roc_auc, fpr, tpr, thresholds, sens_spec_99, sens_spec_95, sens_spec_90, sens_spec_85, sens_spec_80 = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    # performance metrics for each fold if cross validation
    for y_pred, y_true in zip(predictions, test_labels):
        # info for roc plot
        this_fpr, this_tpr, this_thresh = metrics.roc_curve(y_true, y_pred)
        fpr.append(this_fpr)
        tpr.append(this_tpr)
        thresholds.append(this_thresh)

    # interpolate tprs across a given range of
    # threshold values for each fold or class
    interp_tprs = []
    interp_thresh = []
    for i in range(len(fpr)):
        interp_tpr = np.interp(fpr_steps, fpr[i], tpr[i])
        interp_tprs.append(interp_tpr)
        interp_thresh.append(np.interp(fpr_steps, fpr[i], thresholds[i]))
        sens_spec_99.append(interp_tpr[1])
        sens_spec_95.append(interp_tpr[5])
        sens_spec_90.append(interp_tpr[10])
        sens_spec_85.append(interp_tpr[15])
        sens_spec_80.append(interp_tpr[20])
        roc_auc.append(metrics.auc(fpr_steps, interp_tpr))

    # compute mean of tprs across folds or classes
    # at each threshold value
    mean_tprs = np.array(interp_tprs).mean(axis=0)
    mean_auc = float(metrics.auc(fpr_steps, mean_tprs))

    return ROCMetrics(
        {
            "mean_tprs": mean_tprs,
            "mean_auc": mean_auc,
            "interp_tprs": interp_tprs,
            "interp_thresh": interp_thresh,
            "sens_spec_99": sens_spec_99,
            "sens_spec_95": sens_spec_95,
            "sens_spec_90": sens_spec_90,
            "sens_spec_85": sens_spec_85,
            "sens_spec_80": sens_spec_80,
            "roc_auc": roc_auc,
        }
    )


def generate_annotated_protein_plot(
    title: str,
    x1: np_array,
    x2: np_array,
    cluster_label: np_array,
    protein_label: list[str | int],
    proteins_to_annotate: list[str],
) -> io.BytesIO:
    """Generate 2D figure for PCA etc.

    Parameters
    ----------
    title
        Fig title
    x1
        pc1 or other dim
    x2
        pc2 or other dim
    cluster_label
        cluster labels i.e. groupings of proteins
    protein_label
        labels of proteins that match x1/x2 shape
    proteins_to_annotate
        subset of proteins chosen to annotate

    Returns
    -------
        A figure as a BytesIO object.
    """
    for cl in set(cluster_label):
        plt.scatter(x1[cluster_label == cl], x2[cluster_label == cl], label=f"cluster_{cl}")

    for x1i, x2i, prot in zip(x1, x2, protein_label):
        if prot in proteins_to_annotate:
            plt.text(x1i, x2i, prot, fontdict=font)

    plt.title(title)
    plt.legend()

    img_data = io.BytesIO()
    plt.savefig(img_data, format="png", bbox_inches="tight")
    plt.close()

    return img_data


def plot_feature_vs_x(
    feat_list: pd.DataFrame,
    x_axis: str = "features",
    y_axis: str = "mean_roc_auc",
    y_label: str = "Importance",
) -> io.BytesIO:
    """Plots features on x-axis versus importance/other value.

    Parameters
    ----------
    feat_list
        DataFrame of [feature names, X], X=importance for example
    x_axis, optional
        column name to use for x_axis, by default "features"
    y_axis, optional
        column name to use for y_axis, by default "mean_roc_auc"
    y_label, optional
        y label, by default "Importance"

    Returns
    -------
        A figure as a BytesIO object.
    """
    colours = OXcanColors()
    col = colours.get_colors(2)

    # Create figure
    fig = plt.figure(figsize=list(np.array([20, 4]) / 2.54))
    ax = fig.add_axes([0.2, 0.2, 0.75, 0.75])

    feat_list[y_axis].plot(ax=ax, x_compat=True, color=col[0])
    plt.xticks(range(len(feat_list)), feat_list[x_axis], rotation=60)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Axes label
    plt.xlabel("Features", fontsize=11)
    plt.ylabel(y_label, fontsize=11)
    plt.legend(loc="upper right", frameon=False)

    fig = io.BytesIO()
    plt.savefig(fig, format="png", bbox_inches="tight")
    plt.close()

    return fig


def plot_feature_importance(
    feature_names: list[str | int],
    importances: np_array,
    num_features: int = 20,
) -> io.BytesIO:
    """Plot feature importance.

    Parameters
    ----------
    feature_names
        List of feature names
    importances
        Feature importances
    num_features, optional
        number of features to plot, by default 20

    Returns
    -------
        A figure as a BytesIO object.
    """
    # Create figure
    fig = plt.figure(figsize=list(np.array([15, 9]) / 2.54))
    ax = fig.add_axes([0.53, 0.19, 0.42, 0.75])

    indices = np.argsort(importances)

    keep_indices = indices[-num_features:]

    col = [
        OXcanColors().get_2color_shade_of_value(value=x, max_col="pink") for x in np.linspace(1, 0, len(keep_indices))
    ]

    y_ticks = np.arange(0, len(keep_indices))
    ax.barh(y_ticks, importances[keep_indices], color=col)
    plt.yticks(y_ticks, [feature_names[i] for i in keep_indices])
    plt.xlabel("Normalised Feature Importance")

    fig = io.BytesIO()
    plt.savefig(fig, format="png", bbox_inches="tight")
    plt.close()

    return fig


def plot_shapley_additive_explanations(
    shap_values: pd.DataFrame,
    label: str,
    y_explain: list[float] = [np.nan],
    num_features: int = 20,
    num_samples: int = 5,
) -> list[io.BytesIO]:
    """Create plot of Shapley additive explanations for individual samples.

    Parameters
    ----------
    shap_values
        Dataframe of Shapley values with shape
    label
        Label outcomes to be displayed in plots
    y_explain
        Probability outcomes to be displayed in plots
    num_features, optional
        Number of features to show, by default 20
    num_samples, optional
        Number of sample' plots to generate, by default 5

    Returns
    -------
        list of matplotlib.pyplot.figure: Figure objects
    """
    col = [OXcanColors().get_2color_shade_of_value(value=x, max_col="pink") for x in np.linspace(1, 0, num_features)]

    # Get the mean shapley values and standard deviation
    # over cross-validation runs
    shap_values_mean = shap_values.groupby(shap_values.index).mean()

    # Limit number of features and elements
    num_features = min(num_features, shap_values_mean.shape[-1])
    num_samples = shap_values_mean.shape[0] if (num_samples > shap_values.shape[0]) else num_samples
    shap_values_mean = shap_values_mean.iloc[:num_samples]

    # Create plot for each sample in Shapley values data frame
    figs = []

    for (element_index, element_shap_values), y_prob in zip(shap_values_mean.iterrows(), y_explain):
        # Restrict to top 'num_features' in terms of absolute contribution
        element_shap_values = element_shap_values.reindex(element_shap_values.abs().sort_values().index)[
            -num_features:
        ]
        element_shap_values = element_shap_values.sort_values()

        # Create figure
        fig = plt.figure(figsize=list(np.array([16, 10]) / 2.54))
        ax = fig.add_axes([0.45, 0.16, 0.52, 0.75])

        # Plot top feature contributions for every element
        element_shap_values.plot.barh(color=col)  # type: ignore

        # Axis settings
        x_max = element_shap_values.abs().max()
        ax.set_xlim([-x_max * 1.05, +x_max * 1.05])  # type: ignore

        # Axis labels
        ax.set_xlabel("Shapley value", fontsize=10, font="monospace")
        ax.set_title(f"Top contributing features - Sample {element_index}", fontsize=12, font="monospace")

        y_dividing_line = np.argmax(element_shap_values >= 0)
        if y_prob is not np.nan:
            plt.text(
                x=-x_max,
                y=y_dividing_line,  # type: ignore
                s=f"Risk = {int(np.round(y_prob*100))}%",
            )
        plt.text(
            x=x_max - 0.5,
            y=y_dividing_line,  # type: ignore
            s=label,
        )
        plt.hlines(y_dividing_line - 0.5, xmin=-x_max, xmax=x_max, lw=2)
        plt.vlines(0, ymin=-0.5, ymax=element_shap_values.shape[0], lw=1)

        fig = io.BytesIO()
        # Append figure to output
        figs.append(fig)
        plt.savefig(fig, format="png", bbox_inches="tight")
        plt.close()

    return figs


def plot_roc_curve(
    predictions: list[np_array],
    test_labels: list[np_array],
    multiple: bool = False,
    labels: list[str] | None = None,
    average: str | None = None,
    ax: plt.Axes | None = None,
) -> tuple[io.BytesIO, ROCMetrics]:
    """Plots roc curve for a set of predictions from a cross validation.

    Parameters
    ----------
    predictions
        model predictions on cross validation or validation
    test_labels
        true labels on validation set or test sets across folds
    multiple
        plot multiple curves, by default False
    labels, optional.
        legends if multiple is True
    average, optional
        average strategy if multiple is True
    ax, optional
        An Axes object, by default None

    Returns
    -------
        ROC AUC figure and associated data.
    """
    colours = OXcanColors()
    col = colours.get_colors(10)

    # Create figure
    fig = plt.figure(figsize=list(np.array([16, 10]) / 2.54))
    if ax is None:
        ax = fig.add_axes([0.2, 0.2, 0.75, 0.75])

    assert isinstance(ax, plt.Axes)
    ax.plot([0, 1], [0, 1], color="grey", linestyle="dashed", lw=2)

    fpr_steps = np.linspace(0, 1, 101)
    fig_data = compute_roc_curve(predictions, test_labels, fpr_steps)
    fpr_steps = np.concatenate([[0.0], fpr_steps])
    fig_data["mean_tprs"] = np.concatenate([[0.0], fig_data["mean_tprs"]])
    fig_data["interp_thresh"] = [np.concatenate([interp_thresh, [0.0]]) for interp_thresh in fig_data["interp_thresh"]]

    # Plot sub results
    if multiple:
        interp_tprs = fig_data["interp_tprs"]
        roc_auc = fig_data["roc_auc"]
        labels = cast(list[str], labels)
        for i in range(len(labels)):
            interp_tpr = interp_tprs[i]
            plt.plot(
                fpr_steps,
                interp_tpr,
                color=col[i + 1],
                lw=2,
                label=f"{labels[i]} (AUC = {roc_auc[i]:.2f})",
            )
    # plot overall results
    ax.plot(
        fpr_steps,
        fig_data["mean_tprs"],
        color=col[0],
        lw=2,
        label=f"{average} AUC = {fig_data['mean_auc']:.5f}" if average else f"AUC = {fig_data['mean_auc']:.5f}",
    )

    # Axes settings
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])  # type: ignore
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])  # type: ignore
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Axes labels
    ax.set_xlabel("1 - Specificity", fontsize=10, font="monospace")
    ax.set_ylabel("Sensitivity", fontsize=10, font="monospace")
    ax.legend(loc="lower right", frameon=False, prop="monospace")

    fig = io.BytesIO()
    plt.savefig(fig, format="jpeg", dpi=1000, bbox_inches="tight")
    plt.close()

    return fig, fig_data


def violin_plot(data: pd.DataFrame, Y: str, group: str, combinations: list[tuple[str, str]]) -> io.BytesIO:
    """Violin plot, x-axis discrete buckets.

    Performs a two-tailed t-test post hoc and plots it into a violin.

    Parameters
    ----------
    data
        Dataframe with a continuous variable
    Y
        Column to test for variability
    group
        Column with groups
    combinations
        Combinations of groups to be tested for variability

    Returns
    -------
        Figure object
    """
    colours = OXcanColors()
    col = colours.get_colors(15)

    fig = plt.figure(figsize=list(np.array([20, 4]) / 2.54))
    ax = fig.add_axes([0.2, 0.2, 0.75, 0.75])

    # run the post-hoc t-test
    stars = ttest_posthoc(data, group, Y, combinations)

    # get the list of intervals
    x_list = data[group].unique().tolist()

    data.sort_values(by=[group], ascending=True, inplace=True)
    max_val = data[Y].max(axis=0)

    # plot the figure
    ax = cast(plt.Axes, sns.violinplot(x=group, y=Y, data=data, ax=ax, palette=col))

    ax.set_xlabel("Plate and condition", fontsize=10, font="monospace")
    ax.set_ylabel(Y, fontsize=10, font="monospace")
    ax.tick_params(axis="x", labelsize=8, rotation=35)
    ax.tick_params(axis="y", labelsize=8)

    # add *
    for i in stars.keys():
        ax.text(x_list.index(i), max_val + 1, "*", size=12)

    ax.set_facecolor("white")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    fig = io.BytesIO()
    plt.savefig(fig, format="jpeg", dpi=1000, bbox_inches="tight")
    plt.close()

    return fig


def plot_optuna_figures(study: Study, scorers: list[str], figures: dict[str, io.BytesIO]) -> dict[str, io.BytesIO]:
    """Plots figures from the Optuna hyperparameter optimization.

    The optimization history and parameter importances are plotted.

    Parameters
    ----------
    study
        The study object of the optuna optimization.
    scorers
        A list of scorers used for optuna optimization.
    figures
        A dictionary of existing figures to update.

    Returns
    -------
        An updated dictionary which includes figures from optuna optimization.
    """
    if len(scorers) > 1:
        for idx, scorer in enumerate(scorers):
            # optimization history
            figure = plot_optimization_history(study, target=lambda x: x.values[idx], target_name=scorer)
            figures |= {f"optimization_history_{scorer}": io.BytesIO(figure.to_image(format="png"))}

            # param importances
            if len(study.trials) > 1:
                figure = plot_param_importances(study, target=lambda x: x.values[idx], target_name=scorer)
                figures |= {f"param_importances_{scorer}": io.BytesIO(figure.to_image(format="png"))}
    else:
        (scorer,) = scorers

        # optimization history
        figure = plot_optimization_history(study)
        figures |= {f"optimization_history_{scorer}": io.BytesIO(figure.to_image(format="png"))}

        # param importances
        if len(study.trials) > 1:
            figure = plot_param_importances(study)
            figures |= {f"param_importances_{scorer}": io.BytesIO(figure.to_image(format="png"))}

    return figures
