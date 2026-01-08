import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from scipy.stats import ttest_ind_from_stats
from statsmodels.sandbox.stats.multicomp import multipletests
from statsmodels.stats.power import TTestIndPower

from ._types import np_array
from .cli.toml_parser import TomlParser


class Analysis:
    """Statistical analysis.

    Parameters
    ----------
    config
        The full config options.
    """

    def __init__(self, config: TomlParser) -> None:
        self.config = config

    def get_sample_size_estimate(
        self,
        estimated_mean_1: float = 0.95,
        estimated_mean_2: float = 0.9,
        estimated_variance: float = 0.1,
    ) -> int:
        """Sample size estimate.

        Parameters
        ----------
        estimated_mean_1, optional
            mean, by default 0.95.
        estimated_mean_2, optional
            mean, by default 0.9.
        estimated_variance, optional
            variance, by default 0.1.

        Returns
        -------
            N samples required to reject null hypothesis.
        """
        effect_size = (estimated_mean_1 - estimated_mean_2) / estimated_variance
        sample_size_calculator = TTestIndPower()
        n_obs = sample_size_calculator.solve_power(
            effect_size=effect_size,
            alpha=self.config.data.stats.alpha,  # 0.05 is commmon
            power=self.config.data.stats.power,  # 0.95 is common
        )
        return int(np.round(n_obs, 0))


def get_significant_features(
    X: np_array,
    y: np_array,
    protein_list: list[str | int],
) -> pd.DataFrame:
    """Gets the significant set of features.

    Parameters
    ----------
    X
        The data to test, not including the classes.
    y
        The class data.
    protein_list
        The list of proteins.

    Returns
    -------
        A dataframe of statistically significant features with
        their p-values.
    """
    _, pvalue = ttest_ind(
        X[y == 1], X[y == 0], alternative="two-sided", nan_policy="omit"  # this is 2 sided can be larger or smaller
    )  # outputs are tstat, pvalue

    return pd.DataFrame({"features": protein_list, "pvalues": pvalue})


def apply_t_test_minimal_features(result: pd.DataFrame, metric: str, observations: int) -> pd.DataFrame:
    """Perform ttest to find smallest number of features.

    These should be not significant from best performing set.

    Parameters
    ----------
    result
        Result of features to select, mean and std of AUC ROC.
    metric
        String selecting the name of the metric you want to consider from results df.
    observations
        The number of observations.

    Returns
    -------
        Appended t and p value to results.
    """
    result = result.copy()
    # highest roc

    # names for mean and std
    mean_metric = "mean_" + metric
    std_metric = "std_" + metric

    start_index = result[mean_metric].argmax()
    highest_metric = result.iloc[start_index][mean_metric]
    highest_metric_std = result.iloc[start_index][std_metric]
    # n = rfe_multisurf_cross_validation splits
    nobs1 = observations
    nobs2 = observations

    # get t and p value from ttest directly from stats
    stat_store = [
        ttest_ind_from_stats(
            mean1=highest_metric,
            std1=highest_metric_std,
            nobs1=nobs1,
            mean2=mean2,
            std2=std2,
            nobs2=nobs2,
        )
        for mean2, std2 in zip(result[mean_metric], result[std_metric])
    ]

    result["t_statistic_" + metric] = [ss[0] for ss in stat_store]
    result["p_value_" + metric] = [ss[1] for ss in stat_store]

    return result


def ttest_posthoc(df: pd.DataFrame, group: str, feat: str, combinations: list[tuple[str, str]]) -> dict[str, int]:
    """Performs two-tailed t-test post hoc.

    Performs Bonferroni test and prints the corrected p-values
    after False Dicovery Rate correction.

    Parameters
    ----------
    df
        Dataframe with a continuous variable.
    group
        Column with groups.
    feat
        Column tested for variability.
    combinations
        Combinations of groups to be tested for variability.

    Returns
    -------
        Mapping of groups with their significance. 1 if it shows
        variability with at least one other group, 0 otherwise
    """
    p_vals = []
    # perform a t-test post hoc
    for comb in combinations:
        data1 = df[df[group] == comb[0]]
        data2 = df[df[group] == comb[1]]
        _, p = ttest_ind(data1[feat], data2[feat])
        p_vals.append(p)

    # checking significance and return the p-values
    # correction for multiple testing using FDA
    signif_groups = {}

    reject_list, _ = multipletests(p_vals, method="fdr_bh")[:2]
    for reject, c in zip(reject_list, combinations):
        if reject:
            signif_groups[c[0]] = 1
            signif_groups[c[1]] = 1

    return signif_groups
