import numpy as np
import pandas as pd
import pytest
from src.statistical_analysis import Analysis
from src.statistical_analysis import apply_t_test_minimal_features
from src.statistical_analysis import get_significant_features
from src.statistical_analysis import ttest_posthoc


def test_get_significant_features(get_X_y_data_imputed):
    X, y, _, protein_list = get_X_y_data_imputed
    filtered_ttest_results = get_significant_features(X, y, protein_list)

    final_list = filtered_ttest_results[filtered_ttest_results.pvalues > 0].features.to_list()
    assert "A" in final_list, "Pvalue filter has diverged"


def test_apply_t_test_minimal_features(get_X_y_data):
    X, y, proteins = get_X_y_data
    df = pd.DataFrame(X, columns=proteins)
    df["target"] = y
    df["mean_metric"] = np.random.random(X.shape[0])
    df["std_metric"] = np.random.random(X.shape[0]) / X.shape[0]

    output = apply_t_test_minimal_features(df, observations=5, metric="metric")

    assert all(col in output.columns for col in ["t_statistic_metric", "p_value_metric"])
    assert 1.0 in output.p_value_metric


class TestAnalysis:
    def test_get_sample_size_estimate_default(self, parser_with_cli_options):
        analysis = Analysis(parser_with_cli_options)
        output = analysis.get_sample_size_estimate()

        assert output == 105

    @pytest.mark.parametrize(
        ("mean1", "mean2", "var", "output"),
        (
            (0.1, 0.6, 0.2, 5),
            (0.2, 0.8, 0.1, 2),
        ),
    )
    def test_get_sample_size_estimate(self, parser_with_cli_options, mean1, mean2, var, output):
        analysis = Analysis(parser_with_cli_options)
        result = analysis.get_sample_size_estimate(mean1, mean2, var)

        assert result == output
        assert isinstance(result, int)


def test_ttest_posthoc(ttest_posthoc_mock):
    df, combinations = ttest_posthoc_mock
    signif_groups = ttest_posthoc(df=df, group="test", feat="B", combinations=combinations)

    assert isinstance(signif_groups, dict)
    assert all(
        key in [combo for combination in combinations for combo in combination] for key in list(signif_groups.keys())
    )
    assert all(val in [0, 1] for val in signif_groups.values())
