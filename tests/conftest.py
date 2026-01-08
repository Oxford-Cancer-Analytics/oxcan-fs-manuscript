import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import src.data_loader
import src.statistical_analysis
import toml
from sklearn import preprocessing as prep
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.cli.cli_options import AugmentationEnum
from src.cli.cli_options import FeatureSelectionEnum
from src.cli.cli_options import ModelsEnum
from src.cli.cli_options import StatisticsEnum
from src.cli.toml_parser import TomlParser
from src.comparison import ModelComparison
from src.features.features import Features
from src.features.preprocessing import Imputer


@pytest.fixture
def parser(mocker):
    mocked_data = mocker.mock_open(read_data=toml.dumps({}))
    mocker.patch("src.cli.toml_parser.open", mocked_data)
    mocker.patch("src.data_loader.open", mocked_data)
    parser = TomlParser(Path("."))
    return parser


@pytest.fixture
def cli_options():
    return {
        "input_path": None,  # Use None to enable S3 mode
        "output_path": None,  # No longer a CLI parameter, set dynamically
        "stats": StatisticsEnum.TTEST_INDEPENDENT,
        "s3_input_path": Path("testing/test/"),
        "s3_output_path": Path("HRF/123456789"),
        "imputation_strategy": "mar",
        "s3_bucket": "test",
        "s3_labelled_cohort_key": Path("testing/key.csv"),
        "use_batch_corrected": False,
        "model": ModelsEnum.XGBOOST,
        "feature_selection": FeatureSelectionEnum.MULTISURF,
        "augmentation": AugmentationEnum.NONE,
        "dry_run": True,
        "rec_feat_add": False,
        "use_full_data": False,
        "prepare_data": False,
        "best_pipeline": Path(""),
    }


@pytest.fixture
def parser_with_cli_options(parser, cli_options):
    config = parser.read()
    config.add_cli_config(cli_options)

    return config


@pytest.fixture
def feature_class(parser_with_cli_options):
    parser_with_cli_options.data.optimization.optuna.trials = 1
    yield Features(parser_with_cli_options)
    parser_with_cli_options.data.optimization.optuna.trials = 200


@pytest.fixture
def get_X_y_data():
    X = np.array(
        [
            [1, 2, 3, np.nan, 5],
            [5, 6, 7, np.nan, 9],
            [1, 2, 1, np.nan, 1],
            [9, 10, 11, 2, 100],
            [1, 1, 1, np.nan, 1],
        ]
    )
    y = np.array([0, 1, 0, 1, 0])
    protein_list = ["A", "B", "C", "D", "E"]
    return X, y, protein_list


@pytest.fixture
def get_X_y_data_imputed():
    print("Setting up get_X_y_data_imputed")
    X = np.array(
        [
            [1, 2, 3, 4, 5],
            [5, 6, 7, 8, 9],
            [1, 0, 1, 0, 1],
            [9, 10, 11, 0, 100],
            [1, 1, 1, 0, 1],
        ]
    )
    y = np.array([0, 1, 0, 1, 0])
    y_scores = np.array([[0.1, 0.9], [0.15, 0.85], [0.53, 0.47], [0.02, 0.98], [0.13, 0.87]])
    protein_list = ["A", "B", "C", "D", "E"]
    return X, y, y_scores, protein_list


@pytest.fixture
def get_binomial_X_y_data():
    X = np.random.random(size=(100, 5))
    y = np.random.binomial(1, 0.5, size=(100,))
    protein_list = ["A", "B", "C", "D", "E"]
    return X, y, protein_list


@pytest.fixture
def get_optimisation_df(get_X_y_data):
    X, y, protein_list = get_X_y_data
    optimisation_df = pd.DataFrame(X, columns=protein_list)
    optimisation_df.drop(columns=["D"], inplace=True)
    optimisation_df["target"] = y
    return optimisation_df


@pytest.fixture
def get_classwise_imputation_df(get_X_y_data):
    X, y, protein_list = get_X_y_data
    classwise_imputation_df = pd.DataFrame(X, columns=protein_list)
    classwise_imputation_df["target"] = y
    return classwise_imputation_df


@pytest.fixture
def get_rf_base_model():
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", prep.StandardScaler()),
            ("lr", RandomForestClassifier()),
        ]
    )


def get_imputed_data():
    df = pd.DataFrame(
        np.array(
            [
                [1, 2, 3, 4, 5, 6, 7],
                [5, 6, 7, 8, 9, 10, 11],
                [1, 0, 1, 0, 1, 1, 0],
                [9, 10, 11, 0, 100, 12, 67],
                [19, 1, 65, 40, 10, 34, 89],
            ]
        ),
        columns=["A", "B", "C", "D", "E", "F", "G"],
    )
    df["target"] = np.array([0, 1, 0, 1, 1])
    df["label"] = df["target"].apply(lambda x: "control" if x == 0 else "NSLC early")
    return df


class MockS3:
    def get_object(self, Bucket, Key, *args, **kwargs):
        buffer = io.StringIO()
        df = get_imputed_data()
        df["Unnamed: 0"] = range(df.shape[0])

        if Key.endswith(".txt"):
            df.to_csv(buffer, index=False, sep="\t")
        elif Key.endswith(".csv"):
            df.to_csv(buffer, index=False)
        else:
            buffer = io.BytesIO()
            df.to_parquet(buffer, index=False)

        buffer.seek(0)
        return {"Body": buffer}

    def put_object(self, Bucket, Key, Body, *args, **kwargs):
        return {"Body": Body, "ResponseMetadata": {"HTTPHeaders": {"date": "test"}}, "VersionId": "test"}

    def list_objects(self, Bucket, Prefix, *args, **kwargs):
        return {"Contents": [{"Key": "test.csv"}]}


class MockBoto3:
    def client(self, name="s3"):
        return MockS3()


@pytest.fixture(autouse=True)
def mock_s3(mocker):
    mocker.patch.object(src.data_loader, "boto3", MockBoto3)


@pytest.fixture
def mock_check_data(mocker):
    def check_data(data):
        df = get_imputed_data()
        train, test = df.iloc[2:], df.iloc[:2]
        return df, train, test

    mocker.patch.object(src.data_loader, "check_data", check_data)


@pytest.fixture
def full_data():
    return get_imputed_data()


@pytest.fixture
def model_comparison():
    comparison = ModelComparison("test", ["testing.csv", "123.csv"])

    # Create mock dataframes since the files don't exist
    df1 = pd.DataFrame(
        {
            "feature_A": [1, 2, 3, 4, 5],
            "feature_B": [2, 3, 4, 5, 6],
            "feature_C": [3, 4, 5, 6, 7],
        }
    )
    df2 = pd.DataFrame(
        {
            "feature_A": [1.1, 2.1, 3.1, 4.1, 5.1],
            "feature_B": [2.1, 3.1, 4.1, 5.1, 6.1],
            "feature_C": [3.1, 4.1, 5.1, 6.1, 7.1],
        }
    )

    # Manually set the dataframes and model names since _extract_data fails with non-existent files
    comparison.dataframes = [df1, df2]
    comparison.model_names = ["testing", "123"]

    for df in comparison.dataframes:
        df["p_value"] = np.random.uniform(0.00001, 0.05, (len(df),))
        df["n_features_to_select"] = list(range(1, len(df) + 1))
        df["features_chosen"] = [
            {"features_importances": {i: np.random.random() for i in list("ABCDE")}} for _ in range(1, len(df) + 1)
        ]

    return comparison


@pytest.fixture
def ttest_posthoc_mock(mocker, get_X_y_data_imputed):
    def multipletests_mock(*args, **kwargs):
        return [True, False, True, True], "test"

    mocker.patch.object(src.statistical_analysis, "multipletests", multipletests_mock)

    *_, proteins = get_X_y_data_imputed
    df = pd.DataFrame(np.random.randint(1, 100, size=(50, 5)), columns=proteins)
    df["target"] = np.random.randint(0, 1, size=50)
    df["test"] = list(map(chr, np.random.randint(65, 73, size=50).tolist()))
    combinations = [("A", "B"), ("C", "D"), ("E", "F"), ("G", "H")]

    return df, combinations


@pytest.fixture
def pipeline(request):
    clf = request.param if hasattr(request, "param") else RandomForestClassifier()
    return Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=False, with_std=False)),
            ("imputer_mnar", Imputer(method=None, strategy="mar")),
            ("imputer_mar", Imputer(method=None, strategy="mar")),
            ("rf", clf),
        ]
    )


@pytest.fixture
def mock_joblib_parallel(mocker, pipeline):
    class Parallel:
        def __init__(self, *args, **kwargs): ...

        def __call__(self, *args, **kwargs):
            arg = list(args[0])
            return [arg[0][0](*arg[0][1])]

    mocker.patch.object(src.data_preparation, "Parallel", Parallel)
