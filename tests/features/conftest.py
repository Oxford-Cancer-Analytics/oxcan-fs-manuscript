import numpy as np
import pandas as pd
import pytest
import src.features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.features.preprocessing import Imputer
from src.features.preprocessing import Normalize


@pytest.fixture
def mock_optimise_model(mocker):
    def _op_model(self, model_object, parameters_grid, df, optimize):
        tuner = GridSearchCV(model_object, {"imputer_mnar": [Imputer(method=None, strategy="mar")]}, cv=2)
        tuner.fit(df.drop("target", axis=1).fillna(0), df["target"])

        class CVResult:
            def __init__(self, params, values):
                self.params = params
                self.values = values

        res = tuner.cv_results_
        return tuner.best_estimator_, [CVResult(res["params"], res["mean_test_score"])]

    mocker.patch.object(src.features.optimisation.ModelOptimisation, "_optimise_model", _op_model)


@pytest.fixture
def mock_get_optimised_pipeline(mocker):
    def _op_model(self, df, model_only):
        pipe = Pipeline(
            [
                ("imputer_mnar", Imputer(method=None, strategy="mar")),
                ("rf", RandomForestClassifier()),
            ]
        )
        tuner = GridSearchCV(pipe, {"imputer_mnar": [Imputer(method=None, strategy="mar")]}, cv=2)
        tuner.fit(df.drop("target", axis=1), df["target"])

        return pipe, tuner.best_estimator_, tuner

    mocker.patch.object(src.features.Features, "_get_optimised_pipeline", _op_model)


@pytest.fixture
def mock_important_features(mocker):
    def imp_feat(self, X):
        X = X.drop(columns=["target"])
        return pd.DataFrame({"importance_count": np.random.randint(0, 3, size=X.shape[1])}, index=X.columns)

    mocker.patch.object(src.features.selection.FeatureSelection, "get_important_features", imp_feat)


@pytest.fixture
def ttest_features(full_data):
    patient_ids = ["test", "test1", "test2", "test3", "test4"]
    full_data["patient_id"] = patient_ids
    full_data["features"] = list("ABCDE")
    full_data["n_features_to_select"] = list(range(1, len(full_data) + 1))
    full_data["features_chosen"] = [
        {"features_importances": {i: np.random.random() for i in list("ABCDE")}} for _ in range(1, len(full_data) + 1)
    ]
    full_data["mean_roc_auc"] = np.random.random(len(full_data))
    full_data["std_roc_auc"] = np.random.random(len(full_data)) / len(full_data)
    full_data["t_statistic"] = np.random.random(len(full_data)) * 30
    full_data["p_value"] = np.random.uniform(0.00001, 0.05, (len(full_data),))

    return full_data, full_data.iloc[2:], full_data.iloc[:2]


@pytest.fixture
def imputer(request):
    request = request.param if hasattr(request, "param") else ("qrilc", ["A", "B"])
    return Imputer(method=request[0], features=request[1], strategy="mar")


@pytest.fixture
def normalizer(request):
    request = request.param if hasattr(request, "param") else "log2"
    return Normalize(method=request)


@pytest.fixture
def mock_imputer(mocker, get_classwise_imputation_df):
    df = get_classwise_imputation_df
    df.columns = list(range(df.shape[1] - 1)) + ["target"]

    def _detect_mnar(*args, **kwargs):
        mnar_data, _ = df[[3]], df[[0, 1, 2, 4]]
        mnar_data.columns = ["3"]
        return mnar_data, _

    mocker.patch.object(src.features.preprocessing, "detect_mnar", _detect_mnar)
