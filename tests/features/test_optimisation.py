import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from src.cli.toml_parser import ModelsEnum
from src.features.preprocessing import Imputer
from src.features.preprocessing import Normalize


class TestModelOptimisation:
    @pytest.mark.skip(reason="too slow")
    @pytest.mark.parametrize(
        ("model", "tag"),
        (
            (ModelsEnum.RANDOM_FOREST, ModelsEnum.RANDOM_FOREST.value),
            (ModelsEnum.XGBOOST, ModelsEnum.XGBOOST.value),
        ),
    )
    def test_get_best_pipeline(self, feature_class, get_X_y_data, model, tag, mock_optimise_model):
        X, y, proteins = get_X_y_data
        df = pd.DataFrame(X, columns=proteins)
        df["target"] = y

        output = feature_class.optimisation.get_best_pipeline(df, model_type=model, optimize=True)
        est, grid = output

        assert isinstance(grid, pd.DataFrame)
        assert all(col in grid.columns for col in ["params", "mean_test_sens@99spec"])
        assert isinstance(est, Pipeline)
        assert est.steps[-1][0] == tag

    def test_get_best_pipeline_raises(self, feature_class, get_X_y_data):
        X, y, proteins = get_X_y_data
        df = pd.DataFrame(X, columns=proteins)
        df["target"] = y

        with pytest.raises(NotImplementedError):
            feature_class.optimisation.get_best_pipeline(df, model_type=ModelsEnum._TEST_CASE_IGNORE)

    def test_optimise_model(self, feature_class, get_X_y_data, pipeline, mock_optimise_model):
        X, y, proteins = get_X_y_data
        df = pd.DataFrame(X, columns=proteins)
        df["target"] = y

        # Required to increase size to > 10 samples as test_size for single=False, needs to be greater than 1
        df = pd.concat([df, df, df]).reset_index(drop=True)

        output = feature_class.optimisation._optimise_model(
            pipeline,
            {
                "scaler": [StandardScaler(with_mean=False, with_std=False)],
                "imputer_mnar": [Imputer(method=None, strategy="mar")],
                "imputer_mar": [Imputer(method=None, strategy="mar")],
            },
            df=df,
            optimize=True,
        )
        est, grid = output

        assert len(output) == 2
        assert isinstance(est, Pipeline)

    def test_init_model(self, get_X_y_data, feature_class):
        X, y, proteins = get_X_y_data
        df = pd.DataFrame(X, columns=proteins)
        df["target"] = y
        feature_class.optimisation.df = df
        output = feature_class.optimisation.init_model(ModelsEnum.XGBOOST)

        pipeline, params = output

        assert len(output) == 2
        assert isinstance(pipeline, Pipeline)
        assert isinstance(params, dict)
        assert all(key in params for key in ["scaler", "imputer_mnar", "imputer_mar"])
        assert len(pipeline.steps) == 4
