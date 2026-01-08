import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
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


class TestDataPreparation:
    @pytest.mark.parametrize(("name"), ("scaler", "imputer_mnar", "imputer_mar"))
    def test_get_param_combinations(self, get_X_y_data, feature_class, name):
        X, y, proteins = get_X_y_data
        df = pd.DataFrame(X, columns=proteins)
        df["target"] = y
        feature_class.optimisation.df = df
        _, params = feature_class.optimisation.init_model(ModelsEnum.XGBOOST)

        output = feature_class.data_preparation._get_param_combinations(params, name)

        assert isinstance(output, list)
        if name == "scaler":
            assert any(isinstance(inst, (StandardScaler, MinMaxScaler, Normalize)) for inst in output)
        else:
            assert all(isinstance(inst, Imputer) for inst in output)
