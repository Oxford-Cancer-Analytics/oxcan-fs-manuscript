from __future__ import annotations

import logging
from functools import partial
from typing import Any
from typing import cast
from typing import TYPE_CHECKING
from typing import TypedDict

import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.ensemble import BalancedRandomForestClassifier
from optuna import create_study
from optuna import Study
from optuna import Trial
from optuna.samplers import NSGAIISampler
from optuna.samplers import TPESampler
from optuna.trial import FrozenTrial
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import get_scorer
from sklearn.metrics import get_scorer_names
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .._types import np_array
from ..cli.cli_options import ModelsEnum
from .performance_metrics import compute_roc_curve
from .preprocessing import Imputer
from .preprocessing import Normalize

if TYPE_CHECKING:
    from ..cli.toml_parser import TomlParser


logger = logging.getLogger(__name__)


class SklearnModels(TypedDict, total=False):
    """Typing implementation for unpacking parameters."""

    random_state: int


class ModelOptimisation:
    """Feature optimisation methods.

    Parameters
    ----------
    config
        The full config options.

    Attributes
    ----------
    random_state
        The random state from the config.
    fixed_params
        The set of hyperparameters from running the optuna optimizer.
    study
        The study object. Set once the optuna optimizer has ran.
    df
        The data used for optimizing.
    scorers
        A list of scorers used to optimize the optuna optimizer.
    """

    def __init__(self, config: TomlParser) -> None:
        self.config = config

        self.random_state = self.config.data.random_state
        self.fixed_params: dict[str, Any] = {}
        self.study: Study
        self.df: pd.DataFrame

        scorer = self.config.data.optimization.optuna.scorer
        self.scorers: list[str] = scorer if isinstance(scorer, list) else [scorer]

    def get_best_pipeline(
        self,
        df: pd.DataFrame,
        model_type: ModelsEnum = ModelsEnum.XGBOOST,
        model_only: bool = False,
        optimize: bool = False,
    ) -> tuple[Pipeline, pd.DataFrame]:
        """Gets pipeline parameters and passes parameters to grid search.

        Parameters
        ----------
        df
            The full dataset.
        model_type, optional
            The model to be optimised, by default ModelsEnum.XGBOOST.
        model_only, optional
            To only optimise the model and not the preprocessing steps,
            by default False.
        optimize, optional
            To optimize the parameters of the model, by default False.

        Returns
        -------
        A pipeline and results from the Optuna model.
        """
        self.df = df
        model_object, parameters_grid = self.init_model(model_type, model_only=model_only, optimize=optimize)
        logger.info(f"Running {model_type} model.")

        if model_only:
            model_object = Pipeline([model_object.steps[-1]])

        # optimise the model - run grid search
        best_estimator, grid_results = self._optimise_model(
            model_object=model_object, parameters_grid=parameters_grid, df=df, optimize=optimize
        )

        grid_cv_results = pd.DataFrame.from_records(
            [(gr.params, *gr.values) for gr in grid_results],  # type: ignore
            columns=["params", *[f"mean_test_{scorer}" for scorer in self.scorers]],
        )

        return best_estimator, grid_cv_results

    def init_model(
        self, model: ModelsEnum, model_only: bool = False, optimize: bool = False
    ) -> tuple[Pipeline, dict[str, list[Any]]]:
        """Initializes the model pipeline and associated parameters.

        Parameters
        ----------
        model
            The model to initialize.
        model_only, optional
            To only optimise the model and not the preprocessing steps,
            by default False.
        optimize, optional
            To optimize the parameters of the model, by default False.

        Returns
        -------
            A pipeline with an initialized model and associated parameters.

        Raises
        ------
        NotImplementedError
            If the model type is not recognized as part of the ModelsEnum.
        """
        strategy = self.config.cli_data.imputation_strategy

        base_model = Pipeline(
            [
                ("scaler", StandardScaler(with_mean=False, with_std=False)),
                ("imputer_mnar", Imputer(method=None, strategy=strategy)),
                ("imputer_mar", Imputer(method=None, strategy=strategy)),
            ]
        )

        match strategy:
            case "mixed":
                params: dict[str, list[Any]] = {
                    "scaler": [
                        StandardScaler(with_mean=False, with_std=False),
                        MinMaxScaler(clip=True),
                        RobustScaler(with_centering=True, with_scaling=False),  # Median centering
                        Normalize(method="log2"),
                        Normalize(method="quantile_sample"),
                    ],
                    "imputer_mnar": [
                        Imputer(method="qrilc", strategy=strategy),
                        Imputer(method="minprob", strategy=strategy),
                    ],
                    "imputer_mar": [
                        Imputer(method="knn", strategy=strategy),
                        Imputer(method="mice", strategy=strategy),
                        Imputer(method="mean", strategy=strategy),
                        Imputer(method="miss_forest", strategy=strategy),
                    ],
                }
            case "mnar":
                params = {
                    "scaler": [
                        StandardScaler(with_mean=False, with_std=False),
                    ],
                    "imputer_mnar": [
                        Imputer(method="qrilc", strategy=strategy),
                        Imputer(method="minprob", strategy=strategy),
                    ],
                    "imputer_mar": [
                        Imputer(method=None, strategy=strategy),
                    ],
                }
            case "mar":
                params = {
                    "scaler": [
                        StandardScaler(with_mean=False, with_std=False),
                    ],
                    "imputer_mnar": [
                        Imputer(method=None, strategy=strategy),
                    ],
                    "imputer_mar": [
                        Imputer(method="knn", strategy=strategy),
                        Imputer(method="mice", strategy=strategy),
                        Imputer(method="mean", strategy=strategy),
                        Imputer(method="miss_forest", strategy=strategy),
                    ],
                }

        model_params: SklearnModels = {"random_state": self.random_state}

        # Initialise a single model pipeline
        match model:
            case ModelsEnum.XGBOOST:
                # Used to balance the sampling in XGB, use  sqrt(# 0s/# 1s) as suggsted for inbalanced datasets
                scale_pos_weight = np.sqrt(sum(self.df.target == 0) / sum(self.df.target == 1))
                model_object = Pipeline([*base_model.steps, (model.value, xgb.XGBClassifier(**model_params))])

                if optimize:
                    params |= {
                        f"{model.value}__random_state": [self.random_state, self.random_state],
                        f"{model.value}__max_depth": [2, 3],
                        f"{model.value}__learning_rate": [0.001, 0.1, {"log": True}],
                        f"{model.value}__n_estimators": [50, 500],
                        f"{model.value}__subsample": [0.1, 1.0],
                        f"{model.value}__colsample_bytree": [0.1, 1.0],
                        f"{model.value}__min_child_weight": [0, 10],
                        f"{model.value}__gamma": [0, 5],
                        f"{model.value}__lambda": [1e-8, 10.0, {"log": True}],
                        f"{model.value}__alpha": [1e-8, 10.0, {"log": True}],
                        f"{model.value}__n_jobs": [1, 1],
                        f"{model.value}__scale_pos_weight": [scale_pos_weight, scale_pos_weight],
                    }
                else:
                    params |= {
                        f"{model.value}__n_estimators": [200, 200],
                        f"{model.value}__learning_rate": [0.05, 0.05],
                        f"{model.value}__reg_lambda": [0, 0],
                        f"{model.value}__max_depth": [2, 2],
                        f"{model.value}__n_jobs": [1, 1],
                        f"{model.value}__scale_pos_weight": [scale_pos_weight, scale_pos_weight],
                    }

            case ModelsEnum.BALANCED_RANDOM_FOREST:

                model_object = Pipeline(
                    [*base_model.steps, (model.value, BalancedRandomForestClassifier(**model_params))]
                )

                if optimize:
                    params |= {
                        f"{model.value}__n_estimators": [50, 1000],
                        f"{model.value}__max_depth": [2, 10],
                        f"{model.value}__min_samples_split": [2, 10],
                        f"{model.value}__min_samples_leaf": [3, 10],
                        f"{model.value}__max_features": ["sqrt", "log2", None],
                        f"{model.value}__criterion": ["gini", "entropy"],
                        f"{model.value}__min_weight_fraction_leaf": [0.0, 0.5],
                        f"{model.value}__sampling_strategy": ["all"],
                        f"{model.value}__replacement": [False],
                        f"{model.value}__bootstrap": [False],
                    }
                else:
                    params |= {
                        f"{model.value}__n_estimators": [500, 500],
                        f"{model.value}__criterion": ["gini", "gini"],
                        f"{model.value}__max_depth": [10, 10],
                        f"{model.value}__replacement": [False],
                        f"{model.value}__bootstrap": [False],
                        f"{model.value}__sampling_strategy": ["all"],
                    }

            case ModelsEnum.RANDOM_FOREST:

                model_object = Pipeline([*base_model.steps, (model.value, RandomForestClassifier(**model_params))])

                if optimize:
                    params |= {
                        f"{model.value}__n_estimators": [50, 500],
                        f"{model.value}__max_depth": [3, 5],
                        f"{model.value}__min_samples_split": [2, 10],
                        f"{model.value}__min_samples_leaf": [3, 10],
                        f"{model.value}__max_features": ["sqrt", "log2", None],
                        f"{model.value}__criterion": ["gini", "entropy", "log_loss"],
                        f"{model.value}__min_weight_fraction_leaf": [0.0, 0.5],
                        f"{model.value}__max_leaf_nodes": [10, 100],
                    }
                else:
                    params |= {
                        f"{model.value}__n_estimators": [200, 200],
                        f"{model.value}__criterion": ["gini", "gini"],
                        f"{model.value}__max_depth": [3, 3],
                    }

            case _:
                raise NotImplementedError("Model type not recognised")

        if self.fixed_params:
            params = {key: value for key, value in params.items() if model.value not in key}
            params |= self.fixed_params
            logger.info(self.fixed_params)

        params |= {
            # MICE
            "imputer_mar__estimator": [LinearRegression(), RandomForestRegressor(), XGBRegressor(n_estimators=500)],
            #  KNN
            "imputer_mar__weights": ["uniform", "distance"],
            "imputer_mar__n_neighbors": [4, 5, 6, 7, 8],
            # QRILC and minprob
            "imputer_mnar__tune_sigma": [0.5, 1.0],
            # minprob
            "imputer_mnar__quantile": [0.1, 0.01],
        }

        if model_only:
            only_model_params = {key: value for key, value in params.items() if model.value in key}
            params = only_model_params

        return model_object, params

    def _optimise_model(
        self, model_object: Pipeline, parameters_grid: dict[str, list[Any]], df: pd.DataFrame, optimize: bool
    ) -> tuple[Pipeline, dict[str, Any] | list[FrozenTrial]]:
        """Runs a grid search to optimise a model on stratified CV.

        Parameters
        ----------
        model_object
            Pipeline containing the baseline model.
        parameters_grid
            Parameters we want to compare in grid search.
        df
            Training data from ml_ready_data.
        optimize
            To optimize the parameters of the model.

        Returns
        -------
        best_estimator
            The best parameters found in optimisation.
        grid_results
            The trials from Optuna.
        """
        X, y = df.drop("target", axis=1), df["target"]

        sampler = (
            TPESampler(seed=self.random_state) if len(self.scorers) == 1 else NSGAIISampler(seed=self.random_state)
        )
        study = create_study(directions=["maximize"] * len(self.scorers), sampler=sampler)
        n_trials = 1 if not optimize else self.config.data.optimization.optuna.trials
        objective = partial(self._optuna_objective, params=parameters_grid, model_pipeline=model_object, X=X, y=y)
        study.optimize(objective, n_trials=n_trials, gc_after_trial=True, show_progress_bar=True)

        if len(self.scorers) == 1:
            best_params = study.best_params
        else:
            # Get the first best trial as most optimal on pareto front
            best_params = study.best_trials[0].params

        best_estimator = model_object.set_params(**best_params).fit(X, y)
        grid_results = study.trials
        self.study = study

        if optimize:
            self.fixed_params = {key: [value, value] for key, value in best_params.items()}

        return best_estimator, grid_results

    def _optuna_objective(
        self,
        trial: Trial,
        params: dict[str, list[float | dict[str, Any]]],
        model_pipeline: Pipeline,
        X: pd.DataFrame,
        y: pd.Series[float],
    ) -> list[float]:
        fs_config = self.config.data.feature_selection
        cv = RepeatedStratifiedKFold(
            n_splits=fs_config.cross_validation_splits,
            n_repeats=fs_config.cross_validation_repeats,
            random_state=self.random_state,
        )

        param_dict: dict[str, Trial] = {}
        kwargs: dict[str, Any] = {}
        for param_name, param_values in params.items():
            args = param_values[:2]
            if len(param_values) > 2:
                kwargs = cast(dict[str, Any], param_values[-1])
            else:
                kwargs = {}

            if all(isinstance(val, int) and not isinstance(val, bool) for val in args):
                trial_value = "trial.suggest_int"
            elif all(isinstance(val, float) for val in args):
                trial_value = "trial.suggest_float"
            else:
                trial_value = "trial.suggest_categorical"
                args = [param_values]  # type: ignore
                kwargs = {}

            param_dict |= {param_name: eval(trial_value)(param_name, *args, **kwargs)}
        model = model_pipeline.set_params(**param_dict)

        obj_metric = cross_validate(model, X, y, cv=cv, scoring=self._optuna_scorer(self.scorers), n_jobs=-2)

        obj_score = [obj_metric[key].mean() for key in obj_metric if "test" in key]
        return obj_score

    def _optuna_scorer(self, scorer: list[str]) -> dict[str, Any]:
        def sens_at_99_spec(y_true: np_array, y_score: np_array) -> float:
            roc_metrics = compute_roc_curve([y_score], [y_true], np.linspace(0, 1, 101))

            return roc_metrics["sens_spec_99"][0]

        sklearn_scorers = get_scorer_names()
        custom_scorers = {
            "sens@99spec": sens_at_99_spec,
        }
        alias_scorers = {
            "mcc": "matthews_corrcoef",
            "auc": "roc_auc",
        }

        if any([score not in sklearn_scorers + list(custom_scorers) + list(alias_scorers) for score in scorer]):
            raise ValueError(f"One of {scorer} not not in available scoring metrics.")

        scorer_funcs: dict[str, Any] = {}
        for scorer_name in scorer:
            if scorer_name in sklearn_scorers:
                scorer_funcs[scorer_name] = get_scorer(scorer_name)
            elif scorer_name in custom_scorers:
                custom_scorer = custom_scorers[scorer_name]
                scorer_funcs[scorer_name] = make_scorer(custom_scorer, needs_proba=True)
            else:
                scorer_funcs[scorer_name] = get_scorer(alias_scorers[scorer_name])

        return scorer_funcs
