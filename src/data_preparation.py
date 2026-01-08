from __future__ import annotations

from copy import deepcopy
from itertools import product
from typing import Any
from typing import cast
from typing import Literal
from typing import TypeAlias

import numpy as np
import pandas as pd
from joblib import delayed
from joblib import Parallel
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .cli.toml_parser import TomlParser
from .data_loader import save_deliverables
from .features.performance_metrics import compute_model_performance
from .features.performance_metrics import youdens_j
from .figure_generation import compute_roc_curve


OptimalMetrics: TypeAlias = Literal[
    "f1_score",
    "sensitivity",
    "specificity",
    "pos_pred_value",
    "neg_pred_value",
    "average_precision",
    "roc_auc",
    "pr_auc",
    "sen@99spec",
    "sen@95spec",
    "sen@90spec",
]


class DataPreparation:
    """Prepares the data by running through preprocessing steps.

    Parameters
    ----------
    config
        The full config options.

    Attributes
    ----------
    random_state
        Random state for reproducibility extracted from config.
    """

    def __init__(
        self,
        config: TomlParser,
    ) -> None:
        self.config = config

        self.random_state = self.config.data.random_state

    def get_best_pipeline(
        self,
        data: pd.DataFrame,
        base_pipeline: Pipeline,
        param_grid: dict[str, Any],
        metrics: tuple[OptimalMetrics, ...] = ("sen@95spec", "roc_auc", "pr_auc", "pos_pred_value", "neg_pred_value"),
    ) -> Pipeline:
        """Runs through preprocessing combinations to get the best pipeline.

        Parameters
        ----------
        data
            Training data from ml_ready_data.
        base_pipeline
            Pipeline containing the preprocessing and model steps.
        param_grid
            Parameters we want to compare in grid search.
        metrics, optional
            The sequence of metrics to evaluate the best preprocessing
            pipeline on. The first metric listed will be the one used for
            running the rest of the pipeline, by default
            ("sen@95spec", "roc_auc", "pr_auc", "pos_pred_value", "neg_pred_value").

        Returns
        -------
        best_pipeline
            Fitted version of the best_pipeline.
        """
        X, y = data.drop("target", axis=1), data["target"]
        inst_names = ["scaler", "imputer_mnar", "imputer_mar"]

        # Get all combinations for the preprocessing part of the pipeline
        all_combinations = list(product(*[self._get_param_combinations(param_grid, key) for key in inst_names]))

        # All pipelines to include preprocessing steps and default model steps
        pipelines = [
            Pipeline([*[(name, inst) for name, inst in zip(inst_names, combo)], base_pipeline.steps[-1]])
            for combo in all_combinations
        ]

        # Parallelize preprocesisng combinations and evaluate metrics
        result = cast(
            list[tuple[Pipeline, pd.DataFrame]],
            Parallel(n_jobs=-2)(
                delayed(self._evaluate_pipeline)(
                    X,
                    y,
                    pipeline,
                )
                for pipeline in pipelines
            ),
        )

        optimised_pipelines, model_performance = [data[0] for data in result], pd.concat(
            [data[1] for data in result]
        ).reset_index(drop=True)

        top_n_pipelines = model_performance.sort_values(list(metrics), ascending=False).iloc[: len(metrics)]
        best_pipelines = {
            f"best_pipeline_{i + 1}": optimised_pipelines[idx] for i, idx in enumerate(top_n_pipelines.index)
        }

        save_deliverables(
            self.config,
            pickled_dict=best_pipelines,
            csv_dict={"model_performance": model_performance},
            include_metadata=True,
        )

        return best_pipelines["best_pipeline_1"].fit(X, y)

    def _get_param_combinations(self, parameter_grid: dict[str, Any], pipeline_step: str) -> list[Any]:
        """Gets all parameter combinations from a grid.

        The `pipeline_step` is used to get the relevant parameters from
        the `parameter_grid`.

        Parameters
        ----------
        parameter_grid
            The parameters to get combinations of.
        pipeline_step
            The step in the pipeline.

        Returns
        -------
            A list of all combinations of parameters for each step.
        """
        total_options = []
        extra_opts = {
            key.replace(f"{pipeline_step}__", ""): value
            for key, value in parameter_grid.items()
            if f"{pipeline_step}__" in key
        }

        combinations = list(product(*extra_opts.values()))
        for step in parameter_grid[pipeline_step]:
            for combo in combinations:
                new_step = deepcopy(step)
                inst = new_step.set_params(**{key: value for key, value in zip(extra_opts, combo)})
                if inst in total_options:
                    continue

                total_options.append(inst)

        return total_options

    def _get_preprocessed_data(
        self, X: pd.DataFrame, y: pd.Series[float], full_pipeline: Pipeline, name_only: bool = False
    ) -> tuple[pd.DataFrame, str]:
        name = ""
        train_df = deepcopy(X)

        # Get the name combination always and only the imputed dataset
        # if `name_only` is False
        for _, step in full_pipeline.steps[:-1]:
            name += f"{step}_"
            if not name_only:
                train_df = step.fit_transform(train_df, y)

        train_df.columns = X.columns

        name = name.rstrip("_")

        return train_df, name

    def _evaluate_pipeline(
        self,
        X: pd.DataFrame,
        y: pd.Series[float],
        pipeline: Pipeline,
    ) -> tuple[Pipeline, pd.DataFrame]:
        """Runs evaluation metrics on a preprocessing pipeline.

        A 10% validation set is taken from `X` for evaluating the metrics.

        Parameters
        ----------
        X
            The training dataset.
        y
            The training targets.
        pipeline
            The full pipeline, including both preprocessing and model steps.

        Returns
        -------
            An optimised pipeline and model performance metrics.
        """
        out_path = deepcopy(self.config.cli_data.s3_output_path)
        fpr_steps = np.linspace(0, 1, 101)

        # Go through each step in the preprocessing pipeline
        _, name = self._get_preprocessed_data(X, y, pipeline, name_only=True)

        # test_size=10%
        X_train, X_valid, y_train, y_valid = cast(
            tuple[pd.DataFrame, pd.DataFrame, "pd.Series[float]", "pd.Series[float]"],
            train_test_split(X, y, test_size=0.1, random_state=self.random_state, stratify=y),
        )

        new_pipeline = Pipeline(steps=[pipeline.steps[0]])
        # Scale first with fit_transform
        X_train = pd.DataFrame(
            pipeline.steps[0][1].fit_transform(X_train.to_numpy(), y_train.to_numpy()),
            index=X_train.index,
            columns=X_train.columns,
        )
        for step_name, step in pipeline.steps[1:]:
            pipe = step.fit(X_train.to_numpy(), y_train.to_numpy())
            new_pipeline = Pipeline(steps=[*new_pipeline.steps, (step_name, pipe)])

        # Compute optimal threshold for prediction
        y_train_scores = new_pipeline.predict_proba(X_train.to_numpy())
        y_valid_scores = new_pipeline.predict_proba(X_valid.to_numpy())

        thresh = youdens_j(y_valid.to_numpy(), y_valid_scores)

        model_metrics, y_scores = compute_model_performance(
            y_valid_scores,
            y_valid.to_numpy(),
            y_train_scores,
            y_train.to_numpy(),
            threshold=thresh,
        )

        roc_metrics = compute_roc_curve([y_scores], [y_valid.to_numpy()], fpr_steps)
        sen_spec = pd.DataFrame.from_records(
            zip(roc_metrics["mean_tprs"], 1 - fpr_steps),
            columns=["sensitivity", "specificity"],
        ).set_index("specificity")

        for spec_val in [99, 95, 90]:
            model_metrics |= {f"sen@{spec_val}spec": sen_spec.loc[spec_val / 100].to_numpy()[0]}

        model_performance = (
            pd.DataFrame.from_dict(model_metrics, orient="index", columns=[name])
            .T.reset_index()
            .rename({"index": "pipeline"}, axis=1)
        )

        # roc_metrics["roc_auc"] uses the average of interp_tprs
        model_performance["roc_auc"] = np.mean(np.array(roc_metrics["roc_auc"])).astype(float)
        self.config.cli_data.s3_output_path = out_path

        return new_pipeline, model_performance
