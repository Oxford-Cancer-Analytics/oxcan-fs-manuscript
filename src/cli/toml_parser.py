from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import fields
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any
from typing import Literal
from typing import NamedTuple

import toml
from typing_extensions import Self

from .cli_options import AugmentationEnum
from .cli_options import FeatureSelectionEnum
from .cli_options import ModelsEnum
from .cli_options import StatisticsEnum


class Headers(NamedTuple):
    """Holds the toml keys for top-level and headers."""

    top_level: list[str]
    headers: list[str]


class _Descriptors:
    def to_dict(self, true_values: bool = False) -> dict[str, Any]:
        """Recursively covnerts all dataclass instances into dicts.

        Parameters
        ----------
        true_values
            Return only the True values in the dict.

        Returns
        -------
            The dataclass as a dict.
        """
        data_dict = asdict(self)  # type: ignore
        if true_values:
            return self._get_true_values_only(data_dict)
        else:
            return data_dict

    def __getitem__(self, key: str) -> Any:
        """Gets the value from the current instance.

        Parameters
        ----------
        key
            The key to lookup on self.

        Returns
        -------
            The value associated with the `key`.
        """
        return getattr(self, key)

    def __setitem__(self, item: str | dict[str, Any], value: Any) -> None:
        """Sets the `value` by `item` on self.

        Parameters
        ----------
        item
            The item to lookup in `self.__dict__`.
        value
            The value to set.
        """
        if isinstance(item, str) and item in [
            "feature_selection",
            "augmentation",
            "stats",
        ]:
            # Set all values to False set the CLI item to True
            for items in fields(self.__dict__[item]):
                setattr(self.__dict__[item], items.name, False)
        elif isinstance(item, dict):
            self.__dict__ |= item
        else:
            self.__dict__[item] = value

    def _get_true_values_only(self, data: dict[str, Any], _nested_key: str | None = None) -> dict[str, Any]:
        """Gets the True values only from the toml data.

        Parameters
        ----------
        data
            The toml data as a dict.
        _nested_key, optional
            Internally used to compare nested dicts, by default None

        Returns
        -------
            `data` with only True values.
        """
        returned_dict: dict[str, Any] = {}
        for header, values in data.items():
            if isinstance(values, dict):
                nested_dict = self._get_true_values_only(values, header)
                if nested_dict:
                    if header in nested_dict and _nested_key:
                        # Check if any values have not already been assigned to the return dict
                        if _nested_key not in returned_dict:
                            returned_dict[_nested_key] = {}

                        returned_dict[_nested_key] |= nested_dict
                    else:
                        returned_dict |= nested_dict
            elif values:
                if _nested_key:
                    if _nested_key not in returned_dict:
                        returned_dict[_nested_key] = {}
                    returned_dict[_nested_key][header] = values
                else:
                    returned_dict[header] = values
        return returned_dict


@dataclass
class TomlDataOptimization(_Descriptors):
    """The Optimization header options.

    Attributes
    ----------
    optuna
        The Optuna subheader options.
    """

    @dataclass
    class Optuna(_Descriptors):
        """The Optuna subheader options.

        Attributes
        ----------
        scorer
            The scoring method, or combination of methods, to use for optimization.
        trials
            The number of iterations for bayesian search.
        before_feature_selection
            To search the hyperparameter space before feature selection happens.
        after_rfa
            To search the hyperparameter space after feature rfa.
        before_validation
            To search the hyperparameter space before model validation happens.
        """

        scorer: str | list[str] = "sens@99spec"
        trials: int = 200
        before_feature_selection: bool = False
        after_rfa: bool = True
        before_validation: bool = False

    optuna: Optuna = Optuna()


@dataclass
class TomlDataAugmentation(_Descriptors):
    """The Augmentation header options.

    Attributes
    ----------
    smote
        The SMOTE subheader options.
    """

    @dataclass
    class Smote(_Descriptors):
        """The SMOTE subheader options.

        Attributes
        ----------
        new_cancer_ratio, optional
            A new ratio of cancer patients, by default '0.5'.
        new_total_samples, optional
            A new total patients, by default '0'.
        """

        new_cancer_ratio: float = 0.5
        new_total_samples: int = 0

    smote: Smote = Smote()


@dataclass
class TomlDataStats(_Descriptors):
    """The stats header options.

    Attributes
    ----------
    pvalue_thresh
        The threshold value.
    power
        The power analysis.
    alpha
        The alpha.
    """

    pvalue_thresh: float = 0.05
    power: float = 0.95
    alpha: float = 0.05


@dataclass
class TomlDataFeatureSelection(_Descriptors):
    """The feature selection header options.

    Attributes
    ----------
    multisurf_cross_validation_splits
        The number of cross validation splits for MultiSURF.
    cross_validation_splits
        The number of cross validation splits.
    cross_validation_repeats
        The number of repeats over seeds.
    number_of_mi_features
        The number of features to select during MI
    number_of_features
        The number of features to stop at for feature selection.
    addition
        The Addition subheader options.
    """

    @dataclass
    class Addition(_Descriptors):
        """The Addition subheader options.

        Attributes
        ----------
        method
            Which RFA method, either "feature_importance" or "model_performance".
        importance_selection
            The type of feature importance selection method, one of "permutation_importance", "tree_importance",
            "mutual_information", "shap", or "eject_shap".
        """

        method: Literal["feature_importance", "model_performance"] = "feature_importance"
        importance_selection: Literal[
            "permutation_importance", "tree_importance", "mutual_information", "shap", "eject_shap"
        ] = "shap"

    multisurf_cross_validation_splits: int = 50
    cross_validation_splits: int = 4
    cross_validation_repeats: int = 10
    number_of_mi_features: float | Literal["effective_dim"] = "effective_dim"
    number_of_features: int = 150
    addition: Addition = Addition()


@dataclass
class CliConfigs:
    """Required string values for command line interface configuration.

    input_path
        The local path for input data.
    output_path
        The local path for output results.
    stats
        An enum for stats related options.
    s3_input_path
        The Key path in the S3 URI for inputs.
    s3_output_path
        The Key path in the S3 URI for outputs - not manually specified.
    s3_bucket
        The Bucket in the S3 URI.
    s3_labelled_cohort_key
        The S3 key for the labelled cohort file.
    imputation_strategy
        The method for missing value evaluation and imputation
        Can be either "mar", "mnar" or "mixed".
    use_batch_corrected
        Whether to use batch corrected data for model building/feature selection.
        Batch correction will still be carried out regardless of selection.
    model
        An enum for model related options.
    feature_selection
        An enum for feature selection related options.
    augmentation
        An enum for augmentation related options.
    dry_run
        Skips uploading to S3.
    rec_feat_add
        Runs the recursive feature addition.
    use_full_data
        Use all data or training set.
    prepare_data
        To run through the combinations of preprocessing steps.
    best_pipeline
        The path in the S3 URI for the best pipeline.
    """

    input_path: Path | None
    output_path: Path | None
    stats: StatisticsEnum
    s3_input_path: Path | None
    s3_output_path: Path | None
    s3_bucket: str | None
    s3_labelled_cohort_key: Path | None
    imputation_strategy: Literal["mar", "mnar", "mixed"]
    use_batch_corrected: bool
    model: ModelsEnum
    feature_selection: FeatureSelectionEnum
    augmentation: AugmentationEnum
    dry_run: bool
    rec_feat_add: bool
    use_full_data: bool
    prepare_data: bool
    best_pipeline: Path


@dataclass
class TomlData(_Descriptors):
    r"""The overall toml data structure.

    Attributes
    ----------
    random_state
        The random seed to use.
    scale_transform
        Applies StandardScaler to training data and then transforms on the holdout.
    sub_cohort
        Applies a filter to select only sub-cohorts for analysis.
        The regex r"^control\|.*$" matches any string that starts with "control|" for example,
        control|early or control|sclc early depending on your labels.
    pca
        Applies StandardScaler and then pca to training data and then transforms on
        the  holdout.
    stats
        A header key, see :class:`TomlDataStats`.
    feature_selection
        A header key, see :class:`TomlDataFeatureSelection`.
    augmentation
        A header key, see :class:`TomlDataAugmentation`.
    optimization
        A header key, see :class:`TomlDataOptimization`.
    """

    random_state: int = 0
    sub_cohort: Literal["all", r"^control\|.*$"] = "all"
    scale_transform: bool = True
    pca: bool = True

    # Stats
    stats: TomlDataStats = TomlDataStats()

    # Feature Selection
    feature_selection: TomlDataFeatureSelection = TomlDataFeatureSelection()

    # Augmentation
    augmentation: TomlDataAugmentation = TomlDataAugmentation()

    # Optimization
    optimization: TomlDataOptimization = TomlDataOptimization()


class TomlParser:
    """Parses the toml in the config.toml file.

    Parameters
    ----------
    config
        The Path object or str for the config file.

    Attributes
    ----------
    data
        The parsed toml data.
    cli_data
        The data options from the CLI.
    """

    def __init__(self, config: Path) -> None:
        self.config = config
        self.data = TomlData()

        self.cli_data: CliConfigs

    @property
    def headers(self) -> Headers:
        """Class property to get headers from :class:`self.data`.

        Returns
        -------
            A Headers object.
        """
        return self._get_headers(self.data.to_dict())

    def read(self) -> Self:
        """Reads the config and parses the information.

        Returns
        -------
            Itself.
        """
        with open(self.config) as handle:
            toml_dict = toml.load(handle)

        processed_toml = self._format_toml(toml_dict)
        self.data = TomlData(**processed_toml)
        return self

    def add_cli_config(self, cli_configs: dict[str, Any]) -> None:
        """Adds a cli_config object.

        Command line interface configurations.

        Parameters
        ----------
        cli_configs
            User specified configurations
        """
        self.cli_data = CliConfigs(**cli_configs)

    def _get_headers(self, data: dict[str, Any]) -> Headers:
        """Gets the headers in the config file.

        Parameters
        ----------
        data
            The toml file as a string.

        Returns
        -------
            A Headers object, which includes both top-level
            and header keys.
        """
        headers = [key for key in data if isinstance(data[key], dict)]
        top_level_keys = [key for key in data if key not in headers]
        return Headers(top_level_keys, headers)

    def _format_toml(self, data: dict[str, Any]) -> dict[str, Any]:
        """Formats the toml data for dataclass attribute access.

        Parameters
        ----------
        data
            The toml file as a string.

        Returns
        -------
            The formatted toml as a string.
        """

        def create_dataclass_instance(data: dict[str, Any], cls: type[_Descriptors]) -> Any:
            field_types = {f.name: f.type for f in fields(cls)}  # type: ignore
            instance_data = {}
            for key, value in data.items():
                try:
                    class_subheading = eval(f"cls.{field_types[key]}")
                except AttributeError:
                    class_subheading = None

                if isinstance(value, dict) and key in field_types and is_dataclass(class_subheading):
                    instance_data[key] = create_dataclass_instance(value, class_subheading)  # type: ignore
                else:
                    instance_data[key] = value
            return cls(**instance_data)

        class_map = {
            "stats": TomlDataStats,
            "feature_selection": TomlDataFeatureSelection,
            "augmentation": TomlDataAugmentation,
            "optimization": TomlDataOptimization,
        }

        formatted_data = {}
        for key, value in data.items():
            if key in class_map:
                formatted_data[key] = create_dataclass_instance(value, class_map[key])
            else:
                formatted_data[key] = value

        return formatted_data
