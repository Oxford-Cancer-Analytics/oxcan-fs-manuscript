from __future__ import annotations

import io
import logging
import pickle
import subprocess
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any
from typing import cast
from typing import Literal
from typing import overload
from typing import Protocol
from typing import TypeAlias
from typing import TypedDict

import boto3
import Levenshtein
import pandas as pd
import toml
from joblib import delayed
from joblib import Parallel
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing_extensions import NotRequired
from typing_extensions import Unpack

from .cli.toml_parser import TomlParser


MLReadyDataType: TypeAlias = tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]


class ExtractedData(TypedDict):
    """Data extracted from S3.

    Attributes
    ----------
    labelled_cohort
        The associated patient data.
    train
        The training portion of the dataframe.
    holdout
        The validation portion of the dataframe.
    """

    labelled_cohort: pd.DataFrame
    train: pd.DataFrame
    holdout: pd.DataFrame


logger = logging.getLogger(__name__)


def check_data(data: ExtractedData) -> MLReadyDataType:
    """Check data structured properly.

    Parameters
    ----------
    data
        The extracted data dictionary.

    Returns
    -------
        Tuple form of data checked for structural errors.
    """
    labelled_cohort, train, holdout = data["labelled_cohort"], data["train"], data["holdout"]
    patient_ids = train.patient_id.tolist() + holdout.patient_id.tolist()

    if train.isna().any().any():
        assert holdout.isna().any().any()
        print("Data contains missing values")
        logger.warning("Data contains missing values".upper())

    if not train.isna().any().any():
        assert not holdout.isna().any().any()
        logger.info("Data does not contain missing values")

    assert all(["patient_id" in cast(pd.DataFrame, frame).columns for frame in data.values()])
    assert all(pid in labelled_cohort["patient_id"].to_numpy() for pid in patient_ids)
    assert all(isinstance(data, pd.DataFrame) for data in data.values())

    logger.info("Data structural checks performed")
    return labelled_cohort, train, holdout


def _process_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove unnamed columns from dataframe.

    Parameters
    ----------
    df
        Input dataframe.

    Returns
    -------
        Dataframe with unnamed columns removed.
    """
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


def _load_and_process_data_file(
    file_loader: Callable[[str], pd.DataFrame], filename: str, other_features: dict[str, list[Any]]
) -> pd.DataFrame:
    """Load and process a single data file (train.csv or holdout.csv).

    Parameters
    ----------
    file_loader
        Function to load the file (e.g., extract_s3 or pd.read_csv).
    filename
        Name of the file to load.
    other_features
        Dictionary to store additional feature information.

    Returns
    -------
        Processed dataframe.
    """
    df = file_loader(filename)
    df = _process_dataframe_columns(df)

    return df


def extract_data(
    bucket: str, directory: Path, labelled_cohort_key: Path
) -> tuple[ExtractedData, dict[str, list[Any]]]:
    """Extract the ml ready data from S3.

    3 files: labelled_cohort, train, holdout

    Parameters
    ----------
    bucket
        AWS S3 bucket.
    directory
        The path to the files.
    labelled_cohort_key
        The S3 key for the labelled cohort file.

    Returns
    -------
        Training, holdout, and the labelled cohort as well as any other extra feature identifiers.
    """
    data_dict = {}
    other_features: dict[str, list[Any]] = {}

    def s3_file_loader(filename: str) -> pd.DataFrame:
        return cast(pd.DataFrame, extract_s3(bucket=bucket, key=str(directory / filename)))

    for filename in ["train.csv", "holdout.csv"]:
        df = _load_and_process_data_file(s3_file_loader, filename, other_features)
        data_dict[filename.split(".")[0]] = df

    # Load the labelled_cohort
    data_dict["labelled_cohort"] = extract_s3(bucket=bucket, key=str(labelled_cohort_key))

    return ExtractedData(**data_dict), other_features  # type: ignore[typeddict-item]


def extract_local_data(directory: Path) -> tuple[ExtractedData, dict[str, list[Any]]]:
    """Extract the ml ready data from local files.

    3 files: train.csv, holdout.csv, labelled_cohort.csv

    Parameters
    ----------
    directory
        The local path to the directory containing the data files.

    Returns
    -------
        Training, holdout, and the labelled cohort as well as any other extra feature identifiers.

    Raises
    ------
    FileNotFoundError
        If the data directory or required files (train.csv, holdout.csv, labelled_cohort.csv) do not exist.
    """
    data_dict = {}
    other_features: dict[str, list[Any]] = {}

    # Check if directory exists
    if not directory.exists():
        raise FileNotFoundError(f"Data directory not found: {directory}")

    def local_file_loader(filename: str) -> pd.DataFrame:
        file_path = directory / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")
        logger.info(f"Loading local file: {file_path}")
        return pd.read_csv(file_path, low_memory=False)

    for filename in ["train.csv", "holdout.csv"]:
        df = _load_and_process_data_file(local_file_loader, filename, other_features)
        data_dict[filename.split(".")[0]] = df

    # Load the labelled_cohort - try labelled_cohort.csv first, then other common names
    cohort_filenames = ["labelled_cohort.csv", "patient_metadata.csv", "cohort.csv", "metadata.csv"]
    labelled_cohort = None

    for cohort_filename in cohort_filenames:
        cohort_path = directory / cohort_filename
        if cohort_path.exists():
            logger.info(f"Loading local cohort file: {cohort_path}")
            labelled_cohort = pd.read_csv(cohort_path, low_memory=False)
            break

    if labelled_cohort is None:
        raise FileNotFoundError(
            f"No labelled cohort file found in {directory}. " + f"Expected one of: {', '.join(cohort_filenames)}"
        )

    data_dict["labelled_cohort"] = labelled_cohort

    return ExtractedData(**data_dict), other_features  # type: ignore[typeddict-item]


@overload
def extract_s3(bucket: str, key: str, folder: Literal[False] = False) -> pd.DataFrame | Pipeline | None: ...


@overload
def extract_s3(bucket: str, key: str, folder: Literal[True]) -> tuple[list[pd.DataFrame | Pipeline], list[str]]: ...


def extract_s3(
    bucket: str, key: str, folder: bool = False
) -> pd.DataFrame | None | Pipeline | tuple[list[pd.DataFrame | Pipeline], list[str]]:
    """Extracts the csv data.

    Parameters
    ----------
    bucket
        The name of the S3 bucket.
    key
        The key path to the data on S3. If the `key` is a folder, then
        get all files from the folder.
    folder, optional
        Whether the `key` is a folder or not, by default False.

    Returns
    -------
        The data as a dataframe, a list of dataframes with their keys,
         or None if they key is invalid.

    Raises
    ------
    ValueError
        When filtering a folder of objects and there are no files after filtering.
    """
    s3 = boto3.client("s3")

    if folder:
        if "*" in key:
            key, file_key = key.split("*")
        else:
            file_key = key

        keys = s3.list_objects(Bucket=bucket, Prefix=key)
        key_contents = [*keys["Contents"]]

        # The maximum limit is 1000. If it is reached, we should check if there are more
        # files not loaded in the first run
        if len(keys["Contents"]) == 1_000:
            while len(keys["Contents"]) == 1_000:
                keys = s3.list_objects(Bucket=bucket, Prefix=key, Marker=key_contents[-1]["Key"])
                key_contents.extend(keys["Contents"])

        key_pairs = list(filter(lambda x: file_key in x["Key"], key_contents))

        if not key_pairs:
            raise ValueError(f"No key pairs found that match {file_key[1:]}.")

        def _extract_all_data(key_pair: dict[str, Any]) -> tuple[pd.DataFrame | Pipeline, str]:
            name = key_pair["Key"]
            df = cast(pd.DataFrame | Pipeline, extract_s3(bucket, name))

            return df, name

        result = Parallel(n_jobs=-2)(
            delayed(_extract_all_data)(key_pair) for key_pair in filter(lambda x: file_key in x["Key"], key_contents)
        )
        return (
            [cast(tuple[pd.DataFrame | Pipeline, str], data)[0] for data in result],
            [cast(tuple[pd.DataFrame | Pipeline, str], data)[1] for data in result],
        )

    logger.info(f"Loading {key}")
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
    except s3.exceptions.NoSuchKey:
        try:
            # extract the epoch time stamp from the path (key)
            model_name = key.split("/")[-3]

            # split the old key from the right into two parts
            old_key = key.rsplit("/", 1)

            # generate a new key including the epoch associated with that path
            new_key = f"{old_key[0]}/{model_name}_{old_key[1]}"

            logger.info(f"{key.split('/')[-1]} is invalid for bucket '{bucket}'; loading {new_key}")
            obj = s3.get_object(Bucket=bucket, Key=new_key)
        except (s3.exceptions.NoSuchKey, IndexError):
            logger.info(f"Warning: {key=} is invalid for bucket '{bucket}'; returned None.")
            return None

    if ".txt" in key:
        return pd.read_csv(obj["Body"], sep="\t")
    elif ".csv" in key:
        return pd.read_csv(obj["Body"], low_memory=False)
    elif ".pkl" in key:
        return pickle.load(obj["Body"])
    else:
        return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def apply_pca(training: pd.DataFrame, validation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply PCA to the training and validation data.

    Parameters
    ----------
    training
        Training data.
    validation
        Validation data.

    Returns
    -------
        DataFrames for training and validation data with PCA applied.
    """
    # Standardize the numerical columns
    scaler = StandardScaler()
    numerical_columns = [c for c in training.select_dtypes(include="number").columns if c not in ["target"]]
    non_numerical_columns = [c for c in training.columns if c not in numerical_columns]
    training_numerical_scaled = scaler.fit_transform(training[numerical_columns])
    validation_numerical_scaled = scaler.transform(validation[numerical_columns])

    # Apply PCA to scaled numerical columns
    pca = PCA()
    training_pca = pca.fit_transform(training_numerical_scaled)
    validation_pca = pca.transform(validation_numerical_scaled)

    # Create DataFrame with PCA components
    pca_column_names = [f"PCA_{i}" for i in range(pca.n_components_)]
    training_pca_df = pd.DataFrame(training_pca, columns=pca_column_names)
    validation_pca_df = pd.DataFrame(validation_pca, columns=pca_column_names)

    # Append non-numerical columns back to the DataFrames
    non_numerical_columns = [c for c in training.columns if c not in numerical_columns]
    for col in non_numerical_columns:
        training_pca_df[col] = training[col]
        validation_pca_df[col] = validation[col]

    return training_pca_df, validation_pca_df


def find_closest_label(df: pd.DataFrame, label: str) -> str:
    """Find the closest label using Levenshtein distance.

    Parameters
    ----------
    df
        DataFrame containing the data.
    label
        The label to find the closest match for.

    Returns
    -------
        The closest label based on Levenshtein distance.
    """
    label_counts = df["label"].value_counts().to_dict()
    distances = {
        other_label: Levenshtein.distance(label, other_label)
        for other_label in label_counts
        if label_counts[other_label] > 1
    }
    return min(distances, key=distances.get)  # type: ignore


def find_most_correlated_label(df: pd.DataFrame, single_label: str) -> str:
    """Find the most correlated label.

    Parameters
    ----------
    df
        DataFrame containing the data.
    single_label
        The label to find the most correlated match for.

    Returns
    -------
        The most correlated label based on feature correlations.
    """
    label_counts = df["label"].value_counts().to_dict()
    single_label_data = df[df["label"] == single_label].drop(columns=["patient_id", "target", "label"])
    correlations = {}

    for label in label_counts:
        if label_counts[label] > 1:
            label_data = df[df["label"] == label].drop(columns=["patient_id", "target", "label"])
            correlation = single_label_data.corrwith(label_data.mean())
            correlations[label] = correlation.mean()

    return max(correlations, key=correlations.get)  # type: ignore


def reassign_labels(df: pd.DataFrame, threshold: int = 1, distance_metric: str = "levenshtein") -> pd.DataFrame:
    """Reassign labels based on the chosen distance metric.

    Parameters
    ----------
    df
        DataFrame containing the data.
    threshold, optional
        Minimum count threshold for labels to be reassigned, by default 1.
    distance_metric, optional
        The metric to use for reassigning labels ('levenshtein' or 'correlation'), by default "levenshtein".

    Returns
    -------
        DataFrame with reassigned labels.

    Raises
    ------
    ValueError
        If distance_metric not "levenshtein" or "correlation".
    """
    if distance_metric not in ["levenshtein", "correlation"]:
        raise ValueError("distance_metric must be either 'levenshtein' or 'correlation'.")

    label_counts = df["label"].value_counts().to_dict()
    reassigned_labels = {}
    changes = []

    for label in df["label"].unique():
        if label_counts[label] <= threshold:
            if distance_metric == "levenshtein":
                closest_label = find_closest_label(df, label)
                reassigned_labels[label] = closest_label
                changes.append((label, closest_label))
            elif distance_metric == "correlation":
                most_correlated_label = find_most_correlated_label(df, label)
                reassigned_labels[label] = most_correlated_label
                changes.append((label, most_correlated_label))
        else:
            reassigned_labels[label] = label

    if not changes:
        return df

    # Print the changes
    logger.info("Changes made to the labels to avoid problems during CV:")
    logger.info(f"Using {distance_metric} distance to reasign the labels")
    for old_label, new_label in changes:
        label_patient_id = df[df["label"] == old_label].patient_id.iloc[0]
        logger.info(f"For patient_id {label_patient_id}, label changed from {old_label} -> {new_label}")

    old_vc = df["label"].value_counts()
    df["label"] = df["label"].map(reassigned_labels)
    new_vc = df["label"].value_counts()
    combined_vc = pd.concat([old_vc, new_vc], axis=1).round(0)
    combined_vc.columns = ["old count", "new count"]
    logger.info(combined_vc)

    return df


def select_columns_containing_string(df: pd.DataFrame, df_cohort: pd.DataFrame | None, substring: str) -> pd.DataFrame:
    """Selects columns from a dataframe that are categorical and contain a given substring.

    Parameters
    ----------
    df
        The training or validation dataframe with label and target.
    df_cohort
        The dataframe with label cohort information and metadata.
    substring
        The substring to search for within the categorical columns.

    Returns
    -------
        A dataframe with the selected columns and modified label
    """
    # Ensure only categorical columns are considered
    if df_cohort is None:
        return df
    df_cohort.columns = df_cohort.columns.str.lower()
    categorical_columns = df_cohort.select_dtypes(include="object")

    # Filter columns containing the given substring o
    selected_columns = categorical_columns.apply(
        lambda col: col.str.contains(substring.replace("\\ ", ""), case=False)
    ).any()
    selected_columns["label"] = False
    selected_columns["patient_id"] = True

    # Get the column names where the substring was found
    columns_with_substring = selected_columns[selected_columns].index

    # Merge a dataframe with the selected columns and updated label
    df_merged = pd.merge(df, df_cohort[columns_with_substring], how="left", on="patient_id")  # type: ignore
    for col in columns_with_substring:
        if col not in ["patient_id"]:
            df["label"] = df["label"] + " @ " + df_merged[col].fillna("None")

    return df


def load_data(
    config: TomlParser,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, dict[str, list[Any]]]:
    """Load data based on configurations dictionary.

    Parameters
    ----------
    config
        Configuraitons which control which data to load.

    Returns
    -------
        Loaded and checked data.
    """
    logger.info("Loading data from options.")

    # Determine whether to use local or S3 based on explicit CLI arguments
    # Use local if input_path is specified (not None)
    use_local = config.cli_data.input_path is not None

    if use_local:
        assert config.cli_data.input_path is not None  # Type assertion for mypy
        logger.info(f"Loading data from local directory: {config.cli_data.input_path}")
        data, other_features = extract_local_data(directory=config.cli_data.input_path)
    else:
        assert (
            config.cli_data.s3_bucket is not None
            and config.cli_data.s3_input_path is not None
            and config.cli_data.s3_labelled_cohort_key is not None
        )  # Type assertion for mypy
        logger.info(f"Loading data from S3: s3://{config.cli_data.s3_bucket}/{config.cli_data.s3_input_path}")
        data, other_features = extract_data(
            bucket=config.cli_data.s3_bucket,
            directory=config.cli_data.s3_input_path,
            labelled_cohort_key=config.cli_data.s3_labelled_cohort_key,
        )

    logger.info("Checking data in the correct format.")
    labelled_cohort, training, validation = check_data(data)

    # Check on label distribution
    sub_cohort = config.data.sub_cohort
    logger.info(f"This is the initial count of training {training.label.value_counts()}")
    logger.info(f"This is the initial count of validation {validation.label.value_counts()}")

    # Performs sub cohort selection if specified
    if sub_cohort != "all":
        training = select_columns_containing_string(training, labelled_cohort, sub_cohort)
        validation = select_columns_containing_string(validation, labelled_cohort, sub_cohort)

        logger.info("Looking at cohort of data including " + sub_cohort)
        training = training[training.label.str.contains(sub_cohort, case=False)].reset_index(drop=True)
        validation = validation[validation.label.str.contains(sub_cohort, case=False)].reset_index(drop=True)

        logger.info(f"This is the count of selected training {training.label.value_counts()}")
        logger.info(f"This is the count of selected validation {validation.label.value_counts()}")

    # Standardize the data
    if config.data.scale_transform:
        sc = StandardScaler()
        numerical_columns = [c for c in training.select_dtypes(include="number").columns if c not in ["target"]]
        training.loc[:, numerical_columns] = sc.fit_transform(training.loc[:, numerical_columns])
        validation.loc[:, numerical_columns] = sc.transform(validation.loc[:, numerical_columns])  # type: ignore

    if config.data.pca:
        training, validation = apply_pca(training, validation)

    if config.cli_data.use_full_data:  # if don't use full data
        training, validation = pd.concat([training, validation]).reset_index(drop=True), None

    return labelled_cohort, training, validation, other_features


def load_pipeline_file(config: TomlParser, pipeline_path: str) -> Pipeline:
    """Load a pipeline file from either S3 or local filesystem based on configuration.

    Parameters
    ----------
    config
        Configuration containing S3 bucket info and paths.
    pipeline_path
        Path to the pipeline file (relative to S3 bucket or local directory).

    Returns
    -------
    Pipeline
        The loaded sklearn Pipeline object.

    Raises
    ------
    FileNotFoundError
        If the pipeline file cannot be found in either local filesystem or S3.
    """
    from pathlib import Path

    # Determine whether to use local or S3 loading based on explicit CLI arguments
    use_local = config.cli_data.input_path is not None

    if use_local:
        assert config.cli_data.input_path is not None  # Type assertion for mypy
        # Load from local filesystem
        local_path = Path(pipeline_path)
        if not local_path.is_absolute():
            # If relative path, make it relative to input_path
            local_path = config.cli_data.input_path / pipeline_path

        if not local_path.exists():
            raise FileNotFoundError(f"Pipeline file not found at local path: {local_path}")

        logger.info(f"Loading pipeline from local file: {local_path}")
        with open(local_path, "rb") as f:
            return cast(Pipeline, pickle.load(f))
    else:
        assert config.cli_data.s3_bucket is not None  # Type assertion for mypy
        # Load from S3
        logger.info(f"Loading pipeline from S3: s3://{config.cli_data.s3_bucket}/{pipeline_path}")
        pipeline = extract_s3(config.cli_data.s3_bucket, pipeline_path, folder=False)
        if pipeline is None:
            raise FileNotFoundError(f"Pipeline file not found in S3: s3://{config.cli_data.s3_bucket}/{pipeline_path}")
        return cast(Pipeline, pipeline)


class S3Save(TypedDict):
    """Keyword parameters passed to `save_memory_to_s3`"""

    include_metadata: NotRequired[bool]
    model_subfolder: NotRequired[str]


def save_deliverables(
    config: TomlParser,
    csv_dict: Mapping[str, Pipeline | pd.DataFrame | pd.Series[float] | None] = {},
    pickled_dict: MutableMapping[str, Pipeline] = {},
    figure_dict: MutableMapping[str, io.BytesIO] = {},
    log_dict: Mapping[str, io.StringIO] = {},
    **kwargs: Unpack[S3Save],
) -> None:  # pragma: no cover
    """Save all types of deliverables.

    These can be passed in as a dictionary of pandas dataframes,
    figures, or objects.

    Parameters
    ----------
    config
        Toml and CLI configurations.
    csv_dict, optional
        Dictionary of CSV or Pipeline objects, by default {}.
    pickled_dict, optional
        Dictionary of Pickled objects, by default {}.
    figure_dict, optional
        Dictionary of Figures Bytes objects, by default {}.
    log_dict, optional
        Log from running pipeline, by default = {}.
    **kwargs, optional
        Extra arguments to pass to `save_memory_to_s3` or `save_memory_to_local`.
    """
    objects: list[tuple[io.StringIO | io.BytesIO, str]] = []

    # Extract model_id based on whether we're using local or S3 mode
    use_local = config.cli_data.input_path is not None
    if use_local:
        # For local mode, extract from output_path
        assert config.cli_data.output_path is not None, "output_path must be set for local operations"
        model_id = str(config.cli_data.output_path).split("/")[-1]  # Get the unique_id part
    else:
        # For S3 mode, extract from s3_output_path
        assert config.cli_data.s3_output_path is not None, "s3_output_path must be set for S3 operations"
        model_id = str(config.cli_data.s3_output_path).split("/")[1]

    if csv_dict:
        for csv in csv_dict:
            buffer = io.StringIO()
            csv_object = csv_dict[csv]
            if csv_object is None:
                continue

            if isinstance(csv_object, Pipeline):
                # Save the model in a readable format as well as pickled
                params = csv_object.get_params()
                formatted_params = []
                for p in params:
                    formatted_params.append([p, params[p]])

                csv_object = pd.DataFrame(formatted_params)
            csv_object.to_csv(buffer, index=False)
            objects.append((buffer, f"{model_id}_{csv}.csv"))

    if pickled_dict:
        for pickled in pickled_dict:
            bitsbuffer = io.BytesIO()
            pickle.dump(pickled_dict[pickled], bitsbuffer)
            objects.append((bitsbuffer, f"{model_id}_{pickled}.pkl"))

    if figure_dict:
        for figure in figure_dict:
            ext = "html" if "html" in figure else "png"
            objects.append((figure_dict[figure], f"{model_id}_{figure}.{ext}"))

    if log_dict:
        for log in log_dict:
            objects.append((log_dict[log], f"{model_id}_{log}.txt"))

    if use_local:
        save_memory_to_local(objects, config, **kwargs)
    else:
        save_memory_to_s3(objects, config, **kwargs)


def save_memory_to_s3(
    data: list[tuple[io.StringIO | io.BytesIO, str]],
    config: TomlParser,
    model_subfolder: str = "",
    include_metadata: bool = False,
) -> None:
    """Upload data to the S3 bucket.

    Parameters
    ----------
    data
        The data as a buffer and file name to save.
    config
        The full config options.
    model_subfolder, optional
        The subfolder for data to be saved to. Usually one of "imputed",
        "batch_corrected", "validation" or "holdout", by default "".
    include_metadata, optional
        Flag to upload the metadata file, by default False.
    """
    if config.cli_data.dry_run:
        return

    assert (
        config.cli_data.s3_bucket is not None and config.cli_data.s3_input_path is not None
    )  # Type assertion for mypy
    client = boto3.client("s3")
    response_datas = []
    for obj in data:
        assert config.cli_data.s3_input_path is not None, "s3_input_path must be set for S3 operations"
        assert config.cli_data.s3_output_path is not None, "s3_output_path must be set for S3 operations"
        response = client.put_object(
            Bucket=config.cli_data.s3_bucket,
            Key=str(config.cli_data.s3_input_path / config.cli_data.s3_output_path / model_subfolder / obj[1]),
            Body=obj[0].getvalue(),
            StorageClass="INTELLIGENT_TIERING",
        )
        response_datas.append((response, obj[1]))
        logger.info(f"Saved {obj[1]} to bucket {config.cli_data.s3_bucket}")

    if include_metadata:
        meta = MetaData(response_datas, config=config)
        meta_data = meta.create()
        meta.upload(meta_data, client)


def save_memory_to_local(
    data: list[tuple[io.StringIO | io.BytesIO, str]],
    config: TomlParser,
    model_subfolder: str = "",
    include_metadata: bool = False,
) -> None:
    """Save data to local filesystem.

    Parameters
    ----------
    data
        The data as a buffer and file name to save.
    config
        The full config options.
    model_subfolder, optional
        The subfolder for data to be saved to. Usually one of "imputed",
        "batch_corrected", "validation" or "holdout", by default "".
    include_metadata, optional
        Flag to save the metadata file, by default False.
    """
    if config.cli_data.dry_run:
        logger.info("Dry run mode: skipping local file saves")
        return

    # Create output directory structure
    assert config.cli_data.output_path is not None, "output_path must be set for local operations"
    if model_subfolder:
        output_dir = config.cli_data.output_path / model_subfolder
    else:
        output_dir = config.cli_data.output_path

    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    for obj in data:
        file_path = output_dir / obj[1]

        if isinstance(obj[0], io.StringIO):
            # Save text data
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(obj[0].getvalue())
        elif isinstance(obj[0], io.BytesIO):
            # Save binary data
            with open(file_path, "wb") as f:
                f.write(obj[0].getvalue())

        saved_files.append(({"Path": str(file_path)}, obj[1]))
        logger.info(f"Saved {obj[1]} to {file_path}")

    if include_metadata:
        meta = LocalMetaData(saved_files, config=config)
        meta_data = meta.create()
        meta.save(meta_data)


class MetaDataProtocol(Protocol):
    """Protocol for metadata handling."""

    responses: list[tuple[dict[str, Any], str]]
    config: TomlParser
    current_user: str
    timestamp_id: str

    def get_current_git_commit(self) -> str:
        """Gets the current git commit SHA."""
        ...

    def get_file_info(self) -> str:
        """Gets the file information specific to the storage type."""
        ...

    def get_storage_path(self) -> str:
        """Gets the storage path."""
        ...

    def get_timestamp(self) -> str:
        """Gets the timestamp for the metadata."""
        ...

    def create(self) -> str:
        """Creates the metadata file content."""
        ...


class MetaData:
    """Create and upload a metadata file to S3.

    Parameters
    ----------
    responses
        A list of S3 response objects from successful upload.
    config
        The full config options.

    Attributes
    ----------
    current_user
        The current user ID extracted from the S3 output path.
    timestamp_id
        The timestamp ID extracted from the S3 output path.
    s3_path
        The full S3 path where files are stored.
    """

    def __init__(self, responses: list[tuple[dict[str, Any], str]], config: TomlParser) -> None:
        self.responses = responses
        self.config = config
        assert config.cli_data.s3_output_path is not None, "s3_output_path must be set for S3 operations"
        self.current_user, self.timestamp_id = config.cli_data.s3_output_path.parts[:2]
        self.s3_path = (
            f"s3://{config.cli_data.s3_bucket}/{config.cli_data.s3_input_path}/{config.cli_data.s3_output_path}/"
        )

    def get_current_git_commit(self) -> str:
        """Gets the current git commit SHA.

        Returns
        -------
            The git commit at HEAD.
        """
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "Git not available or not a git repository"

    def get_file_info(self) -> str:
        """Gets the versions for each filename.

        Returns
        -------
            Formatted filename versions.
        """
        versions = "Versions:\n"

        for response in self.responses:
            resp, filename = response
            versions += f"{filename.split('.')[0]} | {resp['VersionId']}\n"

        return versions

    def get_storage_path(self) -> str:
        """Gets the S3 storage path.

        Returns
        -------
            The S3 path as a string.
        """
        return self.s3_path

    def get_timestamp(self) -> str:
        """Gets the timestamp from S3 response.

        Returns
        -------
            The timestamp from the S3 response.
        """
        return self.responses[0][0]["ResponseMetadata"]["HTTPHeaders"]["date"]

    def create(self) -> str:
        """Creates the metadata file content.

        Returns
        -------
            The data for the metadata file.
        """
        commit = self.get_current_git_commit()
        timestamp = self.get_timestamp()
        storage_path = self.get_storage_path()

        data = (
            f"Unique ID={self.timestamp_id}\n"
            f"User ID={self.current_user}\n"
            f"Commit={commit}\n"
            f"Timestamp={timestamp}\n"
            f"Storage Path={storage_path}\n"
            "\n"
            f"{self.get_file_info()}\n"
            "\n"
            f"Toml Config:\n"
            f"{toml.dumps(self.config.data.to_dict())}\n"
            "\n"
            f"CLI Options:\n"
            f"{self.config.cli_data}\n"
        )

        return data

    def upload(self, data: str, client: Any) -> None:
        """Uploads the meta data file to s3 storage.

        Parameters
        ----------
        data
            The data to upload.
        client
            The s3 client.
        """
        assert (
            self.config.cli_data.s3_bucket is not None
            and self.config.cli_data.s3_input_path is not None
            and self.config.cli_data.s3_output_path is not None
        )  # Type assertion for mypy
        client.put_object(
            Bucket=self.config.cli_data.s3_bucket,
            Key=str(
                self.config.cli_data.s3_input_path
                / self.config.cli_data.s3_output_path
                / f"{self.timestamp_id}_meta_data.txt"
            ),
            Body=data,
            StorageClass="INTELLIGENT_TIERING",
        )


class LocalMetaData:
    """Create a local metadata file.

    Parameters
    ----------
    responses
        A list of local file response objects from successful save.
    config
        The full config options.

    Attributes
    ----------
    current_user
        The current user ID extracted from the output path.
    timestamp_id
        The timestamp ID extracted from the output path.
    local_path
        The local path where files are stored.
    """

    def __init__(self, responses: list[tuple[dict[str, Any], str]], config: TomlParser) -> None:
        self.responses = responses
        self.config = config
        assert config.cli_data.output_path is not None, "output_path must be set for local operations"

        # Extract user and timestamp from the local output path structure: results/{current_user}/{unique_id}/
        path_parts = config.cli_data.output_path.parts
        if len(path_parts) >= 3 and path_parts[0] == "results":
            self.current_user = path_parts[1]
            self.timestamp_id = path_parts[2]
        else:
            # Fallback if path structure is different
            self.current_user = "unknown"
            self.timestamp_id = "unknown"

        self.local_path = config.cli_data.output_path

    def save(self, data: str) -> None:
        """Saves the metadata file to local storage.

        Parameters
        ----------
        data
            The data to save.
        """
        metadata_path = self.local_path / f"{self.timestamp_id}_meta_data.txt"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        with open(metadata_path, "w") as f:
            f.write(data)

        logger.info(f"Metadata saved to: {metadata_path}")

    def get_current_git_commit(self) -> str:
        """Gets the current git commit SHA.

        Returns
        -------
            The git commit at HEAD.
        """
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "Git not available or not a git repository"

    def get_file_info(self) -> str:
        """Gets the paths for each saved filename.

        Returns
        -------
            Formatted filename paths.
        """
        paths = "Saved Files:\n"

        for response in self.responses:
            resp, filename = response
            paths += f"{filename.split('.')[0]} | {resp['Path']}\n"

        return paths

    def get_storage_path(self) -> str:
        """Gets the local storage path.

        Returns
        -------
            The local path as a string.
        """
        return str(self.local_path)

    def get_timestamp(self) -> str:
        """Gets the current timestamp.

        Returns
        -------
            The current timestamp in ISO format.
        """
        import datetime

        return datetime.datetime.now().isoformat()

    def create(self) -> str:
        """Creates the metadata file content.

        Returns
        -------
            The data for the metadata file.
        """
        commit = self.get_current_git_commit()
        timestamp = self.get_timestamp()
        storage_path = self.get_storage_path()

        data = (
            f"Unique ID={self.timestamp_id}\n"
            f"User ID={self.current_user}\n"
            f"Commit={commit}\n"
            f"Timestamp={timestamp}\n"
            f"Storage Path={storage_path}\n"
            "\n"
            f"{self.get_file_info()}\n"
            "\n"
            f"Toml Config:\n"
            f"{toml.dumps(self.config.data.to_dict())}\n"
            "\n"
            f"CLI Options:\n"
            f"{self.config.cli_data}\n"
        )

        return data
