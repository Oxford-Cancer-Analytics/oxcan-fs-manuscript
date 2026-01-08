import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from src.data_loader import check_data
from src.data_loader import extract_data
from src.data_loader import extract_s3
from src.data_loader import ExtractedData
from src.data_loader import load_data
from src.data_loader import save_memory_to_s3


def test_data_checks():
    """Data checks X and y are correct size and match lists."""
    X = np.random.random(size=(5, 10))
    y = np.random.binomial(1, 0.5, size=(5,))
    protein_list = [f"prot_{p}" for p in range(10)]
    patient_list = [f"pat_{p}" for p in range(5)]
    full_frame = pd.DataFrame(X, columns=protein_list)
    full_frame["patient_id"] = patient_list
    full_frame["target"] = y
    full_frame["experiment_type"] = full_frame["target"].apply(lambda x: "control" if x == 0 else "NSLC early")
    data = {
        "labelled_cohort": full_frame,
        "train": full_frame.iloc[:3, :],
        "holdout": full_frame.iloc[3:, :],
    }
    # assertions are in function
    output = check_data(ExtractedData(**data))

    assert len(output) == 3
    assert all(isinstance(value, pd.DataFrame) for value in output)


def test_extract_data():
    output = extract_data("test_bucket", Path("test_key/testing"), Path("test_key/label.csv"))

    assert len(output) == 2
    assert all(isinstance(value, pd.DataFrame) for value in output[0].values())
    assert ["train", "holdout", "labelled_cohort"] == list(output[0].keys())


@pytest.mark.parametrize("ext", ("txt", "csv", "parquet"))
def test_extract_s3(ext):
    output = extract_s3("test_bucket", f"test_key.{ext}")

    assert isinstance(output, pd.DataFrame)


@pytest.mark.parametrize(("full_data", "out_type"), ((True, type(None)), (False, pd.DataFrame)))
def test_load_from_options(parser_with_cli_options, full_data, out_type, mock_check_data):
    parser_with_cli_options.cli_data.use_full_data = full_data
    output = load_data(parser_with_cli_options)

    assert len(output) == 4

    _, train, test, _ = output
    assert isinstance(test, out_type)
    assert isinstance(train, pd.DataFrame)


@pytest.mark.parametrize("dry_run", (True, False))
def test_save_memory_to_s3(parser_with_cli_options, get_X_y_data_imputed, dry_run):
    X, y, _, proteins = get_X_y_data_imputed
    df = pd.DataFrame(X, columns=proteins)
    df["target"] = y

    buffer = io.StringIO()
    df.to_csv(buffer, index=False)

    parser_with_cli_options.cli_data.dry_run = dry_run
    output = save_memory_to_s3([(buffer, "test.csv")], parser_with_cli_options, include_metadata=True)

    assert output is None
