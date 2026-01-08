import datetime
import io
import logging
import os
import pwd
import time
from pathlib import Path

import typer
from src.cli.cli_options import AugmentationEnum
from src.cli.cli_options import FeatureSelectionEnum
from src.cli.cli_options import ModelsEnum
from src.cli.cli_options import StatisticsEnum
from src.cli.toml_parser import TomlParser
from src.data_loader import save_deliverables
from src.features.features import Features


current_time = str(datetime.datetime.now()).split(".")[0]
Path("logs").mkdir(exist_ok=True)
app = typer.Typer()
logging.basicConfig(
    filename=f"logs/runtime {current_time}.log",
    filemode="w",
    encoding="utf-8",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s:%(filename)s - %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
)


@app.command()
def main(
    input_path: Path | None = typer.Option(None, help="Path to local input data folder."),
    stats: StatisticsEnum = typer.Option(StatisticsEnum.TTEST_INDEPENDENT, help="Statistical test to use."),
    s3_input_path: Path | None = typer.Option(
        None,
        help="Path to S3 input data.",
    ),
    s3_bucket: str | None = typer.Option(None, help="S3 bucket name."),
    s3_labelled_cohort_key: Path | None = typer.Option(None, help="S3 key for labelled cohort file."),
    imputation_strategy: str = typer.Option("mar", help="Imputation strategy to use."),
    use_batch_corrected: bool = typer.Option(False, help="Whether to use batch corrected data."),
    model: ModelsEnum = typer.Option(ModelsEnum.XGBOOST, help="Model to use."),
    feature_selection: FeatureSelectionEnum = typer.Option(
        FeatureSelectionEnum.MUTUAL_INFORMATION, help="Feature selection method to use."
    ),
    augmentation: AugmentationEnum = typer.Option(AugmentationEnum.NONE, help="Data augmentation method to use."),
    dry_run: bool = typer.Option(False, help="Whether to perform a dry run."),
    rec_feat_add: bool = typer.Option(True, help="Whether to use recursive feature addition."),
    use_full_data: bool = typer.Option(False, help="Whether to use the full dataset."),
    prepare_data: bool = typer.Option(False, help="Whether to prepare the data."),
    best_pipeline: Path = typer.Option(Path(""), help="Path to the best pipeline."),
) -> None:
    """All CLI options specified will override defaults.

    Either --input-path must be specified for local mode OR both --s3-bucket and --s3-input-path must be specified
    for S3 mode. All other options are optional and specify what the pipeline should do.

    If `--prepare-data` is True, then data pre-processing combinations (scaling and imputation only) are run to
    find the best combination for the data specified.

    If `--prepare-data` is False and `--best-pipeline` is specified, then the feature
    selection pipeline will run using this chosen pipeline. This includes the optimal
    preprocessing steps and a default `model`. Data will be imputed using this pipeline
    which will be used for the downstream ML analysis and data collection.

    If `--prepare-data` is False and `--best-pipeline` is the default Path(""), then the
    data should already be imputed as the pipeline will optimise the
    `--model` for feature selection.
    """
    cli_configs = locals().copy()

    # Validate that either local OR S3 paths are specified, but not both or neither
    local_specified = input_path is not None
    s3_specified = s3_bucket is not None and s3_input_path is not None

    if not local_specified and not s3_specified:
        typer.echo(
            "Error: You must specify either --input-path for local mode OR both --s3-bucket and --s3-input-path for S3 mode."
        )
        raise typer.Exit(1)

    if local_specified and s3_specified:
        typer.echo(
            "Error: You cannot specify both local (--input-path) and S3 (--s3-bucket, --s3-input-path) options. Choose one mode."
        )
        raise typer.Exit(1)

    # Set defaults for unspecified paths to None for proper dataclass handling
    if not local_specified:
        input_path = None  # Keep as None for CliConfigs

    if not s3_specified:
        s3_bucket = None  # Keep as None for CliConfigs
        s3_input_path = None  # Keep as None for CliConfigs

    logging.info("Getting toml and CLI configurations.")

    unique_id = str(int(time.time()))
    current_user = pwd.getpwuid(os.getuid())[0]

    if s3_specified:
        if not s3_labelled_cohort_key:
            typer.echo(
                "Error: When using S3 mode, you must specify --s3-labelled-cohort-key for the labelled cohort file."
            )
            raise typer.Exit(1)

        # Set S3 output path for S3 runs
        cli_configs["s3_output_path"] = Path(f"{current_user}/{unique_id}/")
        cli_configs["output_path"] = None
        logging.info(f"S3 Output Path: {cli_configs['s3_input_path']}/{cli_configs['s3_output_path']}")
    else:
        # Set local output path with same structure for local runs
        cli_configs["output_path"] = Path(f"{cli_configs['input_path']}/results/{current_user}/{unique_id}/")
        cli_configs["s3_output_path"] = None
        logging.info(f"Local Output Path: {cli_configs['output_path']}")

    logging.info("Getting toml and CLI configurations.")

    config = TomlParser(Path("config.toml")).read()  # toml configs
    config.add_cli_config(cli_configs)  # add cli configs

    Features(config).run_pipeline()

    log = open(f"logs/runtime {current_time}.log").read()
    save_deliverables(
        config,
        log_dict={"log": io.StringIO(log)},
    )

    logging.info("oxcan-fs completed.")


if __name__ == "__main__":
    try:
        app()
    except Exception:
        import traceback

        logging.error(traceback.format_exc())
        raise
