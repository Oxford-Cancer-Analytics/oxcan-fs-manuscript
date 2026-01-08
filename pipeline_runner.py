import signal
import subprocess
import sys

import numpy as np
import pandas as pd


def restore_files() -> None:
    """Restore main.py and config.toml from git."""
    for f in ["main.py", "config.toml"]:
        try:
            subprocess.run(["git", "restore", f], check=False)
            print(f"Restored {f} to git version.")
        except Exception as e:
            print(f"Failed to restore {f}: {e}")


def handle_interrupt(sig: int, _frame: object) -> None:  # noqa: U101
    """Handle Ctrl+C / SIGTERM and restore main.py before exit.

    Parameters
    ----------
    sig
        The signal number received.
    _frame
        The current stack frame (unused).
    """
    print(f"\nInterrupted (signal {sig}). Restoring main.py before exit...")
    restore_files()
    sys.exit(1)


# Register signal handlers
signal.signal(signal.SIGINT, handle_interrupt)
signal.signal(signal.SIGTERM, handle_interrupt)


pipeline_runs = pd.read_excel("pipeline_runs.xlsx").convert_dtypes()
config_options = list(filter(lambda x: "config" in x, pipeline_runs.columns))
cli_options = list(filter(lambda x: "cli" in x, pipeline_runs.columns))

print("Available CLI options in Excel:", cli_options)
print("Available config options in Excel:", config_options)

# Validate that we have the necessary input path columns
has_local_option = any("cli_input-path" in col for col in pipeline_runs.columns)
has_s3_options = any("cli_s3-bucket" in col for col in pipeline_runs.columns) and any(
    "cli_s3-input-path" in col for col in pipeline_runs.columns
)

if not has_local_option and not has_s3_options:
    print("Warning: Excel file doesn't contain proper input path columns.")
    print("Expected: 'cli_input-path' for local mode OR both 'cli_s3-bucket' and 'cli_s3-input-path' for S3 mode")
    print("Available columns:", list(pipeline_runs.columns))

for row_num in range(pipeline_runs.shape[0]):
    # Skip if blank row
    if pipeline_runs.loc[row_num].isna().all():
        print("Skipped", row_num)
        continue

    print(f"\nProcessing row {row_num}")

    cli_params = [f"--{param.split('cli_')[1]}" for param in cli_options]
    cli_values: list[str] = [pipeline_runs.loc[row_num, param] for param in cli_options]  # type: ignore

    # Check what type of input is specified in CLI options for this specific row
    s3_bucket_value = next(
        (pipeline_runs.loc[row_num, param] for param in cli_options if param == "cli_s3-bucket"), None
    )
    s3_input_path_value = next(
        (pipeline_runs.loc[row_num, param] for param in cli_options if param == "cli_s3-input-path"), None
    )
    input_path_value = next(
        (pipeline_runs.loc[row_num, param] for param in cli_options if param == "cli_input-path"), None
    )

    # Check if values are actually present (not NaN or empty)
    has_s3_values = (
        s3_bucket_value is not None
        and not pd.isna(s3_bucket_value)
        and str(s3_bucket_value).strip()
        and s3_input_path_value is not None
        and not pd.isna(s3_input_path_value)
        and str(s3_input_path_value).strip()
    )
    has_input_path_value = (
        input_path_value is not None and not pd.isna(input_path_value) and str(input_path_value).strip()
    )

    # Determine mode and validate
    if has_s3_values:
        mode = "s3"
        print(f"Using S3 mode for row {row_num}")
    elif has_input_path_value:
        mode = "local"
        print(f"Using local mode for row {row_num}")
    else:
        # Neither mode properly specified - skip execution and show error
        print(
            f"Error: Row {row_num} - You must specify either --input-path for local "
            + "mode OR both --s3-bucket and --s3-input-path for S3 mode."
        )
        continue

    try:
        # Update main.py output path
        with open("main.py") as file:
            lines = file.readlines()
            new_out_path = pipeline_runs.loc[row_num, "main_output_exension_path"]
            local_input_path = pipeline_runs.loc[row_num, "cli_input-path"]

            if mode == "s3":
                index = next(
                    idx
                    for idx, line in enumerate(lines)
                    if 'cli_configs["s3_output_path"]' in line and "Path(" in line
                )
                out_path = lines[index]
                first, second = out_path.split(" = ")
                second = second.replace(second, f'Path(f"{{current_user}}/{{unique_id}}_{new_out_path}/")\n')
                lines[index] = first + " = " + second
            else:
                index = next(
                    idx for idx, line in enumerate(lines) if 'cli_configs["output_path"]' in line and "Path(" in line
                )
                out_path = lines[index]
                first, second = out_path.split(" = ")
                second = second.replace(
                    second, f'Path(f"{local_input_path}/results/{{current_user}}/{{unique_id}}_{new_out_path}/")\n'
                )
                lines[index] = first + " = " + second

            open("main.py", "w").writelines(lines)

        # Update config.toml
        with open("config.toml") as config:
            lines = config.readlines()

            for config_option in config_options:
                option = config_option.split("config_")[1]
                (index,) = (idx for idx, line in enumerate(lines) if option in line)
                value = pipeline_runs.loc[row_num, config_option]

                if not isinstance(value, (int, float, np.int64, np.float64)):  # type: ignore
                    if isinstance(value, np.bool_):
                        value = f"{str(value).lower()}"
                    elif isinstance(value, str) and value.isdigit():
                        value = int(value)
                    elif isinstance(value, str) and any(val in value for val in ["[", "]"]):
                        value = eval(value)
                    else:
                        value = f"'{value}'"
                lines[index] = f"{option} = {value}\n"

            open("config.toml", "w").writelines(lines)

        # Build the base command
        base_cmd = ["python", "-m", "main"]

        # Boolean CLI options that use --flag/--no-flag syntax
        boolean_cli_options = {
            "--use-batch-corrected",
            "--dry-run",
            "--rec-feat-add",
            "--use-full-data",
            "--prepare-data",
        }

        for param, value in zip(cli_params, cli_values):
            if not pd.isna(value) and str(value).strip():
                # Handle boolean options: convert true/false to --flag/--no-flag
                if param in boolean_cli_options:
                    str_value = str(value).strip().lower()
                    if str_value == "true":
                        base_cmd.append(param)
                    elif str_value == "false":
                        base_cmd.append(f"--no-{param[2:]}")  # Convert --flag to --no-flag
                    # Skip if not a valid boolean string
                else:
                    base_cmd.extend([param, str(value)])

        print(f"Executing command: {' '.join(base_cmd)}")
        subprocess.run(base_cmd)
    finally:
        restore_files()

print("\nAll pipeline runs completed.")
