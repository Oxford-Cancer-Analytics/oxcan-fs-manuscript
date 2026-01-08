# oxcan-fs

Feature Selection Pipeline: For analyzing DIA mass spectrometry data to find the best protein signal candidates.

## Overview

This pipeline performs comprehensive feature selection and machine learning analysis on proteomics data to identify optimal protein biomarkers. It supports various feature selection methods, machine learning models, and data augmentation techniques.

## Installation

Using Python 3.10, create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

The pipeline can be run in two modes:

### 1. Single Pipeline Run (`main.py`)

Run the feature selection pipeline with specific parameters:

```bash
python main.py [OPTIONS]
```

#### Basic Examples

**Run with local data:**
```bash
python main.py --input-path /path/to/your/data
```

**Data preparation mode:**
```bash
python main.py --prepare-data --input-path /path/to/your/data
```

**Use specific model and feature selection:**
```bash
python main.py --input-path /path/to/your/data --model xgboost --feature-selection MutualInformation --augmentation smote_tomek
```

**Dry run (no file saved for debugging):**
```bash
python main.py --input-path /path/to/your/data --dry-run
```

**Enterprise S3 data source:**
```bash
python main.py --s3-bucket BUCKET --s3-input-path your/s3/data/path/ --s3-labelled-cohort-key your/s3/key/path.csv
```

#### Command Line Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--input-path` | Path | None | Path to local input data folder containing CSV files (required for local mode) |
| `--stats` | StatisticsEnum | `ttest_ind` | Statistical test to use |
| `--s3-input-path` | Path | None | Path to S3 input data (required for S3 mode) |
| `--s3-bucket` | str | None | S3 bucket name (required for S3 mode) |
| `--s3-labelled-cohort-key` | Path | None | S3 key for labelled cohort file (required for S3 mode) |
| `--imputation-strategy` | str | `mar` | Imputation strategy (`mar`, `mnar`, `mixed`) |
| `--use-batch-corrected` | bool | `False` | Whether to use batch corrected data |
| `--model` | ModelsEnum | `xgboost` | Machine learning model to use |
| `--feature-selection` | FeatureSelectionEnum | `MutualInformation` | Feature selection method |
| `--augmentation` | AugmentationEnum | `none` | Data augmentation method |
| `--dry-run` | bool | `False` | Run without saving results |
| `--rec-feat-add` | bool | `True` | Enable recursive feature addition |
| `--use-full-data` | bool | `False` | Use full dataset instead of subset |
| `--prepare-data` | bool | `False` | Run preprocessing combinations to find optimal pipeline |
| `--best-pipeline` | Path | `""` | Path to best pipeline configuration (S3 mode only) |

> **Note**: You must specify either `--input-path` for local mode OR both `--s3-bucket` and `--s3-input-path` for S3 mode. Output paths are automatically generated based on timestamp and user information.

#### Available Options

**Models (`--model`):**
- `xgboost` - XGBoost Classifier (default)
- `balanced_random_forest` - Balanced Random Forest Classifier
- `random_forest` - Random Forest Classifier

**Feature Selection Methods (`--feature-selection`):**
- `MutualInformation` - Mutual Information (default)
- `BorutaShap` - Boruta SHAP feature selection
- `MultiSURF` - MultiSURF algorithm
- `UnivariateFeature` - Univariate feature selection

**Data Augmentation (`--augmentation`):**
- `none` - No augmentation (default)
- `smote_tomek` - SMOTE + Tomek links
- `smote_ratio` - SMOTE ratio adjustment
- `smote_een` - SMOTE + Edited Nearest Neighbors
- `smote_balanced` - SMOTE balanced sampling
- `kde_balanced` - KDE balanced sampling

**Statistical Tests (`--stats`):**
- `ttest_ind` - Independent t-test (default)

**Imputation Strategies (`--imputation-strategy`):**
- `mar` - Missing At Random (default)
- `mnar` - Missing Not At Random
- `mixed` - Mixed strategy

### 2. Matrix Pipeline Run (`pipeline_runner.py`)

Run multiple pipeline configurations from an Excel file:

```bash
python pipeline_runner.py
```

This script reads configurations from `pipeline_runs.xlsx` and runs the pipeline with different parameter combinations. It:

1. Reads pipeline configurations from `pipeline_runs.xlsx`
2. Updates `config.toml` with configuration-specific parameters
3. Updates and runs `main.py` with CLI-specific parameters
4. Processes each row in the Excel file as a separate pipeline run
5. If the pipeline finishes or is interrupted, updated files are restored

The Excel file should contain columns with prefixes:
- `config_*` - Parameters that update the TOML configuration file
- `cli_*` - Parameters passed as command-line arguments

## Data Requirements

The pipeline supports both local files (default) and S3-based data input for enterprise deployments.

### Local Data Input (Default)

The pipeline expects the following files in your input directory:

#### Required Data Files
- `train.csv` - Training dataset with features and labels
- `holdout.csv` - Validation/test dataset with features and labels
- `patient_metadata.csv` (optional) - Patient metadata (will be auto-generated if missing)

#### Data Format Requirements
Each CSV file must contain:
- `patient_id` - Unique patient identifier
- `target` - Binary target variable (0/1)
- `label` - Text labels for the conditions
- Feature columns (protein measurements, etc.)

#### Local Directory Structure Example
```
your-data/
├── train.csv
├── holdout.csv
└── labelled_cohort.csv (optional)
```

### S3 Data Input (Enterprise)

For enterprise deployments, the pipeline can load data from S3:

```bash
python main.py --s3-bucket your-bucket --s3-input-path your/s3/data/path/ --s3-labelled-cohort-key your/s3/key/path.csv
```

#### S3 Structure Example
```
s3://your-bucket/your-path/
├── train.csv
├── holdout.csv
└── (patient metadata loaded automatically)
```

## Configuration

The pipeline uses both TOML configuration (`config.toml`) and command-line arguments.

### Key Configuration Sections

**General Settings:**
- `random_state` - Random seed for reproducibility (default: 0)
- `sub_cohort` - Cohort filtering (default: 'all', options: 'all', regex pattern)
- `scale_transform` - Apply scaling transformation (default: True)
- `pca` - Apply PCA transformation (default: False)

**Statistics (`[stats]`):**
- `pvalue_thresh` - P-value threshold (default: 0.05)
- `power` - Statistical power (default: 0.95)
- `alpha` - Significance level (default: 0.05)

**Feature Selection (`[feature_selection]`):**
- `multisurf_cross_validation_splits` - CV splits for MultiSURF (default: 50)
- `cross_validation_splits` - CV splits for validation (default: 2)
- `cross_validation_repeats` - CV repeats (default: 25)
- `number_of_mi_features` - Features to select during MI (default: 0.035)
- `number_of_features` - Target number of features (default: 30)

**Feature Selection Addition (`[feature_selection.addition]`):**
- `method` - RFA method (default: "feature_importance", options: "feature_importance", "model_performance")
- `importance_selection` - Feature importance method (default: "shap", options: "shap", "eject_shap", "tree_importance", "mutual_information", "permutation_importance")

**Augmentation SMOTE (`[augmentation.smote]`):**
- `new_cancer_ratio` - New ratio of cancer patients (default: 0.5)
- `new_total_samples` - New total patients (default: 0)

**Optimization Optuna (`[optimization.optuna]`):**
- `scorer` - Scoring method for multiobjective optimization (default: ["mcc", "sens@99spec"])
- `trials` - Number of bayesian search iterations (default: 300)
- `before_feature_selection` - Search hyperparameters before feature selection (default: False)
- `after_rfa` - Search hyperparameters after RFA (default: True)
- `before_validation` - Search hyperparameters before validation (default: False)

## Pipeline Modes

### Data Preparation Mode (`--prepare-data`)

When `--prepare-data` is True:
- Runs preprocessing combinations (scaling and imputation)
- Finds optimal preprocessing steps for the specified S3 data
- **Does not run feature selection** - only preprocessing optimization
- Use this to determine the best preprocessing pipeline before feature selection

### Feature Selection Mode (default)

When `--prepare-data` is False:
- If `--best-pipeline` is specified: Uses the specified preprocessing pipeline from S3
- If `--best-pipeline` is empty: Assumes data is already preprocessed or runs basic preprocessing
- Runs feature selection and model optimization

## Output

### Local Output

Results are automatically saved to a generated directory structure inside the `local_input_path folder`:
- Feature importance rankings
- Model performance metrics
- Cross-validation results
- Model pipelines (pickled)
- Detailed logs in the `logs/` directory

**Local Output Structure:**
```
results/
└── {username}/
    └── {timestamp_id}/
        ├── validation/
        │   ├── {timestamp}_feature_importance.csv
        │   ├── {timestamp}_model_pipeline.pkl
        │   └── ...
        ├── holdout/
        │   ├── {timestamp}_performance_metrics.csv
        │   └── ...
        └── {timestamp}_metadata.txt
```

### S3 Output (Enterprise)

For S3 mode, results are uploaded to S3 (unless `--dry-run` is used):

**S3 Output Structure:**
```
s3://bucket/input-path/username/timestamp/
├── validation/
│   ├── feature_importance.csv
│   ├── model_pipeline.pkl
│   └── ...
├── holdout/
│   ├── performance_metrics.csv
│   └── ...
└── metadata.txt
```


## Examples

**Complete feature selection pipeline with local data:**
```bash
python main.py \
  --input-path /path/to/your/data \
  --model random_forest \
  --feature-selection MultiSURF \
  --augmentation smote_balanced \
  --rec-feat-add \
```

**Data preparation only:**
```bash
python main.py \
  --prepare-data \
  --input-path /path/to/your/data
```

**Dry run (no file saves):**
```bash
python main.py \
  --input-path /path/to/your/data \
  --model xgboost \
  --dry-run
```

**Enterprise S3 pipeline:**
```bash
python main.py \
  --s3-bucket BUCKET \
  --s3-input-path your/s3/data/path/ \
  --model random_forest \
  --feature-selection MultiSURF
```

For help with all available options:
```bash
python main.py --help
```
