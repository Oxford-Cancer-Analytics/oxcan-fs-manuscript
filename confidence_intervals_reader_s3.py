import io

import boto3
import pandas as pd

# AWS S3 config
bucket_name = "pm-lc-toronto-22"

user_prefix = (
    "FSv4.0.1/HRF_202407311100_medcent_min100_coverage70_top14removed_oe480/hrferguson/1722429660/imputed/luke/"
)
# user_prefix = "FSv4.0.1/HRF_202407171320_pmlct22_preprocessed_olink_x600/luke/"

keywords = ["Boots", "_all_"]
# keywords = ["Boots"]
# keywords = ["_CV_"]

# Setup S3 client and paginator
s3 = boto3.client("s3")
paginator = s3.get_paginator("list_objects_v2")

# Step 1: List all model folders under luke/
response = s3.list_objects_v2(Bucket=bucket_name, Prefix=user_prefix, Delimiter="/")
model_folders = [cp["Prefix"] for cp in response.get("CommonPrefixes", [])]

# Step 2: Scan for matching CSVs under holdout/** for each model
dataframes = []

for model_prefix in model_folders:
    # holdout_prefix = model_prefix + "holdout/"
    holdout_prefix = model_prefix + "validation/semi_holdout/"

    for page in paginator.paginate(Bucket=bucket_name, Prefix=holdout_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            filename = key.split("/")[-1]

            if all(k in filename for k in keywords) and filename.endswith(".csv"):
                print(f"Reading: {key}")

                try:
                    obj = s3.get_object(Bucket=bucket_name, Key=key)
                    df = pd.read_csv(io.BytesIO(obj["Body"].read()))
                    df["source_s3_key"] = key  # Add trace column
                    dataframes.append(df)
                except Exception as e:
                    print(f"Failed to read {key}: {e}")

# Step 3: Combine and save
if dataframes:
    print(f"number of dataframes found: {len(dataframes)}")
    final_df = pd.concat(dataframes, ignore_index=True)
    output_csv = "CV_all_semiholdout_combined_olink_all_cohorts.csv"
    final_df.to_csv(output_csv, index=False)
    print(f"Saved combined data to: {output_csv}")
else:
    print("No matching files found or readable.")
