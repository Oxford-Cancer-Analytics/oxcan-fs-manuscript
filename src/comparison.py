from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from src.data_loader import extract_s3


class ModelComparison:
    """A class to compare biomarkers across different models.

    Parameters
    ----------
    bucket, optional
        The S3 bucket (for S3 mode), by default "".
    file_paths, optional
        A list of file paths to retrieve from S3 or local filesystem, by default None.
    use_local, optional
        Whether to use local filesystem instead of S3, by default False.
    local_base_path, optional
        Base path for local files when use_local=True, by default "results".

    Attributes
    ----------
    dataframes
        List of loaded dataframes from the specified file paths.
    sig_figs
        List of significant figures for each model.
    model_names
        List of extracted model names from file paths.
    """

    def __init__(
        self,
        bucket: str = "",
        file_paths: list[str] | None = None,
        use_local: bool = False,
        local_base_path: str | Path = "results",
    ) -> None:
        if file_paths is None:
            file_paths = []

        self.bucket = bucket
        self.file_paths = file_paths
        self.use_local = use_local
        self.local_base_path = Path(local_base_path)

        self.dataframes = self._extract_data()

        self.sig_figs: list[int] = []
        self.model_names: list[str] = []

    def get_significant_features(self) -> list[pd.Series[float]]:
        """Extracts the significant data series.

        Returns
        -------
            All significant data values in a series.
        """
        sig_series = []
        sig_figs = []

        for frame in self.dataframes:
            data = frame.loc[frame.p_value < 0.05]
            n_features_to_select = data.n_features_to_select.to_numpy()[0]
            assert n_features_to_select is not None
            sig_figs.append(int(n_features_to_select))

            (features_chosen,) = data.loc[data.n_features_to_select == n_features_to_select].features_chosen

            name = "features_importances"
            significant_features = pd.Series(features_chosen[name], name=name).sort_values(ascending=False)
            sig_series.append(significant_features)

        self.sig_figs = sig_figs
        return sig_series

    def compare(self, data: list[pd.Series[float]] | None = None) -> pd.DataFrame:
        """Compares each set of features.

        Parameters
        ----------
        data, optional
            The series of data for each model, by default None.

        Returns
        -------
            A comparison with average feature importance across models.
        """
        if data is None:
            data = self.get_significant_features()

        total_unique_features = set()
        for feature_frame, sig_figs in zip(data, self.sig_figs):
            top_features_series = feature_frame[:sig_figs]

            for feature in top_features_series.index.to_numpy():
                total_unique_features.add(feature)

        total_features = list(sorted(total_unique_features))
        dataframe = self._compare_features(cast(list[str], total_features), data)

        return dataframe

    def _compare_features(self, features: list[str], series_data: list[pd.Series[float]]) -> pd.DataFrame:
        zeros = np.zeros(shape=(len(features), len(self.model_names)))
        dataframe = pd.DataFrame(data=zeros, columns=self.model_names, index=features)

        for feature in features:
            for model_name, data in zip(self.model_names, series_data):
                try:
                    dataframe.loc[feature, model_name] = data.loc[feature]
                except KeyError:
                    continue

        dataframe["avg"] = dataframe.mean(axis=1)
        dataframe = dataframe.sort_values(by="avg", ascending=False)
        return dataframe

    def _extract_data(self) -> list[pd.DataFrame]:
        """Extract data from files using either local filesystem or S3.

        Returns
        -------
            List of dataframes loaded from the specified file paths.

        Raises
        ------
        FileNotFoundError
            If a local file is not found when use_local=True.
        """
        frames = []
        model_names = []
        for path in self.file_paths:
            if self.use_local:
                # Load from local filesystem
                local_path = self.local_base_path / path
                if not local_path.exists():
                    raise FileNotFoundError(f"Local file not found: {local_path}")
                frame = pd.read_csv(local_path, low_memory=False)
            else:
                # Load from S3
                frame = extract_s3(bucket=self.bucket, key=path)

            frames.append(frame)
            model_name = self._extract_model_name(path)
            model_names.append(model_name)

        self.model_names = model_names
        return frames

    def _extract_model_name(self, filename: str) -> str:
        *_, model_name_ext = filename.rpartition("rfe_important_features_")
        model_name = model_name_ext.split(".")[0]

        return model_name
