import itertools
import math
import re
from typing import cast

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from joblib import delayed
from joblib import Parallel
from src._types import np_array


class EjectTree:
    """Eject Trees for Exacy Shapley values.

    Parameters
    ----------
    model
        An XGBClassifier model.

    Attributes
    ----------
    trees_df
        A dataframe of all tree nodes, splits, and values for the model.
    tree_index
        The current index of the tree.
    model_trees
        All model tree attributes.
    """

    def __init__(self, model: xgb.XGBClassifier) -> None:
        if model.__class__.__name__ != "XGBClassifier":
            raise NotImplementedError(f"{model.__class__} has not been implemented yet for EjectShapley values.")

        self.model = model
        self.trees_df = model.get_booster().trees_to_dataframe()
        self.tree_index = 0

        shap_load = shap.explainers._tree.XGBTreeModelLoader(model.get_booster())
        self.model_trees = shap_load.get_trees()

    def feature_path(self, X: np_array) -> list[int]:
        """Get features along a samples decision path.

        Parameters
        ----------
        X
            A single sample.

        Returns
        -------
            Features seen for a particular tree with this sample.
        """
        use_idxs = []

        tree_nodes = self.trees_df
        feature_nodes = tree_nodes.loc[tree_nodes.Tree == self.tree_index]

        next_id = f"{self.tree_index}-0"
        while True:
            current_node = feature_nodes.loc[feature_nodes.ID == next_id]
            (feature,) = current_node.Feature

            if feature == "Leaf":
                break

            if hasattr(self.model, "feature_names_in_"):
                feature = np.where(self.model.feature_names_in_ == feature)[0][0]
            else:
                feat_ints = re.findall(r"(\d)", feature)
                feature = int("".join(feat_ints))
            use_idxs.append(feature)
            (split,) = current_node.Split

            if (X[feature] <= split) or np.isnan(X[feature]).all():
                (next_id,) = current_node.Yes
            else:
                (next_id,) = current_node.No

        return use_idxs

    def predict_eject(self, X: np_array, ftr_idxs: list[int]) -> float:
        """Predicts the previous node value or the leaf value.

        If a feature in the tree is not present in a samples decision path, the tree
        ejects early and returns the previous node value.

        Parameters
        ----------
        X
            A single sample.
        ftr_idxs
            Features seen for a particular tree with this sample.

        Returns
        -------
            The predicted node/leaf value.
        """
        feature_nodes = self.trees_df[self.trees_df.Tree == self.tree_index]

        next_id = f"{self.tree_index}-0"
        while True:
            current_node = feature_nodes[feature_nodes.ID == next_id]
            (feature,) = current_node.Feature

            if feature == "Leaf":
                return self.predict(X)

            if hasattr(self.model, "feature_names_in_"):
                feature = np.where(self.model.feature_names_in_ == feature)[0][0]
            else:
                feat_ints = re.findall(r"(\d)", feature)
                feature = int("".join(feat_ints))

            if feature not in ftr_idxs:
                return self.model_trees[self.tree_index].values[int(next_id.split("-")[1])][0]

            (split,) = current_node.Split
            if (X[feature] <= split) or np.isnan(X[feature]).all():
                (next_id,) = current_node.Yes
            else:
                (next_id,) = current_node.No

    def predict(self, X: np_array) -> float:
        """Predicts the terminal value for a sample.

        Parameters
        ----------
        X
            A single sample.

        Returns
        -------
            The terminal leaf value.
        """
        tree = self.model_trees[self.tree_index]
        cur_idx = 0
        while cur_idx >= 0:
            this_idx = cur_idx

            feat = tree.features[cur_idx]
            threshold = tree.thresholds[cur_idx]
            if X[feat] <= threshold or np.isnan(X[feat]):
                cur_idx = tree.children_left[cur_idx]
            else:
                cur_idx = tree.children_right[cur_idx]

        return tree.values[this_idx][0]

    def shap_values(self, data: pd.DataFrame) -> np_array:
        """Gets Eject Shapley values across all trees, samples, and features.

        Parameters
        ----------
        data
            A data matrix.

        Returns
        -------
            Eject Shapley values.
        """
        with Parallel(n_jobs=-2) as parallel:
            sv_trees = cast(
                list[np_array],
                parallel(
                    delayed(self._shap_values)(tree_index, data) for tree_index in range(self.model.n_estimators)
                ),
            )

        return np.sum(sv_trees, axis=0)

    def _shap_values(self, tree_index: int, data: pd.DataFrame) -> np_array:
        """Internal function to compute Eject Shapley values for a single tree.

        Parameters
        ----------
        tree_index
            The tree to use in the model.
        data
            A data matrix.

        Returns
        -------
            Eject Shapley values for a single tree.
        """
        SVs_ej = np.zeros(data.shape)
        self.tree_index = tree_index
        for idx_sample, sample in enumerate(data.to_numpy()):
            ej_svs = self._eject_shap_features(sample)

            for idx_feature in range(data.shape[1]):
                SVs_ej[idx_sample][idx_feature] += ej_svs[idx_feature]

        return SVs_ej

    def _eject_shap_features(self, X: np_array) -> list[float]:
        """Produces Eject Shapley values for each feature of a sample.

        Parameters
        ----------
        X
            A single sample.

        Returns
        -------
            Eject Shapley values for each feature.
        """
        svs = [0.0 for _ in range(len(X))]
        use_idxs = self.feature_path(X)

        for iftr in use_idxs:
            ftr_idxs = [jftr for jftr in use_idxs if jftr != iftr]
            for isize in range(len(use_idxs)):
                pre_factor = 1.0 / (len(use_idxs) * math.comb(len(use_idxs) - 1, isize))
                for combo in itertools.combinations(ftr_idxs, isize):
                    plus_set = list(combo)
                    plus_set.append(iftr)
                    svs[iftr] += pre_factor * (self.predict_eject(X, plus_set) - self.predict_eject(X, list(combo)))
        return svs


"""
----------------------------------------------------------------------------------------------------------------------------
Copyright (c) 2021, Biodesix, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the organization (Biodesix, Inc) nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL BIODESIX, INC BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
----------------------------------------------------------------------------------------------------------------------------
"""
