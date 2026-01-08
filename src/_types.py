from typing import Literal
from typing import NamedTuple
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier


KDEType = Literal["balanced", "imbalanced_early", "imbalanced_healthy", "none"]


class Frames(NamedTuple):
    """A set of preprocessed data."""

    name: KDEType
    data: DataFrame


DimensionReduction = Literal["ae", "pca", "tsne", "randomtree"]
ClusteringMethod = Literal["hierarchical", "randomforest", "kmeans"]
BaseEstimator: TypeAlias = (  # type: ignore
    RandomForestClassifier | LogisticRegression | MLPClassifier | SVC | DecisionTreeClassifier | XGBClassifier
)
np_array: TypeAlias = NDArray[np.float64]
