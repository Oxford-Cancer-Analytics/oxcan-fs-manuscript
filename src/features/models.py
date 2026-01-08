import joblib
import numpy as np
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest

from .._types import np_array


class UnivariateFeature:
    """Univariate model using F-test feature scoring.

    Parameters
    ----------
    n_features_to_select
        The number of features to select.
    n_jobs
        The number of cores to use for parallel processing.
        -1 selects all, -2 selects all but one.
    """

    def __init__(self, n_features_to_select: int, n_jobs: int) -> None:
        self.n_features_to_select = n_features_to_select  # used to mock models
        self.n_jobs = n_jobs  # n_jobs used to mock skrebate models

        self.feature_importances_: list[float]

    def fit(self, X_train: np_array, y_train: np_array) -> None:
        """Select k best using F test.

        Scores are normalized to [a,b] a = -1, b = 1
        normalized = a + ( x - x.min() ) ( b - a ) / ( x.max() - x.min() )

        Parameters
        ----------
        X_train
            Single feature.
        y_train
            Target labels.
        """
        selector = SelectKBest(f_classif, k=self.n_features_to_select)
        with joblib.parallel_backend(n_jobs=self.n_jobs, backend="loky"):
            selector.fit(X_train, y_train)
        scores = -np.log10(np.array(selector.pvalues_))
        scores = 2 * ((scores - scores.min()) / (scores.max() - scores.min())) - 1
        self.feature_importances_ = scores
