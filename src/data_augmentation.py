import logging
from typing import cast

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

from ._types import np_array
from .cli.cli_options import AugmentationEnum

logger = logging.getLogger(__name__)


def optimize_bw(x: np_array) -> KernelDensity:
    """Find best bandwidth for KDE.

    Parameters
    ----------
    x
        Protein abundance values.

    Returns
    -------
        Optimized KDE bandwidth.
    """
    # use grid search cross-validation to optimize the bandwidth
    params = {"bandwidth": np.logspace(-1, 1, 20)}
    grid = GridSearchCV(
        KernelDensity(algorithm="auto", kernel="gaussian", metric="euclidean"), params, cv=min(x.shape[0], 5)
    )
    grid.fit(x)

    return cast(KernelDensity, grid.best_estimator_)


def up_sample(x: pd.DataFrame, _delta: int, last_index: int) -> pd.DataFrame:
    """Add KDE sampled patients.

    To speed up process, override optimize_bw and hard code value:
    _kde = KernelDensity(kernel="gaussian", bandwidth=0.5).fit(x)

    Parameters
    ----------
    x
        Protein signal.
    _delta
        Patients to add.
    last_index
        Index to add onto.

    Returns
    -------
        Up sampled patients.
    """
    _kde = optimize_bw(x.to_numpy()).fit(x)
    x_up = np.array(_kde.sample(n_samples=_delta))
    concat_x = np.concatenate([x, x_up])
    idx = x.index.to_list() + list(range(last_index, last_index + x_up.shape[0]))
    return pd.DataFrame(concat_x, index=idx, columns=x.columns)


def down_sample(x: np_array, _delta: int) -> pd.DataFrame:
    """Subtract random patients.

    Parameters
    ----------
    x
        Protein signal.
    _delta
        Number to remove.

    Returns
    -------
        Deleted patients.
    """
    # number must be positive integer
    _to_subtract = np.abs(_delta)
    # original size
    original_size = x.shape[0]
    # randomly selected indices to remove
    delete_index = np.random.choice(np.array(range(0, original_size)), _to_subtract, replace=False)
    # down sampled data
    x = np.delete(x, delete_index, axis=0)
    return pd.DataFrame(x)


def get_number_of_patients_to_delta(y: np_array, final_ratio_cancer: float, final_total: int) -> dict[str, int]:
    """Generate information about dataset.

    Parameters
    ----------
    y
        Cancer labels, 1 == cancer.
    final_ratio_cancer
        The ratio of cancer samples.
    final_total
        Final number of patients after augmentation.

    Returns
    -------
        Cancer and healthy patients to add or subtract.
    """
    cancer = sum(y == 1)
    healthy = sum(y == 0)

    healthy_delta = final_total / (1 + final_ratio_cancer) - healthy
    cancer_delta = final_total - healthy - cancer - healthy_delta
    assert cancer + cancer_delta + healthy + healthy_delta == final_total, "Final total does not add up."
    assert (
        abs((cancer + cancer_delta) / (healthy + healthy_delta) - final_ratio_cancer) < 0.001
    ), "Final ratio was not met"
    return {
        "healthy_delta": np.round(healthy_delta, 0).astype(int),
        "cancer_delta": np.round(cancer_delta, 0).astype(int),
    }


def augment_cancer_patients(
    X: pd.DataFrame,
    y: np_array,
    final_ratio_cancer: float = 0.5,
    final_total: int = 100,
) -> tuple[pd.DataFrame, np_array]:
    """Augement the sample ratios.

    ratio_cancer = (cancer + cancer_added)/(healthy + healthy_added)
    total = cancer + cancer_added + healthy + healthy_added
    QED
    cancer_added = total - healthy - cancer - healthy_added
    healthy_added = total / (1 + ratio) - healthy

    Parameters
    ----------
    X
        Protein signal.
    y
        Cancer label.
    final_ratio_cancer, optional
        Final ratio of number of cancer to healthy pateints, by default 0.5.
    final_total, optional
        Final number of patients after augmentation, by default 100.

    Returns
    -------
        The data and label augmented.
    """
    x_1, x_0 = X[y == 1], X[y == 0]

    # find number to sample
    _delta = get_number_of_patients_to_delta(y, final_ratio_cancer, final_total)

    if _delta["healthy_delta"] < 0:
        # remove healthy patients
        x_0 = down_sample(x_0, _delta["healthy_delta"])
    else:
        x_0 = up_sample(x_0, _delta["healthy_delta"], last_index=X.index[-1])  # type: ignore
    if _delta["cancer_delta"] < 0:
        # remove cancer patients
        x_1 = down_sample(x_1, _delta["cancer_delta"])
    else:
        x_1 = up_sample(x_1, _delta["cancer_delta"], last_index=X.index[-1])  # type: ignore

    X_aug = pd.concat([x_0, x_1])
    y_aug = np.concatenate([np.zeros(len(x_0)), np.ones(len(x_1))])

    total_cancer_final_check = sum(y_aug == 1)
    total_healthy_final_check = sum(y_aug == 0)
    assert abs(total_cancer_final_check + total_healthy_final_check - final_total) < 2, "Augmentation failed"
    assert (
        abs(total_cancer_final_check / total_healthy_final_check - final_ratio_cancer) < 0.1
    ), "Augmentation failed larger than 10percent error"

    return X_aug, y_aug


def apply_kde_augmentation(
    X: pd.DataFrame,
    y: np_array,
) -> tuple[pd.DataFrame, np_array]:
    """Helper function to augment the data using KDE.

    Parameters
    ----------
    X
        Protein signal.
    y
        Cancer label.

    Returns
    -------
        Augmented X and y
    """
    final_total = len(y)  # we want to keep the same number of patients
    final_cancer_ratio = 1
    final_total = int(np.max(np.unique(y, return_counts=True)[1]) * 2)

    x, y = augment_cancer_patients(X, y, final_ratio_cancer=final_cancer_ratio, final_total=final_total)

    return pd.DataFrame(x, columns=X.columns), y.astype(int)


def apply_smote_augmentation(
    X: pd.DataFrame,
    y: np_array,
    augment_flag: AugmentationEnum,
    new_cancer_ratio: float = 0.5,
    new_total_samples: int = 0,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np_array]:
    """Helper function to augment the data using SMOTE.

    Parameters
    ----------
    X
        Features.
    y
        Target.
    augment_flag
        smote_balanced, smote_tomek, smote_een, smote_ratio.
    new_cancer_ratio, optional
        A new ratio of cancer patients, by default '0.5'.
    new_total_samples, optional
        A new total patients, by default 0.
    random_state, optional
        Random state for reproducibility, by default 42.

    Returns
    -------
        Augmented X and y.
    """
    if augment_flag == "smote_balanced":
        smote = SMOTE(sampling_strategy="auto", random_state=random_state)
        X_aug, y_aug = smote.fit_resample(X, y)  # type: ignore

    elif augment_flag == "smote_tomek":
        smote = SMOTETomek(sampling_strategy="auto", random_state=random_state)
        X_aug, y_aug = smote.fit_resample(X, y)  # type: ignore

    elif augment_flag == "smote_een":
        smote = SMOTEENN(sampling_strategy="auto", random_state=random_state)
        X_aug, y_aug = smote.fit_resample(X, y)  # type: ignore

    elif augment_flag == "smote_ratio":
        # Keeping cancer patients same, change control to get new cancer ratio
        vc = pd.Series(y).value_counts()
        n = (1 - new_cancer_ratio) / new_cancer_ratio
        sampling_strategy_dict = {0: int(vc[1] * n), 1: vc[1]}

        sampler = SMOTE if int(vc[1] * n) >= vc[0] else RandomUnderSampler
        smote = sampler(sampling_strategy=sampling_strategy_dict, random_state=random_state)  # type: ignore
        X_aug, y_aug = smote.fit_resample(X, y)  # type: ignore

    # Rescale the augmented cancer and contol to the desired size
    if new_total_samples:
        vc_aug = pd.Series(y_aug).value_counts()
        total_aug = vc_aug[0] + vc_aug[1]
        scale = new_total_samples / total_aug
        sampling_strategy_dict_aug = {0: int(np.ceil(vc_aug[0] * scale)), 1: int(np.ceil(vc_aug[1] * scale))}

        sampler = SMOTE if scale >= 1.0 else RandomUnderSampler
        smote_aug = sampler(sampling_strategy=sampling_strategy_dict_aug, random_state=random_state)  # type: ignore
        X_aug, y_aug = smote_aug.fit_resample(X_aug, y_aug)  # type: ignore

    return pd.DataFrame(X_aug, columns=X.columns), y_aug.astype(int)  # type: ignore


def apply_augmentation(
    X: pd.DataFrame,
    y: np_array,
    augment_flag: AugmentationEnum,
    new_cancer_ratio: float = 0.5,
    new_total_samples: int = 0,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np_array]:
    """Helper function to augment the data using either KDE or SMOTE.

    Parameters
    ----------
    X
        Features.
    y
        Target.
    augment_flag
        smote_balanced, smote_tomek, smote_een, smote_ratio,
        kde_balanced.
    new_cancer_ratio, optional
        A new ratio of cancer patients, by default '0.5'.
    new_total_samples, optional
        A new total patients, by default 0.
    random_state, optional
        Random state for reproducibility, by default 42.

    Returns
    -------
        Augmented X and y.
    """
    if "kde_balanced" in augment_flag:
        X_aug, y_aug = apply_kde_augmentation(X, y)
    if "smote" in augment_flag:
        X_aug, y_aug = apply_smote_augmentation(X, y, augment_flag, new_cancer_ratio, new_total_samples, random_state)

    return X_aug, y_aug
