from enum import Enum


class ModelsEnum(str, Enum):
    """Model options."""

    BALANCED_RANDOM_FOREST = "balanced_random_forest"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    _TEST_CASE_IGNORE = "Test"


class FeatureSelectionEnum(str, Enum):
    """Feature selection options."""

    UNIVARIATE = "UnivariateFeature"
    MULTISURF = "MultiSURF"
    MUTUAL_INFORMATION = "MutualInformation"
    BORUTA_SHAP = "BorutaShap"


class StatisticsEnum(str, Enum):
    """Statistical tests options."""

    TTEST_INDEPENDENT = "ttest_ind"


class AugmentationEnum(str, Enum):
    """Augmentation options."""

    NONE = "none"  # No augmentation
    SMOTE_TOMEK = "smote_tomek"  # Performs smote and then downsamples with Tomek links
    SMOTE_RATIO = "smote_ratio"  # Keep the same number of cancer patients and adjusts ratio
    SMOTE_EEN = "smote_een"  # Performs smote and then downsamples with Edited nearest neighbours
    SMOTE_BALANCED = "smote_balanced"  # Keep the same number of patients just balance
    KDE_BALANCED = "kde_balanced"  # Keep the same number of patients just balance
