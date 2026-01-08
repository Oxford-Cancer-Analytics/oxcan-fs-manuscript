import pandas as pd
from skrebate import MultiSURF
from src.features.models import UnivariateFeature


def test_multisurf(get_X_y_data_imputed):
    X, y, _, protein_list = get_X_y_data_imputed
    mfs = MultiSURF(n_features_to_select=3, n_jobs=-1)
    mfs.fit(X, y)
    results = pd.DataFrame({"features": protein_list, "importances": mfs.feature_importances_}).sort_values(
        by="importances", ascending=False
    )
    final_list = results[results.importances > 0].features.to_list()
    assert "A" in final_list, "Multisurf has diverged"


def test_univariate(get_X_y_data_imputed):
    X, y, _, protein_list = get_X_y_data_imputed
    ufs = UnivariateFeature(n_features_to_select=3, n_jobs=-1)
    ufs.fit(X, y)
    results = pd.DataFrame({"features": protein_list, "importances": ufs.feature_importances_}).sort_values(
        by="importances", ascending=False
    )
    final_list = results[results.importances > 0.95].features.to_list()
    assert "B" in final_list, "Univariate has diverged"
