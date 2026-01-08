import pytest
from src.cli.toml_parser import Headers
from src.cli.toml_parser import TomlData
from src.cli.toml_parser import TomlDataAugmentation
from src.cli.toml_parser import TomlDataFeatureSelection
from src.cli.toml_parser import TomlDataStats


class TestParser:
    def test_parser_defaults(self, parser):
        toml_parser = parser.read()

        assert toml_parser.data == TomlData()

    def test_parser_read(self, parser):
        toml_parser = parser.read()

        assert parser == toml_parser

    @pytest.mark.parametrize(
        ("input", "output"),
        (
            (
                {"stats": {}, "feature_selection": {}, "augmentation": {}},
                Headers(top_level=[], headers=["stats", "feature_selection", "augmentation"]),
            ),
            (
                {
                    "test": "",
                    "top_key": "",
                    "final_top_key": "",
                    "inner": {},
                    "nothing": {},
                    "nested": {},
                },
                Headers(
                    top_level=["test", "top_key", "final_top_key"],
                    headers=["inner", "nothing", "nested"],
                ),
            ),
        ),
    )
    def test_get_headers(self, parser, input, output):
        headers = parser._get_headers(input)

        assert isinstance(headers, Headers)
        assert headers == output
        assert len(headers) == 2

    def test_headers(self, parser):
        assert isinstance(parser.headers, Headers)
        assert parser.headers == Headers(
            top_level=["random_state", "sub_cohort", "scale_transform", "pca"],
            headers=[
                "stats",
                "feature_selection",
                "augmentation",
                "optimization",
            ],
        )

    @pytest.mark.parametrize(
        ("input", "output"),
        (
            (
                {
                    "stats": {},
                    "feature_selection": {},
                    "augmentation": {},
                },
                {
                    "stats": TomlDataStats(),
                    "feature_selection": TomlDataFeatureSelection(
                        multisurf_cross_validation_splits=50, cross_validation_splits=4
                    ),
                    "augmentation": TomlDataAugmentation(
                        smote=TomlDataAugmentation().Smote(
                            new_cancer_ratio=0.5,
                            new_total_samples=0,
                        ),
                    ),
                },
            ),
            (
                {
                    "test": "",
                    "top_key": "",
                    "final_top_key": "",
                    "inner": {},
                    "nothing": {"test": 1},
                    "nested": {},
                },
                {
                    "test": "",
                    "top_key": "",
                    "final_top_key": "",
                    "inner": {},
                    "nothing": {"test": 1},
                    "nested": {},
                },
            ),
        ),
    )
    def test_format_toml(self, parser, input, output):
        formatted_toml = parser._format_toml(input)

        assert isinstance(formatted_toml, dict)
        assert formatted_toml == output
        assert len(set(output.keys()).difference(formatted_toml.keys())) == 0

    def test_to_dict(self, parser):
        toml_parser = parser.read()

        assert toml_parser.data.to_dict() == {
            "random_state": 0,
            "sub_cohort": "all",
            "scale_transform": True,
            "pca": True,
            "stats": {
                "pvalue_thresh": 0.05,
                "power": 0.95,
                "alpha": 0.05,
            },
            "feature_selection": {
                "multisurf_cross_validation_splits": 50,
                "cross_validation_splits": 4,
                "cross_validation_repeats": 10,
                "number_of_mi_features": "effective_dim",
                "number_of_features": 150,
                "addition": {
                    "method": "feature_importance",
                    "importance_selection": "shap",
                },
            },
            "augmentation": {
                "smote": {
                    "new_cancer_ratio": 0.5,
                    "new_total_samples": 0,
                },
            },
            "optimization": {
                "optuna": {
                    "scorer": "sens@99spec",
                    "trials": 200,
                    "before_feature_selection": False,
                    "after_rfa": True,
                    "before_validation": False,
                },
            },
        }
        assert toml_parser.data.to_dict(true_values=True) == {
            "sub_cohort": "all",
            "scale_transform": True,
            "pca": True,
            "stats": {
                "pvalue_thresh": 0.05,
                "power": 0.95,
                "alpha": 0.05,
            },
            "feature_selection": {
                "multisurf_cross_validation_splits": 50,
                "cross_validation_splits": 4,
                "cross_validation_repeats": 10,
                "number_of_mi_features": "effective_dim",
                "number_of_features": 150,
                "addition": {
                    "importance_selection": "shap",
                    "method": "feature_importance",
                },
            },
            "augmentation": {
                "smote": {
                    "new_cancer_ratio": 0.5,
                },
            },
            "optimization": {
                "optuna": {
                    "scorer": "sens@99spec",
                    "trials": 200,
                    "after_rfa": True,
                },
            },
        }

    def test_getitem(self, parser):
        toml_parser = parser.read()

        assert toml_parser.data["random_state"] == toml_parser.data.random_state

    def test_setitem(self, parser):
        toml_parser = parser.read()

        with pytest.raises(AttributeError):
            assert toml_parser.data.testing

        toml_parser.data["testing"] = 42
        assert toml_parser.data.testing == 42

        toml_parser.data[{"testing": 41}] = "value"
        assert toml_parser.data.testing == 41

    def test_setitem_false(self, parser):
        toml_parser = parser.read()

        assert not all(val is False for val in toml_parser.data.stats.to_dict().values())
