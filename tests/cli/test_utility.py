from src.cli.utility import diff


class TestUtility:
    def test_diff_no_change(self):
        data = {"test": "pass"}
        result = diff(data, data)

        assert result == {}

    def test_diff_change(self):
        default_data = {"test": "fail"}
        cli_data = {"test": "pass"}
        result = diff(default_data, cli_data)

        assert result == cli_data

    def test_diff_nested_dict(self):
        default_data = {"test": "fail", "nested": {"value": "true"}}
        cli_data = {"test": "pass", "nested": {"value": "false"}}
        result = diff(default_data, cli_data)

        assert result == cli_data
