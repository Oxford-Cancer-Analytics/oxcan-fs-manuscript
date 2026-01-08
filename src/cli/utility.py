from typing import Any


def diff(default: dict[str, Any], cli: dict[str, Any], _nested_key: str | None = None) -> dict[str, Any]:
    """Gets the difference between two dicts.

    Parameters
    ----------
    default
        The first dict to compare.
    cli
        The second dict to compare to `default`.
    _nested_key, optional
        Internally used to compare nested dicts, by default None

    Returns
    -------
        The difference between the two dicts, keeping the same
        structure.
    """
    diff_dict: dict[str, Any] = {}
    for (default_key, default_value), (_, cli_value) in zip(default.items(), cli.items()):
        if isinstance(default_value, dict) and isinstance(cli_value, dict):
            diff_dict |= diff(default_value, cli_value, default_key)
            continue

        if default_value != cli_value:
            if _nested_key:
                diff_dict[_nested_key] = {default_key: cli_value}
            else:
                diff_dict[default_key] = cli_value

    return diff_dict
