import json
from pathlib import Path

import matplotlib.colors as mplc
import numpy as np


class OXcanColors:
    """Class to allow the usage of branding colours in plots.

    Parameters
    ----------
    scheme_file
        The file to load in color options.
    """

    colors: dict[str, list[str]] = {}

    def __init__(self, scheme_file: str = "colors.json") -> None:
        full_path = Path(__file__).parent / scheme_file
        if full_path.exists():
            with open(full_path) as cf:
                self.colors = json.load(cf)
        else:
            raise ValueError("Colour scheme file does not exist!")

    def __str__(self) -> str:  # noqa: D105
        return " ".join([i for i in self.colors.keys()])

    def get_colors(self, num_colors: int = -1) -> list[str]:
        """Returns a list of colours.

        Parameters
        ----------
        num_colors, optional
            by default -1
        """
        list_col = [v for _, v in self.colors.items()]
        list_col = list(np.concatenate(np.array(list_col).T))
        return list_col[:num_colors]  # type: ignore

    def get_shades(self, primary: str = "blue", num_shades: int = 4) -> list[str]:
        """Returns shades of a primary color.

        Parameters
        ----------
        primary, optional
            by default "blue"
        num_shades, optional
            by default 4

        Returns
        -------
            list of shades

        Raises
        ------
        ValueError
            If the color is not a valid value in `self.colors`.
        """
        if primary not in self.colors:
            raise ValueError("Primary color does not exist in palette!")
        return self.colors[primary][:num_shades]

    def get_2color_shade_of_value(
        self, value: float, min_col: str = "blue", max_col: str = "white", min_val: float = 0.0, max_val: float = 1.0
    ) -> str:
        """Returns the HEX representation of the primary colour shade.

        Parameters
        ----------
        value
            A numeric value between min_val and max_val
        min_col, optional
            Colour of the min range, by default "blue"
        max_col, optional
            Colour of the max range, by default "white"
        min_val, optional
            Minimum range for shadding, by default 0.0
        max_val, optional
            Maximum range for shadding, by default 1.0

        Returns
        -------
        A colour in hex format.
        """
        if min_col[0] != "#":
            min_col = self.colors[min_col][2]
        if max_col[0] != "#":
            max_col = self.colors[max_col][2]

        value = (value - min_val) / (max_val - min_val)
        min_col_rgb = np.array(mplc.to_rgb(min_col))
        max_col_rgb = np.array(mplc.to_rgb(max_col))

        return mplc.to_hex((1 - value) * min_col_rgb + value * max_col_rgb)  # type: ignore
