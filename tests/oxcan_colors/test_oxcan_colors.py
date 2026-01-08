import pytest
from src.oxcan_colors import OXcanColors


class TestOXcanColors:
    def test_init(self):
        with pytest.raises(ValueError):
            assert OXcanColors("test.json")

    def test_str_repr(self):
        colors = OXcanColors()
        output = str(colors)

        assert output == "blue orange green turquoise mint purple black pink grey white"

    @pytest.mark.parametrize(("num_colors", "out_len"), ((10, 10), (0, 0), (-4, 36)))
    def test_get_colors(self, num_colors, out_len):
        colors = OXcanColors()
        output = colors.get_colors(num_colors)

        assert isinstance(output, list)
        assert all([color.startswith("#") for color in output])
        assert len(output) == out_len

    @pytest.mark.parametrize(("color", "num"), (("blue", 10), ("pink", 2), ("white", 36)))
    def test_get_shades(self, color, num):
        colors = OXcanColors()
        output = colors.get_shades(primary=color, num_shades=num)

        assert isinstance(output, list)
        assert len(output) == min(len(colors.colors[color]), num)
        assert all([color.startswith("#") for color in output])

    def test_get_shades_raises(self):
        colors = OXcanColors()

        with pytest.raises(ValueError):
            assert colors.get_shades("test_color")

    @pytest.mark.parametrize(
        ("min_color", "max_color"), (("#4dca33", "green"), ("orange", "white"), ("pink", "#d50f61"))
    )
    def test_get_2color_shade_of_value(self, min_color, max_color):
        colors = OXcanColors()
        output = colors.get_2color_shade_of_value(0.5, min_col=min_color)

        assert output.startswith("#")
