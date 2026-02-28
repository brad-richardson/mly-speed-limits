"""Tests for slc.viz — color mapping and map construction."""

from slc.viz import speed_color


def test_speed_color_known():
    assert speed_color(35) == "#fee090"
    assert speed_color(25) == "#abd9e9"
    assert speed_color(55) == "#a50026"


def test_speed_color_none():
    assert speed_color(None) == "#aaaaaa"


def test_speed_color_unknown_high():
    c = speed_color(999)
    assert c.startswith("#")


def test_speed_color_returns_string():
    for v in [15, 20, 25, 30, 35, 40, 45, 50, 55, 65, 70, None]:
        assert isinstance(speed_color(v), str)
