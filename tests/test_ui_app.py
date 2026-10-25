"""UI Integration Tests for Streamlit App using AppTest (Gate 4)."""

import pytest
from streamlit.testing.v1 import AppTest


def test_app_boots_cleanly_default():
    """Verify app.py boots headless with default settings without exceptions."""
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert not at.exception, f"App raised exception on boot: {at.exception}"


@pytest.mark.parametrize("vehicle_index", [0, 1, 2])
@pytest.mark.parametrize("rain,current", [
    (2.0, 1.0),  # Extreme severe monsoon & torrential current
    (0.5, 0.0),  # Arid calm weather
])
def test_app_extreme_settings_across_vehicles(vehicle_index, rain, current):
    """
    Verify the application runs with zero exceptions across all 3 vehicles
    under extreme weather conditions (rain 2.0, current 1.0, rain 0.5, current 0.0).
    """
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert not at.exception, f"Initial boot failed: {at.exception}"

    # Change vehicle selectbox
    at.sidebar.selectbox[0].select_index(vehicle_index)
    # Set sliders: rainfall (index 5) and current (index 6) in sidebar
    # Find slider widgets by label
    for slider in at.sidebar.slider:
        if "Rainfall" in slider.label:
            slider.set_value(rain)
        elif "Current" in slider.label:
            slider.set_value(current)

    at.run()
    assert not at.exception, f"App raised exception under vehicle index {vehicle_index}, rain {rain}, current {current}: {at.exception}"
