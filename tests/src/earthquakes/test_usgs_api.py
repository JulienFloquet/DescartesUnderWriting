import pytest
from datetime import datetime, timedelta

from earthquakes.usgs_api import build_api_url


def test_build_api_url_exact_match():
    """Test full exact URL match for strict verification."""
    assets = (10.0, 20.0)
    endtime = datetime(2019, 2, 2)
    minimum_magnitude = 5.0
    radius = 50

    starttime = (endtime - timedelta(days=200 * 365)).strftime("%Y-%m-%d")
    endtime_str = endtime.strftime("%Y-%m-%d")

    expected_url = (
        "https://earthquake.usgs.gov/fdsnws/event/1/query"
        f"?format=geojson&starttime={starttime}&endtime={endtime_str}"
        f"&latitude=10.0&longitude=20.0"
        f"&minmagnitude=5.0&maxradiuskm=50"
    )

    result = build_api_url(assets, endtime, minimum_magnitude, radius)

    assert result == expected_url


@pytest.mark.parametrize(
    "assets, minimum_magnitude, radius",
    [
        ((0.0, 0.0), 1.0, 10),
        ((-45.123, 179.999), 2.5, 500),
        ((89.999, -179.999), 9.9, 1),
    ],
)
def test_build_api_url_various_inputs(assets, minimum_magnitude, radius):
    endtime = datetime(2020, 1, 1)

    result = build_api_url(assets, endtime, minimum_magnitude, radius)

    lat, lon = assets

    assert f"latitude={lat}" in result
    assert f"longitude={lon}" in result
    assert f"minmagnitude={minimum_magnitude}" in result
    assert f"maxradiuskm={radius}" in result