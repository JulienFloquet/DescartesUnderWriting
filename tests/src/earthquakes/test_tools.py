import pytest
import numpy as np
import pandas as pd

from earthquakes.tools import get_haversine_distance, EARTH_RADIUS


def test_haversine_known_distance():
    newYork = (40.7128, -74.0060)
    london = (51.5074, -0.1278)
    
    distance = get_haversine_distance(
        lat_series=[newYork[0]], lon_series=[newYork[1]],
        lat_point=london[0], lon_point=london[1]
    )
    
    expected_distance = 5570
    np.testing.assert_allclose(distance, expected_distance, rtol=0.01)


def test_haversine_array_input():
    lats = np.array([0, 90])
    lons = np.array([0, 0])
    
    ref_lat, ref_lon = 0, 0
    distances = get_haversine_distance(lats, lons, ref_lat, ref_lon)
    
    expected = np.array([0, np.pi/2 * EARTH_RADIUS])
    np.testing.assert_allclose(distances, expected)


def test_haversine_invalid_input():
    with pytest.raises(ValueError):
        get_haversine_distance(['a', 'b'], ['c', 'd'], 0, 0)