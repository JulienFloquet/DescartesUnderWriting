EARTH_RADIUS = 6378

TIME_COLUMN = "time"
PAYOUT_COLUMN = "payout"
MAGNITUDE_COLUMN = "mag"
DISTANCE_COLUMN = "distance"
LATITUDE_COLUMN = "latitude"
LONGITUDE_COLUMN = "longitude"

import numpy as np
import pandas as pd


def get_haversine_distance(
    lat_series: Iterable[float],
    lon_series: Iterable[float],
    lat_point: float,
    lon_point: float,
) -> np.ndarray:
    """
    Compute the haversine distance between a collection of
    latitude/longitude points and a single reference point.

    Parameters
    ----------
    lat_series : pandas.Series or array‑like
        Latitudes of the locations you want to measure *from* (degrees).
    lon_series : pandas.Series or array‑like
        Longitudes of the locations you want to measure *from* (degrees).
    lat_point : float
        Latitude of the reference point (degrees).
    lon_point : float
        Longitude of the reference point (degrees).

    Returns
    -------
    distances : np.ndarray
        Distance from each (lat,lon) pair to the reference point, expressed in km.
    """

    lat_arr = np.asarray(lat_series, dtype=np.float64)
    lon_arr = np.asarray(lon_series, dtype=np.float64)

    lat1 = np.radians(lat_arr)          # points from the dataset
    lon1 = np.radians(lon_arr)
    lat2 = np.radians(lat_point)        # reference point
    lon2 = np.radians(lon_point)

    # Apply the haversine formula (vectorised).
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    distances = EARTH_RADIUS * c

    return distances
    


def compute_payouts(
    earthquake_data: pd.DataFrame,
    payout_rules: List[Dict],
) -> Dict[int, float]:

    """
    Compute yearly aggregated payouts for earthquake events.

    Parameters
    ----------
    earthquake_data : pd.DataFrame
        Must contain the columns:
            - time     : datetime (date event)
            - mag      : float    (magnitude)
            - depth    : float    (distance from epicenter)
    payout_rules : List[Dict]
        Each dict defines a rule for the payout by the radius and magnitude

    Returns
    -------
    Dict[int, float]
        Mapping of "year" → "max payout" for that year.
    """

    earthquake_data_copy = earthquake_data.copy()

    # We need to verify that column time is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(earthquake_data_copy["time"]):
        earthquake_data_copy["time"] = pd.to_datetime(earthquake_data_copy["time"], errors="coerce")

    earthquake_data_copy = earthquake_data_copy.dropna(subset=["time"])

    def compute_payout(row) -> float:
        payouts = [
            rule["payout"]
            for rule in payout_rules
            if row["depth"] <= rule["radius"]
            and row["mag"] >= rule["magnitude"]
        ]
        return max(payouts) if payouts else 0.0

    earthquake_data_copy["payout"] = earthquake_data_copy.apply(compute_payout, axis=1)
    earthquake_data_copy["year"] = earthquake_data_copy["time"].dt.year

    yearly_totals = earthquake_data_copy.groupby("year")["payout"].max()

    return yearly_totals.to_dict()



def compute_burning_cost(
    payouts: Dict[int, float], 
    start_year: float, 
    end_year: float,
) -> float :

    """
    Calculates the average burning cost over a given period.

    Parameters
    ----------
    payouts: dict[int, float]
        The percentage of payout associated with that year. If a year is not present, 
        its payout is 0.
    start_year: float
        Start year of the period to be analyzed.
    end_year: float
        End year of the period to be analyzed.

    Returns
    ------
    float
        The average cost. Returns 0 if the period does not include any years.
    """
    
    if start_year > end_year:
        raise ValueError("start_year must be less than or equal to end_year")

    total_years = end_year - start_year + 1
    if total_years == 0:
        return 0.0

    total_pct = sum(payouts.get(year, 0.0) for year in range(start_year, end_year + 1))

    return total_pct / total_years