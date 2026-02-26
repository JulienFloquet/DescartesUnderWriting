"""
Utilities for querying the USGS Earthquake API.
"""

import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
import pandas as pd
import asyncio
import aiohttp



def build_api_url(
    assets: Tuple[float, float],
    endtime: datetime,
    minimum_magnitude: float,
    radius: float,
) -> str:
    """
    Return the full USGS query URL for a single latitude/longitude pair.
    
    Parameters
    ----------
    assets : float
        Latitude and longitude of the asset.
    end_date : datetime
        Maximum date in YYYY-MM-DD format.
    minimum_magnitude : float
        Magnitude threshold.
    radius : float
        Search radius in kilometers.

    Returns
    -------
    str
        url for API
    """
    
    lat, lon = assets
    base = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = (
        f"?format=geojson&starttime={(endtime- timedelta(days=200 * 365)).strftime("%Y-%m-%d")}"
        f"&endtime={endtime.strftime("%Y-%m-%d")}"
        f"&latitude={lat}&longitude={lon}"
        f"&minmagnitude={minimum_magnitude}&maxradiuskm={radius}"
    )
    return base + params



def get_earthquake_data(
    latitude: float,
    longitude: float,
    radius: float,
    minimum_magnitude: float,
    end_date: datetime,
) -> pd.DataFrame:
    """
    Queries the USGS catalog and returns the results as a DataFrame.

    Parameters
    ----------
    latitude : float
        Latitude of the asset.
    longitude : float
        Asset longitude.
    radius : float
        Search radius in kilometers.
    minimum_magnitude : float
        Magnitude threshold.
    end_date : datetime
        Maximum date in YYYY-MM-DD format.

    Returns
    -------
    pd.DataFrame
        A DataFrame where each row represents an earthquake(date in descending order). 
        The main columns are :
        [
            'time', 'latitude', 'longitude', 'depth', 'mag', 'magType',
            'nst', 'gap', 'dmin', 'rms', 'net', 'id', 'updated',
            'place', 'type', 'horizontalError', 'depthError',
            'magError', 'magNst', 'status', 'locationSource',
            'magSource'
        ]
    """

    url = build_api_url(
                assets=[latitude, longitude],
                endtime=end_date,
                minimum_magnitude=minimum_magnitude,
                radius=radius,
            )

    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            # JSON encoded in UTF‑8
            raw_bytes = response.read()
            charset = response.headers.get_content_charset() or "utf-8"
            raw_text = raw_bytes.decode(charset)
            data = json.loads(raw_text)
    except urllib.error.HTTPError as exc:
        # HTTP error
        raise exc
    except urllib.error.URLError as exc:
        # Network errors
        raise exc

    records = []
    for feat in data.get("features", []):
        prop = feat["properties"]
        geom = feat["geometry"]["coordinates"]  # [lon, lat, depth]

        ids = prop.get("ids")
        event_id = None

        if ids:
            id_list = ids.strip(",").split(",")
            event_id = id_list[0] if id_list else None
        else:
            event_id = prop.get("id")

        record = {
            "time": pd.to_datetime(prop.get("time"), unit="ms", utc=True).round("ms").strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-4] + "Z",
            "latitude": geom[1],
            "longitude": geom[0],
            "depth": geom[2],
            "mag": prop.get("mag"),
            "magType": prop.get("magType"),
            "nst": prop.get("nst"),
            "gap": prop.get("gap"),
            "dmin": prop.get("dmin"),
            "rms": prop.get("rms"),
            "net": prop.get("net"),
            "id": event_id,
            "updated": pd.to_datetime(prop.get("updated"), unit="ms", utc=True).round("ms").strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-4] + "Z"
                     if prop.get("updated") is not None else np.nan,
            "place": prop.get("place"),
            "type": prop.get("type"),
            "horizontalError": prop.get("horizontalError"),
            "depthError": prop.get("depthError"),
            "magError": prop.get("magError"),
            "magNst": prop.get("magNst"),
            "status": prop.get("status"),
            "locationSource": prop.get("locationSource"),
            "magSource": prop.get("magSource"),
        }
        records.append(record)

    earthquake_data = pd.DataFrame.from_records(records)

    if not earthquake_data.empty:
        earthquake_data = earthquake_data.sort_values("time", ascending=False).reset_index(drop=True)
    return earthquake_data



async def get_earthquake_data_for_multiple_locations(
    assets: Iterable[Tuple[float, float]],
    end_date: datetime,
    minimum_magnitude: float,
    radius: float,
    max_concurrent_requests: int = 50,
) -> pd.DataFrame:
    """
    Async wrapper to query multiple locations concurrently by calling
    the synchronous get_earthquake_data function.

    Parameters
    ----------
    latitude : Iterable[Tuple[float, float]]
        Latitude and longitude of the assets.
    longitude : float
        Asset longitude.
    minimum_magnitude : float
        Magnitude threshold.
    radius : float
        Search radius in kilometers.
    end_date : datetime
        Maximum date in YYYY-MM-DD format.


    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame for all locations.
    """
    semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def fetch(loc: Tuple[float, float]) -> pd.DataFrame:
        async with semaphore:
            loop = asyncio.get_running_loop()
            try:
                earthquake_data = await loop.run_in_executor(
                    None,  
                    get_earthquake_data,
                    loc[0], loc[1], radius, minimum_magnitude, end_date
                )
                earthquake_data["query_latitude"] = loc[0]
                earthquake_data["query_longitude"] = loc[1]
                return earthquake_data
            except Exception as exc:
                return pd.DataFrame([{
                    "error": str(exc),
                    "query_latitude": loc[0],
                    "query_longitude": loc[1]
                }])

    tasks = [fetch(loc) for loc in assets]
    results = await asyncio.gather(*tasks)
    return pd.concat(results, ignore_index=True)