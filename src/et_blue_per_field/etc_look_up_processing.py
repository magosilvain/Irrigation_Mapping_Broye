import pandas as pd
import ee
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np


def _add_day_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a Day column to the dataframe based on dekade values.

    Args:
        df (pd.DataFrame): Input dataframe containing a 'dekade' column

    Returns:
        pd.DataFrame: DataFrame with new 'Day' column
    """
    # Create mapping dictionary for dekade to day
    dekade_to_day = {1: 1, 2: 11, 3: 21}

    # Add Day column using map
    df["Day"] = df["dekade"].map(dekade_to_day)

    return df


def add_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a Date column to the dataframe by combining Year, Month, and Day columns.

    Args:
        df (pd.DataFrame): Input dataframe containing 'Year', 'Month', and 'Day' columns

    Returns:
        pd.DataFrame: DataFrame with new 'Date' column as datetime type
    """
    # Add Day column
    df = _add_day_column(df)

    # Combine Year, Month, Day into a datetime column
    df["Date"] = pd.to_datetime(df[["Year", "Month", "Day"]])

    return df


def create_et_mask(
    image: ee.Image,
    etc_df: pd.DataFrame,
    band_name: str,
    threshold: float,
) -> ee.Image:
    """
    Create a mask for an image based on ETc thresholds from a lookup table.

    Args:
        image: Input image to be masked
        etc_df: DataFrame containing ETc values with columns: ['ETc', 'Date']
        band_name: Name of the band to compare against ETc
        threshold: Value between 0 and 1 to multiply with ETc

    Returns:
        ee.Image with values 0 (masked) where pixel > threshold * ETc
        and 1 (unmasked) where pixel ≤ threshold * ETc
    """
    # Validate threshold
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")

    # Get image date
    image_date = ee.Date(image.get("system:time_start"))

    # Convert dates and ETc values to Earth Engine objects
    lookup_data = ee.List(
        [
            ee.Dictionary(
                {"date": ee.Date(date.strftime("%Y-%m-%d")), "etc": ee.Number(etc)}
            )
            for date, etc in zip(etc_df["Date"], etc_df["ETc"])
        ]
    )

    # Function to find matching date
    def find_match(item, prev):
        date = ee.Dictionary(item).get("date")
        etc = ee.Dictionary(item).get("etc")
        is_match = ee.Date(date).millis().eq(image_date.millis())
        return ee.Dictionary(prev).set(
            "etc", ee.Algorithms.If(is_match, etc, ee.Dictionary(prev).get("etc"))
        )

    # Find matching ETc value
    initial = ee.Dictionary({"etc": ee.Number(0)})
    match_result = ee.Dictionary(lookup_data.iterate(find_match, initial))
    etc_value = ee.Number(match_result.get("etc"))

    # Calculate threshold value
    threshold_value = etc_value.multiply(threshold)

    # Create mask (0 where value > threshold*ETc, 1 otherwise)
    mask = image.select(band_name).lte(threshold_value)

    return mask


def _create_etc_lookup(etc_df: pd.DataFrame) -> ee.List:
    """Convert ETc DataFrame into EE lookup format.

    Args:
        etc_df: DataFrame with Date and ETc columns

    Returns:
        ee.List of dictionaries with date and etc values
    """
    return ee.List(
        [
            ee.Dictionary(
                {
                    "date": ee.Date(date.strftime("%Y-%m-%d")),
                    "etc": ee.Number(float(etc)),
                }
            )
            for date, etc in zip(etc_df["Date"], etc_df["ETc"])
        ]
    )


def _process_single_image(
    image: ee.Image, collection_idx: int, lookup_data: ee.List, et_band_name: str = "ET"
) -> ee.Feature:
    """Process a single image and compute ET statistics.

    Handles empty/masked images by returning null values for statistics.

    Args:
        image: Input ET image
        collection_idx: Index of current collection (1-based for output naming)
        lookup_data: List of EE dictionaries with date/ETc pairs
        et_band_name: Name of the ET band in the image

    Returns:
        Feature containing date, min_et, etc, and et_ratio properties
    """
    image_date = ee.Date(image.get("system:time_start"))

    # Check if image has any valid pixels
    valid_pixels = image.select(et_band_name).unmask(None)
    valid_count = (
        valid_pixels.multiply(0)
        .add(1)
        .reduceRegion(reducer=ee.Reducer.count(), maxPixels=int(1e9))
        .get(et_band_name)
    )

    def compute_stats() -> ee.Feature:
        # Get minimum ET value
        min_et = ee.Number(
            valid_pixels.reduceRegion(
                reducer=ee.Reducer.median(), maxPixels=int(1e9)
            ).get(et_band_name)
        )

        # Find matching ETc value
        def find_match(item: ee.Dictionary, prev: ee.Dictionary) -> ee.Dictionary:
            date = ee.Dictionary(item).get("date")
            etc = ee.Dictionary(item).get("etc")
            is_match = ee.Date(date).millis().eq(image_date.millis())
            return ee.Dictionary(prev).set(
                "etc", ee.Algorithms.If(is_match, etc, ee.Dictionary(prev).get("etc"))
            )

        etc_value = ee.Number(
            ee.Dictionary(
                lookup_data.iterate(find_match, ee.Dictionary({"etc": ee.Number(0)}))
            ).get("etc")
        )

        # Compute ratio if ETc > 0
        et_ratio = ee.Algorithms.If(etc_value.gt(0), min_et.divide(etc_value), None)

        return ee.Feature(
            None,
            {
                "date": image_date.format("YYYY-MM-dd"),
                f"min_et_{collection_idx+1}": min_et,
                "etc": etc_value,
                f"et_ratio_{collection_idx+1}": et_ratio,
            },
        )

    def empty_feature() -> ee.Feature:
        return ee.Feature(
            None,
            {
                "date": image_date.format("YYYY-MM-dd"),
                f"min_et_{collection_idx+1}": None,
                "etc": None,
                f"et_ratio_{collection_idx+1}": None,
            },
        )

    return ee.Algorithms.If(
        ee.Number(valid_count).gt(0), compute_stats(), empty_feature()
    )


def compute_et_ratio_timeseries(
    et_collections: List[ee.ImageCollection],
    etc_df: pd.DataFrame,
    et_band_name: str = "ET",
) -> pd.DataFrame:
    """Compute time series of ET ratios for multiple collections.

    Args:
        et_collections: List of ET image collections to process
        etc_df: DataFrame with Date and ETc columns
        et_band_name: Name of the ET band in the images

    Returns:
        DataFrame with columns for date, min_et_X, etc, and et_ratio_X
        where X is the collection index (1-based)
    """
    # Convert ETc data to EE format
    lookup_data = _create_etc_lookup(etc_df)

    # Process each collection
    all_results = []
    for idx, collection in enumerate(et_collections):
        results = (
            collection.map(
                lambda img: _process_single_image(img, idx, lookup_data, et_band_name)
            )
            .sort("date")
            .toList(collection.size())
        )
        all_results.append(results.getInfo())

    # Combine results into DataFrame
    data = []
    for time_idx in range(len(all_results[0])):
        combined_props = {}
        date_str = all_results[0][time_idx]["properties"]["date"]
        combined_props["date"] = datetime.strptime(date_str, "%Y-%m-%d").date()

        for collection_idx in range(len(et_collections)):
            props = all_results[collection_idx][time_idx]["properties"]
            for key in [
                f"min_et_{collection_idx+1}",
                "etc",
                f"et_ratio_{collection_idx+1}",
            ]:
                combined_props[key] = props.get(key)

        data.append(combined_props)

    return pd.DataFrame(data).sort_values("date")


def plot_multiple_et_ratio_timeseries(
    df: pd.DataFrame, labels: List[str] = None
) -> None:
    """Plot multiple ET ratio time series."""
    plt.figure(figsize=(12, 6))

    ratio_columns = [col for col in df.columns if col.startswith("et_ratio_")]
    if labels is None:
        labels = [f"Collection {i+1}" for i in range(len(ratio_columns))]

    for col, label in zip(ratio_columns, labels):
        plt.plot(df["date"], df[col], marker="o", label=label)

    plt.axhline(y=0.6, color="r", linestyle="--", alpha=0.5, label="Threshold")
    plt.title("Time Series of ET/ETc Ratios (Using Minimum ET)")
    plt.xlabel("")
    plt.ylabel("ET/ETc Ratio")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    sns.despine()
    plt.show()
