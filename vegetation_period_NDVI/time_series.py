import ee
import math
from typing import List, Dict, Any
from utils.composites import harmonized_ts
from .data_loading import (
    load_sentinel2_data,
    ndvi_band_to_int,
    ndvi_band_to_float,
    add_time_data,
)

from utils.harmonic_regressor_ndvi import HarmonicRegressor


def extract_time_ranges(time_range: List[str], agg_interval: int) -> ee.List:
    """
    Extract time intervals for generating temporal composites from Sentinel collections.

    Args:
        time_range (List[str]): Start and end dates in 'YYYY-MM-DD' format.
        agg_interval (int): Number of days for each interval.

    Returns:
        ee.List: List of time intervals. Each interval is an ee.List with [start_date, end_date].
    """
    start_date = ee.Date(time_range[0])
    end_date = ee.Date(time_range[1])

    interval_no = (
        ee.Date(time_range[1])
        .difference(ee.Date(time_range[0]), "day")
        .divide(agg_interval)
        .round()
    )
    month_check = ee.Number(30).divide(agg_interval).ceil()
    rel_delta = (
        ee.Number(end_date.difference(start_date, "day"))
        .divide(ee.Number(30.5).multiply(interval_no))
        .ceil()
    )

    end_date = start_date.advance(
        start_date.advance(rel_delta, "month")
        .difference(start_date, "day")
        .divide(month_check),
        "day",
    )

    time_intervals = ee.List([ee.List([start_date, end_date])])

    def add_interval(x, previous):
        x = ee.Number(x)
        start_date1 = ee.Date(
            ee.List(ee.List(previous).reverse().get(0)).get(1)
        )  # end_date of last element
        end_date1 = start_date1.advance(
            start_date1.advance(rel_delta, "month")
            .difference(start_date1, "day")
            .divide(month_check),
            "day",
        )
        return ee.List(previous).add(ee.List([start_date1, end_date1]))

    time_intervals = ee.List(
        ee.List.sequence(2, interval_no).iterate(add_interval, time_intervals)
    )

    return time_intervals


def prepare_harmonized_data(
    yearly_sentinel_data: ee.ImageCollection,
    time_intervals: ee.List,
    agg_type: str = "geomedian",
) -> ee.ImageCollection:
    """Prepare harmonized data from Sentinel-2 imagery.

    Args:
        yearly_sentinel_data (ee.ImageCollection): Sentinel-2 imagery for a year.
        time_intervals (ee.List): List of time intervals for aggregation.
        agg_type (str): Aggregation type for harmonized_ts function. Default is "geomedian".

    Returns:
        ee.ImageCollection: Harmonized time series of Sentinel-2 imagery.
    """
    harmonized_data = harmonized_ts(
        yearly_sentinel_data.map(ndvi_band_to_int),
        ["NDVI_int", "NDVI"],
        time_intervals,
        {"agg_type": agg_type},
    ).map(lambda img: ndvi_band_to_float(ee.Image(img)))
    return harmonized_data.map(add_time_data)


def interpolate_missing_data(
    harmonized_data: ee.ImageCollection, window_days: int = 30
) -> ee.ImageCollection:
    """Interpolate missing data in the harmonized time series."""

    def interpolate_image(image: ee.Image) -> ee.Image:
        current_date = ee.Date(image.get("system:time_start"))
        mean_filtered_image = harmonized_data.filterDate(
            current_date.advance(-window_days / 2, "day"),
            current_date.advance(window_days / 2, "day"),
        ).mean()
        return mean_filtered_image.where(image, image).copyProperties(
            image, ["system:time_start"]
        )

    return ee.ImageCollection(harmonized_data.map(interpolate_image))


def get_harmonic_ts(
    year: int,
    aoi: ee.Geometry,
    time_intervals: ee.List,
    agg_type: str = "geomedian",
    interpolation_window: int = 30,
    omega: float = 1.5,
    max_harmonic_order: int = 2,
    parallel_scale: int = 2,
) -> Dict[str, Any]:
    """
    Generate a harmonized time series of sentinel 2 data with harmonic regression for a given year and area.

    Args:
        year (int): The year for which to generate the time series.
        aoi (ee.Geometry): The area of interest.
        time_intervals (ee.List): List of time intervals for aggregation.
        agg_type (str): Aggregation type for harmonized_ts function. Default is "geomedian".
        interpolation_window (int): Window size in days for interpolation. Default is 30.
        omega (float): Omega parameter for HarmonicRegressor. Default is 1.5.
        max_harmonic_order (int): Maximum harmonic order for HarmonicRegressor. Default is 2.
        parallel_scale (int): Parallel scale for HarmonicRegressor. Default is 2.

    Returns:
        Dict[str, Any]: Dictionary containing fitted data, regression coefficients, and phase/amplitude.
    """
    try:
        yearly_sentinel_data = load_sentinel2_data(year, aoi)

        harmonized_data = prepare_harmonized_data(
            yearly_sentinel_data, time_intervals, agg_type
        )

        harmonized_data = interpolate_missing_data(
            harmonized_data, interpolation_window
        )

        regressor = HarmonicRegressor(
            omega=omega,
            max_harmonic_order=max_harmonic_order,
            vegetation_index="NDVI",
            parallel_scale=parallel_scale,
        )
        regressor.fit2(harmonized_data)

        fitted_data = regressor.predict(harmonized_data)
        regression_coefficients = regressor._regression_coefficients
        phase_amplitude = regressor.get_phase_amplitude()

        return {
            "fitted_data": fitted_data,
            "regression_coefficients": regression_coefficients,
            "phase_amplitude": phase_amplitude,
        }
    except Exception as e:
        print(f"An error occurred in get_harmonic_ts: {str(e)}")
        raise
