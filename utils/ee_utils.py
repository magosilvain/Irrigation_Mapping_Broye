import ee
from typing import List, Dict, Any, Optional, Literal, Union
from dataclasses import dataclass
from enum import Enum


class MosaicType(str, Enum):
    RECENT = "recent"
    LEAST_CLOUDY = "least_cloudy"


class AggregationType(str, Enum):
    GEOMEDIAN = "geomedian"
    MEAN = "mean"
    MAX = "max"
    MIN = "min"
    MOSAIC = "mosaic"
    MEDIAN = "median"


@dataclass
class AggregationOptions:
    band_name: str = "NDVI"
    agg_type: AggregationType = AggregationType.MEDIAN
    mosaic_type: MosaicType = MosaicType.RECENT


def harmonized_ts(
    masked_collection: ee.ImageCollection,
    band_list: List[str],
    time_intervals: List[List[ee.Date]],
    options: Optional[Union[Dict[str, Any], AggregationOptions]] = None,
) -> ee.ImageCollection:
    """
    Generates a harmonized time series from a Sentinel-2 Image Collection while preserving projection.

    Args:
        masked_collection: The Sentinel-2 image collection with applied masks
        band_list: List of band names to include in the aggregation
        time_intervals: List of time intervals for aggregation
        options: Optional parameters as dictionary or AggregationOptions instance

    Returns:
        ee.ImageCollection: A collection of aggregated images with consistent projection
    """
    if isinstance(options, dict):
        options = {**options}  # Create a copy
    else:
        options = options or AggregationOptions()
        options = vars(options)

    first_image = masked_collection.first()
    reference_band = ee.String(band_list[0])
    original_projection = first_image.select(reference_band).projection()
    original_scale = original_projection.nominalScale()

    projection_info = {
        "original_projection": original_projection,
        "original_scale": original_scale,
    }
    masked_collection = masked_collection.set(projection_info)

    def _stack_bands(time_interval: ee.List, stack: ee.List) -> ee.List:
        outputs = aggregate_stack(
            masked_collection, band_list, time_interval, {**options, **projection_info}
        )
        return ee.List(stack).add(ee.Image(outputs))

    stack = ee.List([])
    agg_stack = ee.List(time_intervals).iterate(_stack_bands, stack)

    return (
        ee.ImageCollection(ee.List(agg_stack))
        .map(lambda img: ensure_projection(img, original_projection, original_scale))
        .sort("system:time_start")
    )


def _create_empty_image(
    band_list: List[str],
    original_projection: ee.Projection,
    original_scale: ee.Number,
    timestamp: Dict[str, Any],
) -> ee.Image:
    """Creates an empty image with consistent projection."""
    empty_image = ee.Image.constant(0).rename(band_list[0])
    for band in band_list[1:]:
        empty_image = empty_image.addBands(ee.Image.constant(0).rename(band))
    return (
        empty_image.setDefaultProjection(original_projection, None, original_scale)
        .set(timestamp)
        .float()
    )


def _apply_reducer(
    collection: ee.ImageCollection,
    reducer: ee.Reducer,
    band_list: List[str],
    original_projection: ee.Projection,
    original_scale: ee.Number,
    timestamp: Dict[str, Any],
) -> ee.Image:
    """Applies a reducer with consistent projection."""
    return (
        collection.reduce(reducer)
        .rename(band_list)
        .setDefaultProjection(original_projection)
        .reproject(crs=original_projection, scale=original_scale)
        .set(timestamp)
    )


def _apply_mosaic(
    collection: ee.ImageCollection,
    mosaic_type: MosaicType,
    original_projection: ee.Projection,
    original_scale: ee.Number,
    timestamp: Dict[str, Any],
) -> ee.Image:
    """Applies mosaic operation with consistent projection."""
    if mosaic_type == MosaicType.RECENT:
        mosaic_image = collection.mosaic()
    elif mosaic_type == MosaicType.LEAST_CLOUDY:
        mosaic_image = collection.sort("CLOUDY_PIXEL_PERCENTAGE").mosaic()
    else:
        raise ValueError(f"Invalid mosaic_type: {mosaic_type}")

    return (
        mosaic_image.setDefaultProjection(original_projection)
        .reproject(crs=original_projection, scale=original_scale)
        .set(timestamp)
    )


def aggregate_stack(
    masked_collection: ee.ImageCollection,
    band_list: List[str],
    time_interval: ee.List,
    options: Dict[str, Any],
) -> ee.Image:
    """
    Generates a temporally-aggregated image for a given time interval.
    """
    agg_type = AggregationType(options.get("agg_type", AggregationType.MEDIAN))
    mosaic_type = MosaicType(options.get("mosaic_type", MosaicType.RECENT))
    original_projection = options["original_projection"]
    original_scale = options["original_scale"]

    time_interval = ee.List(time_interval)
    start_date, end_date = [ee.Date(time_interval.get(i)) for i in range(2)]

    timestamp = _calculate_timestamp(start_date, end_date)
    filtered_collection = _filter_collection(
        masked_collection, start_date, end_date, band_list
    )

    if agg_type == AggregationType.MOSAIC:
        return ee.Algorithms.If(
            filtered_collection.size().gt(0),
            _apply_mosaic(
                filtered_collection,
                mosaic_type,
                original_projection,
                original_scale,
                timestamp,
            ),
            _create_empty_image(
                band_list, original_projection, original_scale, timestamp
            ),
        )

    reducer = _get_reducer(agg_type, len(band_list))
    return ee.Algorithms.If(
        filtered_collection.size().gt(0),
        _apply_reducer(
            filtered_collection,
            reducer,
            band_list,
            original_projection,
            original_scale,
            timestamp,
        ),
        _create_empty_image(band_list, original_projection, original_scale, timestamp),
    )


def _calculate_timestamp(start_date: ee.Date, end_date: ee.Date) -> Dict[str, Any]:
    """Calculate timestamp for middle of interval."""
    agg_interval_days = end_date.difference(start_date, "day")
    mid_point = start_date.advance(ee.Number(agg_interval_days.divide(2)).ceil(), "day")
    return {"system:time_start": mid_point.millis()}


def _filter_collection(
    collection: ee.ImageCollection,
    start_date: ee.Date,
    end_date: ee.Date,
    band_list: List[str],
) -> ee.ImageCollection:
    """Filter collection by date range and select bands."""
    return collection.filterDate(start_date, end_date).select(band_list)


def _get_reducer(agg_type: AggregationType, band_count: int) -> ee.Reducer:
    """Get appropriate reducer based on aggregation type."""
    reducers = {
        AggregationType.GEOMEDIAN: ee.Reducer.geometricMedian(band_count),
        AggregationType.MEAN: ee.Reducer.mean(),
        AggregationType.MAX: ee.Reducer.max(),
        AggregationType.MIN: ee.Reducer.min(),
        AggregationType.MEDIAN: ee.Reducer.median(),
    }
    return reducers[agg_type]


def ensure_projection(
    image: ee.Image, target_projection: ee.Projection, target_scale: ee.Number
) -> ee.Image:
    """Ensures an image has the specified projection and scale."""
    return image.reproject(
        crs=target_projection, scale=target_scale
    ).setDefaultProjection(crs=target_projection, scale=target_scale)
