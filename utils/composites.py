import ee
from typing import List, Dict, Any, Optional


def harmonized_ts(
    masked_collection: ee.ImageCollection,
    band_list: List[str],
    time_intervals: List[List[ee.Date]],
    options: Optional[Dict[str, Any]] = None,
) -> ee.ImageCollection:
    """
    Generates a harmonized time series from a Sentinel-2 Image Collection while preserving projection.

    Args:
        masked_collection (ee.ImageCollection): The Sentinel-2 image collection with applied masks.
        band_list (List[str]): List of band names to include in the aggregation.
        time_intervals (List[List[ee.Date]]): List of time intervals for aggregation.
        options (Optional[Dict[str, Any]]): Optional parameters for aggregation.

    Returns:
        ee.ImageCollection: A collection of aggregated images with consistent projection.
    """
    options = options or {}
    band_name = options.get("band_name", "NDVI")
    agg_type = options.get("agg_type", "median")
    mosaic_type = options.get("mosaic_type", "recent")

    # Get projection from first image to ensure consistency
    first_image = masked_collection.first()
    reference_band = ee.String(band_list[0])
    original_projection = first_image.select(reference_band).projection()
    original_scale = original_projection.nominalScale()

    # Store projection information in the collection properties
    masked_collection = masked_collection.set(
        {"original_projection": original_projection, "original_scale": original_scale}
    )

    time_intervals = ee.List(time_intervals)

    def _stack_bands(time_interval, stack):
        outputs = aggregate_stack(
            masked_collection,
            band_list,
            time_interval,
            {
                "agg_type": agg_type,
                "band_name": band_name,
                "mosaic_type": mosaic_type,
                "original_projection": original_projection,
                "original_scale": original_scale,
            },
        )
        return ee.List(stack).add(ee.Image(outputs))

    stack = ee.List([])
    agg_stack = ee.List(time_intervals).iterate(_stack_bands, stack)

    # Ensure consistent projection in the final collection
    return (
        ee.ImageCollection(ee.List(agg_stack))
        .map(lambda img: ensure_projection(img, original_projection, original_scale))
        .sort("system:time_start")
    )


def ensure_projection(
    image: ee.Image, target_projection: ee.Projection, target_scale: ee.Number
) -> ee.Image:
    """
    Ensures an image has the specified projection and scale.

    Args:
        image (ee.Image): Input image.
        target_projection (ee.Projection): Target projection.
        target_scale (ee.Number): Target scale in meters.

    Returns:
        ee.Image: Image with enforced projection and scale.
    """
    return image.reproject(
        crs=target_projection, scale=target_scale
    ).setDefaultProjection(crs=target_projection, scale=target_scale)


def aggregate_stack(
    masked_collection: ee.ImageCollection,
    band_list: List[str],
    time_interval: ee.List,
    options: Dict[str, Any],
) -> ee.Image:
    """
    Generates a temporally-aggregated image for a given time interval with consistent projection.

    Args:
        masked_collection (ee.ImageCollection): The Sentinel-2 image collection with applied masks.
        band_list (List[str]): List of band names to include in the aggregation.
        time_interval (ee.List): A list containing start and end ee.Date objects.
        options (Dict[str, Any]): Optional parameters including projection information.

    Returns:
        ee.Image: An aggregated image with consistent projection.
    """
    band_name = options.get("band_name", "NDVI")
    agg_type = options.get("agg_type", "median")
    mosaic_type = options.get("mosaic_type", "recent")
    original_projection = options.get("original_projection")
    original_scale = options.get("original_scale")

    time_interval = ee.List(time_interval)
    start_date = ee.Date(time_interval.get(0))
    end_date = ee.Date(time_interval.get(1))

    # Set timestamp for middle of interval
    agg_interval_days = end_date.difference(start_date, "day")
    timestamp = {
        "system:time_start": start_date.advance(
            ee.Number(agg_interval_days.divide(2)).ceil(), "day"
        ).millis()
    }

    # Filter collection and select bands
    filtered_collection = masked_collection.filterDate(start_date, end_date).select(
        band_list
    )

    def create_empty_image():
        """Creates an empty image with consistent projection."""
        empty_image = ee.Image.constant(0).rename(band_list[0])
        for band in band_list[1:]:
            empty_image = empty_image.addBands(ee.Image.constant(0).rename(band))
        return (
            empty_image.setDefaultProjection(original_projection, None, original_scale)
            .set(timestamp)
            .float()
        )

    def apply_reducer(reducer):
        """Applies a reducer with consistent projection."""
        return (
            filtered_collection.reduce(reducer)
            .rename(band_list)
            .setDefaultProjection(original_projection)
            .reproject(crs=original_projection, scale=original_scale)
            .set(timestamp)
        )

    def apply_mosaic():
        """Applies mosaic operation with consistent projection."""
        if mosaic_type == "recent":
            mosaic_image = filtered_collection.mosaic()
        elif mosaic_type == "least_cloudy":
            mosaic_image = filtered_collection.sort("CLOUDY_PIXEL_PERCENTAGE").mosaic()
        else:
            raise ValueError(f"Invalid mosaic_type: {mosaic_type}")

        return (
            mosaic_image.setDefaultProjection(original_projection)
            .reproject(crs=original_projection, scale=original_scale)
            .set(timestamp)
        )

    # Select and apply appropriate reducer
    if agg_type == "geomedian":
        reducer = ee.Reducer.geometricMedian(len(band_list))
    elif agg_type == "mean":
        reducer = ee.Reducer.mean()
    elif agg_type == "max":
        reducer = ee.Reducer.max()
    elif agg_type == "min":
        reducer = ee.Reducer.min()
    elif agg_type == "mosaic":
        return ee.Algorithms.If(
            filtered_collection.size().gt(0), apply_mosaic(), create_empty_image()
        )
    else:  # default to median
        reducer = ee.Reducer.median()

    return ee.Algorithms.If(
        filtered_collection.size().gt(0), apply_reducer(reducer), create_empty_image()
    )
