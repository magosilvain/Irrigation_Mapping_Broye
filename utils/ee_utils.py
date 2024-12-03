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
    """Creates an empty (masked) image with consistent projection."""
    # Create first band as masked
    empty_image = ee.Image.constant(0).rename(band_list[0]).mask(ee.Image(0))

    # Add additional bands, all masked
    for band in band_list[1:]:
        empty_image = empty_image.addBands(
            ee.Image.constant(0).rename(band).mask(ee.Image(0))
        )

    # Set proper projection and timestamp
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
    """
    Applies mosaic operation with consistent projection.

    Args:
        collection: Input image collection
        mosaic_type: Type of mosaic to create
        original_projection: Target projection
        original_scale: Target scale
        timestamp: Timestamp to set on output image

    Returns:
        ee.Image: Mosaicked image with consistent projection
    """
    if mosaic_type == MosaicType.RECENT:
        mosaic_image = collection.mosaic()
    elif mosaic_type == MosaicType.LEAST_CLOUDY:
        mosaic_image = collection.sort("CLOUDY_PIXEL_PERCENTAGE").mosaic()
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
    band_name = options.get("band_name")  # Get band_name from options

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
                # band_name,
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


def set_negative_to_zero(image: ee.Image) -> ee.Image:
    """
    Set all negative values in an image to zero.

    Args:
        image (ee.Image): The input image.

    Returns:
        ee.Image: The image with all negative values set to zero.
    """
    return image.where(image.lt(0), 0)


def merge_collections(
    years: List[int], asset_name: str, special_char: Optional[str] = None
) -> ee.ImageCollection:
    """
    Merge Earth Engine ImageCollections for multiple years.

    Args:
        years: List of years to process
        asset_name: Base name of the asset to merge
        special_char: Optional character to append to asset name

    Returns:
        Merged ImageCollection with consistent projection
    """
    if not years:
        raise ValueError("Years list cannot be empty")

    collections = []
    for year in years:
        path = f"{asset_name}_{year}"
        if special_char:
            path = f"{path}_{special_char}"

        collection = ee.ImageCollection(path)
        collections.append(collection)

    # Get projection from first collection
    projection = collections[0].first().projection()

    # Apply projection and merge
    collections = [
        collection.map(lambda img: img.setDefaultProjection(projection))
        for collection in collections
    ]

    merged = collections[0]
    for collection in collections[1:]:
        merged = merged.merge(collection)

    return merged


def extract_pixel_values(
    image_collection: ee.ImageCollection,
    point: ee.Geometry.Point,
    band: str = "downscaled",
) -> ee.FeatureCollection:
    """
    Extract the pixel value of the specified band for each image in the collection
    at the specified point, with error handling for missing timestamps.

    Args:
        image_collection (ee.ImageCollection): The input image collection.
        point (ee.Geometry.Point): The point at which to extract values.
        band (str): The band to extract values from. Defaults to 'downscaled'.

    Returns:
        ee.FeatureCollection: A feature collection where each feature represents an image
                              and contains the pixel value of the "band" at the point.
    """

    def extract_value(image: ee.Image) -> ee.Feature:
        # Select the specified band
        image_band = image.select(band)

        # Get the scale of the specified band
        scale = image_band.projection().nominalScale()

        # Extract the pixel value at the point
        pixel_value = image_band.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=scale,
            bestEffort=True,
        ).get(band)

        # Retrieve the image acquisition time
        time_start = image.get("system:time_start")

        # Handle potential null time_start
        formatted_date = ee.Algorithms.If(
            ee.Algorithms.IsEqual(time_start, None),
            None,
            ee.Date(time_start).format("YYYY-MM-dd"),
        )

        return ee.Feature(
            None,
            {
                "pixel_value": pixel_value,
                "date": formatted_date,
                "system:time_start": time_start,
            },
        )

    # Map the extraction function over the image collection
    return ee.FeatureCollection(image_collection.map(extract_value))


def aggregate_to_monthly(
    collection: ee.ImageCollection, bands: List[str] = ["downscaled"]
) -> ee.ImageCollection:
    """
    Aggregate an image collection to monthly images, weighted by the number of days each image represents.

    Args:
        collection (ee.ImageCollection): Input collection.
        bands (List[str]): List of band names to aggregate. Defaults to ["downscaled"].

    Returns:
        ee.ImageCollection: Monthly aggregated image collection.
    """

    def aggregate_month(year, month, images):
        images = ee.List(images)
        start_date = ee.Date.fromYMD(year, month, 1)
        end_date = start_date.advance(1, "month")
        days_in_month = end_date.difference(start_date, "day")

        def weight_image(i):
            i = ee.Number(i)
            image = ee.Image(images.get(i))
            next_image = ee.Image(images.get(i.add(1)))
            date = ee.Date(image.get("system:time_start"))
            next_date = ee.Date(
                ee.Algorithms.If(
                    i.eq(images.size().subtract(1)),
                    end_date,
                    next_image.get("system:time_start"),
                )
            )
            weight = next_date.difference(date, "day")
            # Cast the selected bands to a consistent float type
            return (
                image.select(bands)
                .cast(
                    ee.Dictionary.fromLists(bands, ee.List.repeat("float", len(bands)))
                )
                .multiply(weight)
            )

        weighted_sum = ee.ImageCollection.fromImages(
            ee.List.sequence(0, images.size().subtract(1)).map(weight_image)
        ).sum()

        return weighted_sum.set(
            {"system:time_start": start_date.millis(), "year": year, "month": month}
        )

    # Get unique year-month combinations
    dates = collection.aggregate_array("system:time_start")
    unique_year_months = dates.map(lambda d: ee.Date(d).format("YYYY-MM")).distinct()

    def process_year_month(ym):
        ym = ee.String(ym)
        year = ee.Number.parse(ym.slice(0, 4))
        month = ee.Number.parse(ym.slice(5, 7))
        start_date = ee.Date.fromYMD(year, month, 1)
        end_date = start_date.advance(1, "month")

        monthly_images = collection.filterDate(start_date, end_date)
        return aggregate_month(
            year, month, monthly_images.toList(monthly_images.size())
        )

    aggregated = ee.ImageCollection.fromImages(
        unique_year_months.map(process_year_month)
    )

    projection = collection.first().projection()
    scale = projection.nominalScale()

    # Ensure consistent float type for the entire collection
    return aggregated.map(
        lambda img: img.cast(
            ee.Dictionary.fromLists(bands, ee.List.repeat("float", len(bands)))
        ).setDefaultProjection(projection, None, scale)
    ).sort("system:time_start")


def back_to_float(
    image: ee.Image,
    scale: int,
    dynamic: bool = False,
    scaling_factor_property: str = None,
) -> ee.Image:
    """
    Convert an image to float with either static or dynamic scaling.

    Args:
        image: The image to convert
        scale: The default scale to divide by when dynamic=False
        dynamic: If True, reads scale from image property
        scaling_factor_property: Name of image property containing scale factor

    Returns:
        The image converted to float and divided by appropriate scale

    Raises:
        ValueError: If dynamic=True but scaling_factor_property is None
    """
    if dynamic and not scaling_factor_property:
        raise ValueError("scaling_factor_property must be specified when dynamic=True")

    date = image.get("system:time_start")

    if dynamic:
        scale_factor = ee.Number(image.get(scaling_factor_property))
        return image.toFloat().divide(scale_factor).set("system:time_start", date)

    return image.toFloat().divide(scale).set("system:time_start", date)


def back_to_int(image: ee.Image, scale: int) -> ee.Image:
    """
    Convert an image to int and multiply by the scale

    Args:
        image: The image to convert
        scale: The scale to multiply by

    Returns:
        The image converted to int and multiplied by the scale
    """
    date = image.get("system:time_start")
    return image.multiply(scale).toInt().set("system:time_start", date)


def export_image_to_asset(
    image: ee.Image,
    asset_id: str,
    task_name: str,
    year: int,
    aoi: ee.Geometry,
    max_pixels: int = 1e13,
    crs: str = "EPSG:4326",
    scale: int = 10,
) -> ee.batch.Task:
    """
    Export an image to an Earth Engine asset with consistent projection and scale.

    Args:
        image (ee.Image): Image to export
        asset_id (str): Destination asset ID
        task_name (str): Name of the export task
        year (int): Year of the data
        aoi (ee.Geometry): Area of interest for export
        max_pixels (int, optional): Maximum number of pixels to export. Defaults to 1e13.
        crs (str, optional): Coordinate reference system. Defaults to "EPSG:4326".
        scale (int, optional): Resolution in meters. Defaults to 10.

    Returns:
        ee.batch.Task: The export task
    """
    # Ensure consistent projection for export
    image_to_export = image.setDefaultProjection(crs=crs, scale=scale)

    task = ee.batch.Export.image.toAsset(
        image=image_to_export,
        description=task_name,
        assetId=asset_id,
        region=aoi,
        crs=crs,
        scale=scale,
        maxPixels=max_pixels,
    )

    print(f"Exporting {task_name} for {year} to {asset_id}")
    task.start()
    return task


def export_feature_collection(
    collection: ee.FeatureCollection, task_name: str, asset_id: str
):
    """
    Export the feature collection to an Earth Engine asset.

    Args:
        collection: The feature collection to export
        task_name: The name of the export task
        asset_id: The asset ID to export to
    """
    task = ee.batch.Export.table.toAsset(
        collection=collection,
        description=task_name,
        assetId=asset_id,
    )
    task.start()


def print_value_ranges(
    collection: ee.ImageCollection, band_name: str = "ET_blue"
) -> None:
    """
    Print the minimum and maximum values for each image in the collection.

    Args:
        collection (ee.ImageCollection): Collection of images to analyze
        band_name (str): Name of the band to analyze
    """

    def get_minmax(image):
        stats = image.select(band_name).reduceRegion(
            reducer=ee.Reducer.minMax(),
            geometry=image.geometry(),
            scale=30,
            maxPixels=1e9,
        )
        return image.set(
            {"min": stats.get(f"{band_name}_min"), "max": stats.get(f"{band_name}_max")}
        )

    # Map the minmax computation over the collection
    collection_with_stats = collection.map(get_minmax)

    # Get the stats as lists
    stats = (
        collection_with_stats.aggregate_array("min")
        .zip(collection_with_stats.aggregate_array("max"))
        .getInfo()
    )

    # Print results
    for i, (min_val, max_val) in enumerate(stats):
        print(f"Image {i + 1}: Min = {min_val:.2f}, Max = {max_val:.2f}")


def is_image_empty(image: ee.Image) -> bool:
    """
    Check if an image is empty (all bands are masked).

    Args:
        image (ee.Image): The image to check.

    Returns:
        bool: True if the image is empty (fully masked), False otherwise.
    """
    # Get all band names
    band_names = image.bandNames()

    # Create a combined mask across all bands (1 where any band has data)
    combined_mask = image.mask().reduce(ee.Reducer.anyNonZero())

    # Count pixels with valid data using reduceRegion
    valid_pixels = combined_mask.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=image.geometry(),
        scale=image.projection().nominalScale(),
        maxPixels=1e9,
    ).get("any")

    # Convert the computed sum to a number and check if it equals 0
    return ee.Number(valid_pixels).eq(0).getInfo()


def fill_gaps_with_zeros(image: ee.Image) -> ee.Image:
    """
    Fills gaps in an image with zeros.

    Args:
        image (ee.Image): Image to fill gaps in

    Returns:
        ee.Image: Image with gaps filled
    """
    return image.unmask(0)


def normalize_string_client(s: str) -> str:
    """
    Normalize strings on client side for the exclusion and rainfed sets.
    Replaces German umlauts with their ASCII equivalents.
    """
    replacements = {
        "ä": "ae",
        "ö": "oe",
        "ü": "ue",
        "ß": "ss",
        "Ä": "Ae",
        "Ö": "Oe",
        "Ü": "Ue",
    }

    for old, new in replacements.items():
        s = s.replace(old, new)

    return s


def normalize_string_server(ee_string: ee.String) -> ee.String:
    """
    Normalize strings server side using ee.String.replace().
    Must be compatible with client-side normalization.
    """
    return (
        ee_string.replace("ä", "ae", "g")
        .replace("ö", "oe", "g")
        .replace("ü", "ue", "g")
        .replace("ß", "ss", "g")
        .replace("Ä", "Ae", "g")
        .replace("Ö", "Oe", "g")
        .replace("Ü", "Ue", "g")
    )