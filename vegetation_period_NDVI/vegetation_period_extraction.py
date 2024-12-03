import ee
from typing import List, Dict, Any, Callable
from utils.composites import harmonized_ts
from vegetation_period_NDVI.time_series import get_harmonic_ts

# Constants
NDVI_THRESHOLD = 0.35
NDVI_LOW_THRESHOLD = 0.35
AGG_INTERVAL = 15


def create_binary_ndvi_indicator(img: ee.Image, threshold: float) -> ee.Image:
    """
    Create a binary mask from NDVI values based on a threshold.

    Args:
        img (ee.Image): Input image containing NDVI band.
        threshold (float): NDVI threshold for vegetation.

    Returns:
        ee.Image: Binary mask where 1 indicates NDVI above threshold.
    """
    return (
        img.select(
            "fitted"
        )  # You can use fitted or ndvi band here. I should problably not hardcode this. No one is perfect hey?
        .gt(threshold)
        .rename("vegetation")
        .set("system:time_start", img.get("system:time_start"))
    )


def create_binary_mask(
    ndvi_collection: ee.ImageCollection, threshold: float
) -> ee.ImageCollection:
    """
    Create a binary mask collection from an NDVI image collection.

    Args:
        ndvi_collection (ee.ImageCollection): Collection of images with NDVI bands.
        threshold (float): NDVI threshold for vegetation.

    Returns:
        ee.ImageCollection: Collection of binary mask images.
    """
    return ndvi_collection.map(lambda img: create_binary_ndvi_indicator(img, threshold))


def apply_temporal_operation(
    collection: ee.ImageCollection,
    intervals: ee.List,
    operation: Callable[[ee.ImageCollection], ee.Image],
    direction: int = 1,
) -> ee.Image:
    """
    Apply a temporal operation on an image collection over given intervals.

    Args:
        collection (ee.ImageCollection): Input image collection.
        intervals (ee.List): List of time intervals.
        operation (Callable[[ee.ImageCollection], ee.Image]): Function to apply on each interval.
        direction (int): 1 for forward iteration, -1 for backward.

    Returns:
        ee.Image: Result of the temporal operation.
    """

    def map_func(x):
        interval = intervals.get(ee.Number(x))
        start_date = ee.Date(ee.List(interval).get(0))
        end_date = ee.Date(ee.List(interval).get(1))
        filtered_collection = collection.filterDate(
            start_date, end_date.advance(AGG_INTERVAL, "day")
        )
        return operation(filtered_collection, x)

    sequence = (
        ee.List.sequence(0, intervals.size().subtract(1))
        if direction == 1
        else ee.List.sequence(intervals.size().subtract(1), 0, -1)
    )

    return ee.ImageCollection(sequence.map(map_func)).reduce(ee.Reducer.firstNonNull())


def find_start_first_veg_period(
    fittedGreen: ee.ImageCollection, time_intervals: ee.List
) -> ee.Image:
    """
    Find the start of the first vegetation period.

    Args:
        fittedGreen (ee.ImageCollection): Binary vegetation mask collection.
        time_intervals (ee.List): List of time intervals.

    Returns:
        ee.Image: Image representing the start of the first vegetation period.
    """

    def operation(filtered_collection: ee.ImageCollection, x: ee.Number) -> ee.Image:
        sum_image = filtered_collection.sum()
        return (
            ee.Image(ee.Number(x).subtract(2).max(0)).int().updateMask(sum_image.eq(2))
        )

    return apply_temporal_operation(fittedGreen, time_intervals, operation)


def find_end_first_veg_period(
    fittedGreen: ee.ImageCollection, time_intervals: ee.List
) -> ee.Image:
    """
    Find the end of the first vegetation period.

    Args:
        fittedGreen (ee.ImageCollection): Binary vegetation mask collection.
        time_intervals (ee.List): List of time intervals.

    Returns:
        ee.Image: Image representing the end of the first vegetation period.
    """

    def operation(filtered_collection: ee.ImageCollection, x: ee.Number) -> ee.Image:
        sum_image = filtered_collection.sum()
        return ee.Image(ee.Number(x)).int().updateMask(sum_image.eq(2))

    return apply_temporal_operation(
        fittedGreen, time_intervals, operation, direction=-1
    )

def find_end_first_veg_period_v2(
    fittedGreen: ee.ImageCollection, time_intervals: ee.List, second_start: ee.Image
) -> ee.Image:
    """
    Find the end of the first vegetation period.

    Args:
        fittedGreen (ee.ImageCollection): Binary vegetation mask collection.
        time_intervals (ee.List): List of time intervals.
        second_start (ee.Image): Start of the second vegetation period.

    Returns:
        ee.Image: Image representing the end of the first vegetation period.
    """
    def operation(filtered_collection: ee.ImageCollection, x: ee.Number) -> ee.Image:
        sum_image = filtered_collection.sum()
        valid_second_start = second_start.mask()
        return (
            ee.Image.constant(x)  # Wrap x in an ee.Image
            .int()
            .updateMask(sum_image.eq(2))
            .updateMask(valid_second_start.Not().Or(second_start.gt(ee.Image.constant(x))))  # Handle NaN and valid second_start
        )

    return apply_temporal_operation(
        fittedGreen, time_intervals, operation, direction=-1
    )


def find_end_second_veg_period(
    fittedGreen: ee.ImageCollection, time_intervals: ee.List
) -> ee.Image:
    """
    Find the end of the first vegetation period.

    Args:
        fittedGreen (ee.ImageCollection): Binary vegetation mask collection.
        time_intervals (ee.List): List of time intervals.

    Returns:
        ee.Image: Image representing the end of the first vegetation period.
    """

    def operation(filtered_collection: ee.ImageCollection, x: ee.Number) -> ee.Image:
        sum_image = filtered_collection.sum()
        return ee.Image(ee.Number(x)).int().updateMask(sum_image.eq(2))

    return apply_temporal_operation(
        fittedGreen, time_intervals, operation, direction=-1
    )


def detect_double_cropping(
    fittedGreen: ee.ImageCollection,
    fittedGreenFirst: ee.Image,
    fittedGreenLast: ee.Image,
) -> ee.Image:
    """
    Detect double cropping patterns.

    Args:
        fittedGreen (ee.ImageCollection): Binary vegetation mask collection.
        fittedGreenFirst (ee.Image): Start of the first vegetation period.
        fittedGreenLast (ee.Image): End of the first vegetation period.

    Returns:
        ee.Image: Image representing detected double cropping patterns.
    """

    def map_func(m):
        m = ee.Number(m)
        image1 = ee.Image(fittedGreen.toList(fittedGreen.size()).get(m.subtract(1)))
        image2 = ee.Image(fittedGreen.toList(fittedGreen.size()).get(m))
        image3 = ee.Image(fittedGreen.toList(fittedGreen.size()).get(m.add(1)))
        condition = (
            image1.add(image2)
            .eq(2)
            .And(image2.add(image3).eq(1))
            .multiply(fittedGreenFirst.lte(m).And(fittedGreenLast.gte(m)))
        )
        return condition.multiply(ee.Image(m)).int().updateMask(condition.eq(1))

    return ee.ImageCollection(
        ee.List.sequence(1, fittedGreen.size().subtract(2)).map(map_func)
    ).reduce(ee.Reducer.firstNonNull())


def find_start_second_veg_period(
    fittedGreen: ee.ImageCollection, time_intervals: ee.List, end_first: ee.Image
) -> ee.Image:
    """
    Find the start of the second vegetation period.

    Args:
        fittedGreen (ee.ImageCollection): Binary vegetation mask collection.
        time_intervals (ee.List): List of time intervals.
        end_first (ee.Image): End of the first vegetation period.

    Returns:
        ee.Image: Image representing the start of the second vegetation period.
    """

    def operation(filtered_collection: ee.ImageCollection, x: ee.Number) -> ee.Image:
        sum_image = filtered_collection.sum()
        return (
            ee.Image(ee.Number(x).subtract(2).max(0))
            .int()
            .updateMask(sum_image.eq(2))
            .updateMask(end_first.lte(ee.Number(x).subtract(1)))
        )

    return apply_temporal_operation(fittedGreen, time_intervals, operation)


def confirm_double_cropping(
    fittedGreen_low: ee.ImageCollection,
    time_intervals: ee.List,
    end_first: ee.Image,
    start_second_veg_period: ee.Image,
) -> ee.Image:
    """
    Confirm double cropping by identifying periods of low NDVI between crops.

    Args:
        fittedGreen_low (ee.ImageCollection): Binary vegetation mask collection with lower threshold.
        time_intervals (ee.List): List of time intervals.
        end_first (ee.Image): End of the first vegetation period.
        start_second_veg_period (ee.Image): Start of the second vegetation period.

    Returns:
        ee.Image: Image confirming double cropping patterns.
    """

    def operation(filtered_collection: ee.ImageCollection, x: ee.Number) -> ee.Image:
        sum_image = filtered_collection.sum()
        condition = sum_image.eq(0).multiply(
            end_first.add(ee.Image(1))
            .lte(ee.Number(x))
            .And(start_second_veg_period.add(ee.Image(2)).gt(ee.Number(x)))
        )
        return ee.Image(ee.Number(x)).int().updateMask(condition.eq(1))

    double_cropping = apply_temporal_operation(
        fittedGreen_low, time_intervals, operation
    )
    return double_cropping.mask().updateMask(double_cropping.mask())


def convert_intervals_to_months(
    interval_indices: ee.Image, time_intervals: ee.List
) -> ee.Image:
    """
    Convert interval indices to months using the provided time intervals.

    Args:
        interval_indices (ee.Image): Image containing interval indices.
        time_intervals (ee.List): List of time intervals.

    Returns:
        ee.Image: Image with converted month values.
    """

    # Precompute a list of months from the time_intervals
    months = time_intervals.map(
        lambda interval: ee.Date(ee.List(interval).get(0)).get("month")
    )

    # Remap the interval indices to their corresponding months
    return interval_indices.remap(
        ee.List.sequence(0, time_intervals.size().subtract(1)),
        months
    )


def combine_results(
    first_start: ee.Image,
    first_end: ee.Image,
    second_start: ee.Image,
    second_end: ee.Image,
    double_cropping: ee.Image,
    time_intervals: ee.List,
) -> ee.Image:
    """
    Combine all results into a single multi-band image.

    Args:
        first_start (ee.Image): Start of the first vegetation period.
        first_end (ee.Image): End of the first vegetation period.
        second_start (ee.Image): Start of the second vegetation period.
        second_end (ee.Image): End of the second vegetation period.
        double_cropping (ee.Image): Double cropping indicator.
        time_intervals (ee.List): List of time intervals.

    Returns:
        ee.Image: Multi-band image with all vegetation period information.
    """
    result = (
        first_start.rename("firstStart")
        .addBands(first_end.rename("firstEnd"))
        .addBands(
            second_start.rename("secondStart")
            .addBands(second_end.rename("secondEnd"))
            .updateMask(double_cropping.mask())
        )
    )

    start_month = ee.Date(ee.List(time_intervals.get(0)).get(0)).get("month")
    result = result.divide(2).floor().add(ee.Image.constant(start_month))
    result = result.where(result.gt(12), result.subtract(12))

    return result.addBands(double_cropping.gt(0).rename("isDoubleCropping"))

def combine_results_v2(
    first_start: ee.Image,
    first_end: ee.Image,
    double_crop_end: ee.Image,
    second_start: ee.Image,
    second_end: ee.Image,
    double_cropping: ee.Image,
    time_intervals: ee.List,
) -> ee.Image:
    """
    Combine all results into a single multi-band image.

    Args:
        first_start (ee.Image): Start of the first vegetation period.
        first_end (ee.Image): End of the first vegetation period.
        double_crop_end (ee.Image): End of the first vegetation period in case of double cropping.
        second_start (ee.Image): Start of the second vegetation period.
        second_end (ee.Image): End of the second vegetation period.
        double_cropping (ee.Image): Double cropping indicator.
        time_intervals (ee.List): List of time intervals.

    Returns:
        ee.Image: Multi-band image with all vegetation period information.
    """
    # Replace first_end with double_crop_end if double cropping is detected
    adjusted_first_end = first_end.where(double_cropping, double_crop_end)

    # Construct the result image
    result = (
        first_start.rename("firstStart")
        .addBands(adjusted_first_end.rename("firstEnd"))
        .addBands(
            second_start.rename("secondStart")
            .addBands(second_end.rename("secondEnd"))
            .updateMask(double_cropping.mask())
        )
    )

    # Add additional temporal adjustments based on start_month
    start_month = ee.Date(ee.List(time_intervals.get(0)).get(0)).get("month")
    result = result.divide(2).floor().add(ee.Image.constant(start_month))
    result = result.where(result.gt(12), result.subtract(12))

    return result.addBands(double_cropping.gt(0).rename("isDoubleCropping"))

def get_crop_veg_period(
    year: int, aoi: ee.Geometry, time_intervals: ee.List
) -> ee.Image:
    """
    Extract crop vegetation periods for a given year and area.

    Args:
        year (int): Year of analysis.
        aoi (ee.Geometry): Area of interest.
        time_intervals (ee.List): List of time intervals for analysis.

    Returns:
        ee.Image: Multi-band image containing vegetation period information.
    """
    harmonic_ts_dictionary = get_harmonic_ts(year, aoi, time_intervals)
    ndvi_collection = harmonic_ts_dictionary.get("fitted_data")

    veg_mask = create_binary_mask(ndvi_collection, NDVI_THRESHOLD)
    veg_mask_low = create_binary_mask(ndvi_collection, NDVI_LOW_THRESHOLD)

    first_period_start = find_start_first_veg_period(veg_mask, time_intervals)
    first_period_end = find_end_first_veg_period(veg_mask, time_intervals)

    double_crop_end = detect_double_cropping(
        veg_mask, first_period_start, first_period_end
    )

    # Convert double_crop_end indices to months (not necessary, they are already months)
    # double_crop_end_month = convert_intervals_to_months(double_crop_end, time_intervals)

    second_period_start = find_start_second_veg_period(
        veg_mask, time_intervals, double_crop_end
    )

    double_cropping = confirm_double_cropping(
        veg_mask_low, time_intervals, double_crop_end, second_period_start
    )
    return combine_results_v2(
        first_period_start,
        first_period_end,
        double_crop_end, # Use this instead of first_period_end if there is double cropping
        second_period_start,
        first_period_end,  # Using first_period_end as second_period_end
        double_cropping,
        time_intervals,
    )
