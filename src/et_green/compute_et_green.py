import ee
from typing import Optional, Dict, Any

# Module-level constants
MAX_PIXELS_DEFAULT = int(1e13)
MAX_PIXELS_STATS = int(1e9)
DEFAULT_SCALE = 10
DUMMY_VALUE = 1


def validate_image_band(image: ee.Image, band_name: str) -> ee.Number:
    """
    Validate if a band exists in the image and contains valid data.

    Args:
        image (ee.Image): The input image to validate
        band_name (str): The name of the band to check

    Returns:
        ee.Number: 1 if the band exists and contains valid data, 0 otherwise
    """
    try:
        # Get list of band names
        band_names = ee.List(image.bandNames())

        # Convert boolean to number (1 or 0) for band existence
        has_band = ee.Number(ee.Algorithms.If(band_names.contains(band_name), 1, 0))

        # Get valid pixels count
        valid_pixels = ee.Number(
            image.select(band_name)
            .mask()
            .reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=image.geometry(),
                scale=image.projection().nominalScale(),
                maxPixels=MAX_PIXELS_STATS,
            )
            .get(band_name)
        )

        # Convert valid pixels check to number (1 or 0)
        has_valid_pixels = ee.Number(
            ee.Algorithms.If(ee.Number(valid_pixels).gt(0), 1, 0)
        )

        # Multiply the two numbers
        return has_band.multiply(has_valid_pixels)

    except Exception as e:
        print(f"Validation error: {str(e)}")
        return ee.Number(0)


def create_empty_et_image(
    source_image: ee.Image, band_name: str = "ET_green"
) -> ee.Image:
    """
    Create an empty (fully masked) image with metadata from source image.

    Args:
        source_image (ee.Image): Source image to copy metadata from
        band_name (str): Name for the output band

    Returns:
        ee.Image: Empty image with single band and preserved metadata
    """
    empty = ee.Image.constant(0).rename(band_name)
    empty = empty.updateMask(ee.Image.constant(0))
    empty = empty.setDefaultProjection(
        source_image.projection(), None, source_image.projection().nominalScale()
    )
    return empty.set("system:time_start", source_image.get("system:time_start"))


def compute_feature_mean(
    feature: ee.Feature,
    masked_et: ee.Image,
    overall_mean_et: ee.Number,
    et_band_name: str,
    scale: ee.Number,
    max_pixels: int,
) -> ee.Feature:
    """
    Compute mean ET value for a single feature.

    Args:
        feature (ee.Feature): The feature to process
        masked_et (ee.Image): The masked ET image
        overall_mean_et (ee.Number): Fallback mean ET value
        et_band_name (str): Name of the ET band
        scale (ee.Number): Scale for reduction
        max_pixels (int): Maximum pixels for computation

    Returns:
        ee.Feature: Feature with added mean ET property
    """
    feature_mean = masked_et.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=feature.geometry(),
        scale=scale,
        maxPixels=max_pixels,
    ).get(et_band_name)

    # Use overall mean if feature has no valid ET values
    mean_et = ee.Number(
        ee.Algorithms.If(
            ee.Algorithms.IsEqual(feature_mean, None),
            overall_mean_et,
            feature_mean,
        )
    )

    return feature.set("mean_et", mean_et)



def compute_feature_std(
    feature: ee.Feature,
    masked_et: ee.Image,
    overall_mean_et: ee.Number,
    et_band_name: str,
    scale: ee.Number,
    max_pixels: int,
) -> ee.Feature:
    """
    Compute mean and stdDev ET value for a single feature.

    Args:
        feature (ee.Feature): The feature to process
        masked_et (ee.Image): The masked ET image
        overall_mean_et (ee.Number): Fallback mean ET value
        et_band_name (str): Name of the ET band
        scale (ee.Number): Scale for reduction
        max_pixels (int): Maximum pixels for computation

    Returns:
        ee.Feature: Feature with added mean ET property
    """
    feature_mean = masked_et.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=feature.geometry(),
        scale=scale,
        maxPixels=max_pixels,
    ).get(et_band_name)

    feature_std = masked_et.reduceRegion(
        reducer=ee.Reducer.stdDev(),
        geometry=feature.geometry(),
        scale=scale,
        maxPixels=max_pixels,
    ).get(et_band_name)

    # # Use overall mean if feature has no valid ET values
    # mean_et = ee.Number(
    #     ee.Algorithms.If(
    #         ee.Algorithms.IsEqual(feature_mean, None),
    #         overall_mean_et,
    #         feature_mean,
    #     )
    # )

    return feature.set("mean_et", feature_mean).set("std_et", feature_std)




def compute_valid_et_green(
    et_image: ee.Image,
    rainfed_reference: ee.FeatureCollection,
    feature_collection: ee.FeatureCollection,
    et_band_name: str,
    max_pixels: int,
) -> ee.Image:
    """
    Compute ET green for valid input data.

    Args:
        et_image (ee.Image): Input ET image
        rainfed_reference (ee.FeatureCollection): Rainfed reference areas
        feature_collection (ee.FeatureCollection): Features for computation
        et_band_name (str): Name of the ET band
        max_pixels (int): Maximum pixels for computation

    Returns:
        ee.Image: Computed ET green image
    """
    projection = et_image.projection()
    scale = projection.nominalScale()
    time_start = et_image.get("system:time_start")

    # Add a numeric property to rainfed_reference
    rainfed_ref = rainfed_reference.map(lambda f: f.set("dummy", DUMMY_VALUE))

    # Mask the ET image with rainfed reference areas
    masked_et = et_image.updateMask(
        rainfed_ref.reduceToImage(["dummy"], ee.Reducer.first()).mask()
    )

    # Compute the overall mean ET value (fallback for features without rainfed areas)
    overall_mean_et = ee.Number(
        masked_et.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=feature_collection.geometry(),
            scale=scale,
            maxPixels=max_pixels,
        ).get(et_band_name)
    )

    # Compute mean ET values for each feature
    features_with_mean = feature_collection.map(
        lambda f: compute_feature_mean(
            f, masked_et, overall_mean_et, et_band_name, scale, max_pixels
        )
    )

    # Create an image with ET green values for each feature
    et_green = features_with_mean.reduceToImage(["mean_et"], ee.Reducer.first()).rename(
        "ET_green"
    )

    return et_green.setDefaultProjection(projection, None, scale).set(
        "system:time_start", time_start
    )

def compute_valid_et_green_std(
    et_image: ee.Image,
    rainfed_reference: ee.FeatureCollection,
    feature_collection: ee.FeatureCollection,
    et_band_name: str,
    max_pixels: int,
) -> ee.Image:
    """
    Compute ET green for valid input data.

    Args:
        et_image (ee.Image): Input ET image
        rainfed_reference (ee.FeatureCollection): Rainfed reference areas
        feature_collection (ee.FeatureCollection): Features for computation
        et_band_name (str): Name of the ET band
        max_pixels (int): Maximum pixels for computation

    Returns:
        ee.Image: Computed ET green image
    """
    projection = et_image.projection()
    scale = projection.nominalScale()
    time_start = et_image.get("system:time_start")

    # Add a numeric property to rainfed_reference
    rainfed_ref = rainfed_reference.map(lambda f: f.set("dummy", DUMMY_VALUE))

    # Mask the ET image with rainfed reference areas
    masked_et = et_image.updateMask(
        rainfed_ref.reduceToImage(["dummy"], ee.Reducer.first()).mask()
    )

    # Compute the overall mean ET value (fallback for features without rainfed areas)
    overall_mean_et = ee.Number(
        masked_et.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=feature_collection.geometry(),
            scale=scale,
            maxPixels=max_pixels,
        ).get(et_band_name)
    )

    # Compute mean ET values for each feature
    features_with_mean = feature_collection.map(
        lambda f: compute_feature_std(
            f, masked_et, overall_mean_et, et_band_name, scale, max_pixels
        )
    )

    # Create an image with ET green values for each feature
    et_green = features_with_mean.reduceToImage(["mean_et"], ee.Reducer.first()).rename(
        "ET_green"
    )
    et_green_std = features_with_mean.reduceToImage(["std_et"], ee.Reducer.first()).rename(
        "ET_green_std"
    )
    return et_green.addBands(et_green_std).setDefaultProjection(projection, None, scale).set(
        "system:time_start", time_start
    )

def compute_et_green(
    et_image: ee.Image,
    rainfed_reference: ee.FeatureCollection,
    feature_collection: ee.FeatureCollection,
    et_band_name: str = "downscaled",
    max_pixels: int = MAX_PIXELS_DEFAULT,
) -> ee.Image:
    """
    Compute ET green based on the given ET image and rainfed reference areas for each feature
    in the provided feature collection. Returns an empty image with preserved metadata if
    input validation fails.

    Args:
        et_image (ee.Image): An image containing ET values
        rainfed_reference (ee.FeatureCollection): A feature collection of rainfed reference areas
        feature_collection (ee.FeatureCollection): A feature collection over which to compute the ET green values
        et_band_name (str, optional): The name of the band in the ET image containing the ET values
        max_pixels (int, optional): Maximum number of pixels to process

    Returns:
        ee.Image: An image with a single band 'ET_green' containing the computed ET green values
                 for each feature. Returns an empty (masked) image with preserved metadata if
                 validation fails.
    """
    # Validate inputs and convert result to number (1 or 0)
    is_valid = validate_image_band(et_image, et_band_name)

    # Return either computed ET green or empty image based on validation
    return ee.Image(
        ee.Algorithms.If(
            is_valid,
            compute_valid_et_green(
                et_image,
                rainfed_reference,
                feature_collection,
                et_band_name,
                max_pixels,
            ),
            create_empty_et_image(et_image),
        )
    )


def compute_et_green_std(
    et_image: ee.Image,
    rainfed_reference: ee.FeatureCollection,
    feature_collection: ee.FeatureCollection,
    et_band_name: str = "downscaled",
    max_pixels: int = MAX_PIXELS_DEFAULT,
) -> ee.Image:
    """
    Compute ET green based on the given ET image and rainfed reference areas for each feature
    in the provided feature collection. Returns an empty image with preserved metadata if
    input validation fails.

    Args:
        et_image (ee.Image): An image containing ET values
        rainfed_reference (ee.FeatureCollection): A feature collection of rainfed reference areas
        feature_collection (ee.FeatureCollection): A feature collection over which to compute the ET green values
        et_band_name (str, optional): The name of the band in the ET image containing the ET values
        max_pixels (int, optional): Maximum number of pixels to process

    Returns:
        ee.Image: An image with a single band 'ET_green' containing the computed ET green values
                 for each feature. Returns an empty (masked) image with preserved metadata if
                 validation fails.
    """
    # Validate inputs and convert result to number (1 or 0)
    is_valid = validate_image_band(et_image, et_band_name)

    # Return either computed ET green or empty image based on validation
    return ee.Image(
        ee.Algorithms.If(
            is_valid,
            compute_valid_et_green_std(
                et_image,
                rainfed_reference,
                feature_collection,
                et_band_name,
                max_pixels,
            ),
            create_empty_et_image(et_image).addBands(create_empty_et_image(et_image).rename('ET_green_std')),
        )
    )


def calculate_band_std_dev(
    image: ee.Image,
    band_name: str,
    region: Optional[ee.Geometry] = None,
    scale: float = DEFAULT_SCALE,
    max_pixels: int = MAX_PIXELS_STATS,
) -> float:
    """
    Calculate the standard deviation of values in a specified band of an Earth Engine image.

    Args:
        image (ee.Image): The input Earth Engine image
        band_name (str): The name of the band to analyze
        region (ee.Geometry, optional): The region over which to calculate the standard deviation
        scale (float, optional): The scale in meters of the projection to work in
        max_pixels (int, optional): The maximum number of pixels to sample

    Returns:
        float: The standard deviation of the values in the specified band

    Raises:
        ee.EEException: If the specified band is not found in the image or if the computation fails
    """
    # Select the specified band
    single_band_image = image.select(band_name)

    # If no region is specified, use the image bounds
    if region is None:
        region = single_band_image.geometry()

    # Ensure the region is not null or empty
    region = ee.Algorithms.If(
        ee.Algorithms.IsEqual(region, None), single_band_image.geometry(), region
    )

    try:
        # Calculate standard deviation using reduceRegion
        std_dev_dict = single_band_image.reduceRegion(
            reducer=ee.Reducer.stdDev(),
            geometry=region,
            scale=scale,
            maxPixels=max_pixels,
        )

        # Extract the standard deviation value
        std_dev = std_dev_dict.get(band_name)

        return ee.Number(std_dev)

    except ee.EEException as e:
        print(f"Error calculating standard deviation: {str(e)}")
        raise