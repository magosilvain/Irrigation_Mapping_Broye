# Original js scripts are found at https://earthengine.googlesource.com/users/soilwatch/soilErosionApp/


import ee
from typing import Any, Callable, Dict

# Initialize the Earth Engine module.
ee.Initialize()

# Global Cloud Masking Parameters
CLD_PRB_THRESH: int = (
    40  # Cloud probability threshold to mask clouds. 40% is the default value of s2cloudless
)
CLOUD_FILTER: int = (
    60  # Threshold on Sentinel-2 Metadata field determining cloud pixel percentage in image
)
NIR_DRK_THRESH: float = (
    0.15  # Threshold to determine when to consider a dark area a cloud shadow
)
CLD_PRJ_DIST: int = (
    10  # Distance (in number of pixels) to search from detected cloud to find cloud shadows
)
BUFFER: int = (
    50  # Cloud buffer (in meters) around detected cloud pixels to mask additionally
)
MASK_RES: int = (
    60  # Resolution at which to generate and apply the cloud/shadow mask (meters)
)


def load_image_collection(
    collection_name: str, time_range: Dict[str, ee.Date], geom: ee.Geometry
) -> ee.ImageCollection:
    """
    Loads Sentinel-2 Image Collection with corresponding cloud probability information.

    This function imports Sentinel-2 Level-1C or Level-2A data based on the provided collection name,
    filters it by date range and geometry, and joins it with the s2cloudless cloud probability
    data.

    Args:
        collection_name (str): Name of the Sentinel-2 Image Collection (e.g., 'COPERNICUS/S2').
        time_range (Dict[str, ee.Date]): Dictionary with 'start' and 'end' ee.Date objects defining the time range.
        geom (ee.Geometry): Geometry defining the area of interest.

    Returns:
        ee.ImageCollection: A joined ImageCollection of Sentinel-2 images with cloud probability data.
    """
    # Import Sentinel-2 Image Collection and apply filters
    s2 = (
        ee.ImageCollection(collection_name)
        .filterDate(time_range.get("start"), time_range.get("end"))
        .filterBounds(geom)
        .filterMetadata("CLOUDY_PIXEL_PERCENTAGE", "less_than", CLOUD_FILTER)
    )

    # Import and filter s2cloudless Image Collection
    s2_cloudless_col = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filterBounds(geom)
        .filterDate(time_range.get("start"), time_range.get("end"))
    )

    # Define the join condition based on 'system:index'
    join = ee.Join.saveFirst("s2cloudless")
    join_condition = ee.Filter.equals(
        leftField="system:index", rightField="system:index"
    )

    # Apply the join to combine Sentinel-2 and s2cloudless collections
    s2_cl = ee.ImageCollection(
        join.apply(primary=s2, secondary=s2_cloudless_col, condition=join_condition)
    ).sort("system:time_start")

    return s2_cl


def add_cloud_shadow_mask(
    water_valmask: ee.Image, sr_band_scale: float = 1.0
) -> Callable[[ee.Image], ee.Image]:
    """
    Creates a function to add cloud and shadow masks to Sentinel-2 images.

    This function returns a wrapper that can be mapped over an ImageCollection to add
    cloud and shadow mask bands based on specified parameters.

    Args:
        water_valmask (ee.Image): Water validity mask indicating locations of non-water pixels for cloud shadow detection.
        sr_band_scale (float, optional): Scaling factor for Sentinel-2 surface reflectance bands. Defaults to 1.0.

    Returns:
        Callable[[ee.Image], ee.Image]: A function that takes an ee.Image and returns it with cloud and shadow masks added.
    """

    def wrapper(img: ee.Image) -> ee.Image:
        """
        Adds cloud and shadow masks to a Sentinel-2 image.

        Args:
            img (ee.Image): A Sentinel-2 image.

        Returns:
            ee.Image: The input image with added cloud and shadow mask bands.
        """
        # Add cloud component bands
        img_cloud = _add_cloud_bands(img)

        # Add cloud shadow component bands
        img_cloud_shadow = _add_shadow_bands(img_cloud, water_valmask, sr_band_scale)

        # Combine cloud and shadow masks, setting cloud and shadow pixels to 1, else 0
        is_cld_shdw = (
            img_cloud_shadow.select("clouds")
            .add(img_cloud_shadow.select("shadows"))
            .gt(0)
        )

        # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input
        is_cld_shdw = (
            is_cld_shdw.focal_min(2)
            .focal_max(BUFFER * 2 / MASK_RES)
            .reproject(crs=img.select([0]).projection(), scale=MASK_RES)
            .rename("cloudmask")
        )

        # Add the final cloud-shadow mask to the image
        return img_cloud_shadow.addBands(is_cld_shdw)

    return wrapper


def apply_cloud_shadow_mask(img: ee.Image) -> ee.Image:
    """
    Applies the final cloud and shadow mask to a Sentinel-2 image.

    This function masks out cloud and shadow pixels based on the 'cloudmask' band,
    setting those pixels to 0 and others to 1.

    Args:
        img (ee.Image): A Sentinel-2 image with a 'cloudmask' band.

    Returns:
        ee.Image: The input image with cloud and shadow areas masked out.
    """
    # Invert the 'cloudmask' band: clouds/shadows are 0, else 1
    not_cld_shdw = img.select("cloudmask").Not()

    # Update the mask of the image with the inverted cloudmask
    return img.updateMask(not_cld_shdw)


def _add_cloud_bands(img: ee.Image) -> ee.Image:
    """
    Adds cloud probability and cloud mask bands to a Sentinel-2 image.

    This helper function retrieves the cloud probability from the joined s2cloudless data,
    applies a threshold to determine cloud presence, and adds these as new bands to the image.

    Args:
        img (ee.Image): A Sentinel-2 image with joined s2cloudless data.

    Returns:
        ee.Image: The input image with added 'probability' and 'clouds' bands.
    """
    # Retrieve the s2cloudless image and select the 'probability' band
    cld_prb = ee.Image(img.get("s2cloudless")).select("probability")

    # Create a 'clouds' band by thresholding the cloud probability
    is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename("clouds")

    # Add the cloud probability and cloud mask bands to the image
    return img.addBands([cld_prb.rename("probability"), is_cloud])


def _add_shadow_bands(
    img: ee.Image, water_valmask: ee.Image, sr_band_scale: float
) -> ee.Image:
    """
    Adds cloud shadow detection bands to a Sentinel-2 image.

    This helper function identifies potential cloud shadow pixels based on dark NIR areas,
    projects shadows from detected clouds, and adds related bands to the image.

    Args:
        img (ee.Image): A Sentinel-2 image with cloud bands added.
        water_valmask (ee.Image): Water validity mask indicating non-water pixels for shadow detection.
        sr_band_scale (float): Scaling factor for Sentinel-2 surface reflectance bands.

    Returns:
        ee.Image: The input image with added 'dark_pixels', 'cloud_transform', and 'shadows' bands.
    """
    # Identify dark NIR pixels that are not water (potential cloud shadow pixels)
    dark_pixels = (
        img.select("B8")
        .lt(NIR_DRK_THRESH * sr_band_scale)
        .multiply(water_valmask)
        .rename("dark_pixels")
    )

    # Determine the direction to project cloud shadows based on solar azimuth angle
    shadow_azimuth = ee.Number(90).subtract(
        ee.Number(img.get("MEAN_SOLAR_AZIMUTH_ANGLE"))
    )

    # Project shadows from clouds using directional distance transform
    cld_proj = (
        img.select("clouds")
        .directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST / 10)
        .reproject(crs=img.select(0).projection(), scale=MASK_RES)
        .select("distance")
        .mask()
        .rename("cloud_transform")
    )

    # Identify shadows by intersecting dark pixels with cloud projection
    shadows = cld_proj.multiply(dark_pixels).rename("shadows")

    # Add 'dark_pixels', 'cloud_transform', and 'shadows' bands to the image
    return img.addBands([dark_pixels, cld_proj, shadows])


def add_geos3_mask(img: ee.Image) -> ee.Image:
    """
    Applies the GEOS3 algorithm to add a bare soil mask to a Sentinel-2 image.

    The GEOS3 algorithm identifies bare soil pixels based on vegetation indices and spectral characteristics.

    Args:
        img (ee.Image): A Sentinel-2 image.

    Returns:
        ee.Image: An image with the 'GEOS3' bare soil mask band.
    """
    # Rescale the image to [0, 1] reflectance
    img_rs = img.divide(10000).float()

    # Calculate Normalized Difference Vegetation Index (NDVI)
    ndvi = img_rs.normalizedDifference(["B8", "B4"]).rename("NDVI")

    # Calculate Normalized Burn Ratio 2 (NBR2)
    nbr2 = img_rs.normalizedDifference(["B11", "B12"]).rename("NBR2")

    # Calculate Visible-to-Shortwave-Infrared Tendency Index (VNSIR)
    vnsir = (
        ee.Image(1)
        .subtract(
            ee.Image(2)
            .multiply(img_rs.select("B4"))
            .subtract(img_rs.select("B3"))
            .subtract(img_rs.select("B2"))
            .add(
                ee.Image(3).multiply(img_rs.select("B12").subtract(img_rs.select("B8")))
            )
        )
        .rename("VNSIR")
    )

    # Apply GEOS3 conditions to identify bare soil pixels
    geos3 = (
        ndvi.gte(-0.25)
        .And(ndvi.lte(0.25))
        .And(nbr2.gte(-0.3))
        .And(nbr2.lte(0.1))
        .And(vnsir.lte(0.9))
        .rename("GEOS3")
    )

    return geos3
