import ee
from typing import Dict, Any
from utils.s2_mask import load_image_collection, add_cloud_shadow_mask


def load_sentinel2_data(year: int, aoi: ee.Geometry) -> ee.ImageCollection:
    """
    Load Sentinel-2 data for a given year and area of interest, applying cloud and shadow masks.
    Ensures consistent projections across all bands.

    Args:
        year (int): The year for which to load the data.
        aoi (ee.Geometry): The area of interest.

    Returns:
        ee.ImageCollection: The processed Sentinel-2 image collection with consistent projections.
    """
    start_date = ee.Date.fromYMD(year, 1, 1)
    end_date = ee.Date.fromYMD(year, 12, 31)

    s2_filtered = load_image_collection(
        "COPERNICUS/S2_HARMONIZED", {"start": start_date, "end": end_date}, aoi
    ).filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50))

    # Water mask in native resolution
    not_water = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("max_extent").eq(0)

    def process_image(img: ee.Image) -> ee.Image:
        # Get projection from a reference band (B2 is 10m resolution)
        projection = img.select("B2").projection()

        # Project water mask to match Sentinel-2
        not_water_projected = not_water.reproject(projection)

        masked = add_cloud_shadow_mask(not_water_projected)(img)
        with_vars = add_variables(masked)

        # Add time data with consistent projection
        date = ee.Date(img.get("system:time_start"))
        years = date.difference(ee.Date("1970-01-01"), "year")

        time_band = ee.Image(years).float().rename("t").reproject(projection)

        constant_band = ee.Image.constant(1).reproject(projection)

        # Add bands and ensure all have same projection
        result = with_vars.addBands(time_band).addBands(constant_band)

        return result.reproject(projection)

    return s2_filtered.map(process_image).set("sensor_id", 0)


def add_variables(image: ee.Image) -> ee.Image:
    """
    Add NDVI and LSWI bands to the image, and apply cloud masking.
    Ensures consistent projections.

    Args:
        image (ee.Image): Input Sentinel-2 image.

    Returns:
        ee.Image: Image with added bands and consistent projections.
    """
    # Get reference projection from B2 band
    projection = image.select("B2").projection()

    ndvi = (
        image.normalizedDifference(["B8", "B4"])
        .rename("NDVI")
        .toFloat()
        .reproject(projection)
    )

    lswi = (
        image.normalizedDifference(["B8", "B11"]).rename("LSWI").reproject(projection)
    )

    # Create cloud mask with consistent projection
    cloud_mask = (
        ee.Image(1)
        .where(
            image.select("QA60").lt(1024),
            ee.Image(1).where(image.select("B1").gt(0), 0),
        )
        .rename("cloud")
        .where(ndvi.gt(0.99), 1)
        .reproject(projection)
    )

    return (
        image.addBands(ndvi)
        .addBands(lswi)
        .updateMask(cloud_mask.Not())
        .addBands(cloud_mask.multiply(ee.Image(100)))
        .set({"CLOUD_COVER": image.get("CLOUDY_PIXEL_PERCENTAGE")})
        .reproject(projection)
    )


def ndvi_band_to_int(image: ee.Image) -> ee.Image:
    """
    Convert the NDVI band of the image to an integer representation.

    Args:
        image (ee.Image): Input image with NDVI band.

    Returns:
        ee.Image: Image with NDVI band converted to integer representation.
    """

    ndvi_int = image.select("NDVI").multiply(10000).toInt().rename("NDVI_int")
    return image.addBands(ndvi_int)


def ndvi_band_to_float(image: ee.Image) -> ee.Image:
    """
    Convert the NDVI band of the image from integer to float representation.

    Args:
        image (ee.Image): Input image with NDVI band in integer representation.

    Returns:
        ee.Image: Image with NDVI band converted to float representation.
    """

    ndvi_float = image.select("NDVI_int").toFloat().divide(10000).rename("NDVI")
    return image.addBands(ndvi_float, overwrite=True)


def add_time_data(image: ee.Image) -> ee.Image:
    """
    Add time-related bands to the image, using any existing band for projection reference.

    Args:
        image (ee.Image): Input image.

    Returns:
        ee.Image: Image with added time-related bands in consistent projection.
    """
    # Get projection from any available band instead of specifically B2
    first_band_name = ee.String(image.bandNames().get(0))
    projection = image.select(first_band_name).projection()

    date = ee.Date(image.get("system:time_start"))
    years = date.difference(ee.Date("1970-01-01"), "year")

    time_band = ee.Image(years).float().rename("t").reproject(projection)
    constant_band = ee.Image.constant(1).reproject(projection)

    return image.addBands(time_band).addBands(constant_band)
