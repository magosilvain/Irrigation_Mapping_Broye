import ee
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Callable, Tuple


@dataclass
class CloudMaskParams:
    """Parameters for cloud masking configuration."""

    cloud_prob_threshold: int = 40
    cloud_filter: int = 60
    nir_dark_threshold: float = 0.15
    cloud_proj_distance: int = 10
    buffer: int = 50
    mask_resolution: int = 60
    ndvi_threshold: float = 0.99

class LandsatProcessor:
    """Landsat 8 data processor with cloud and shadow masking."""

    def __init__(self, params: CloudMaskParams = CloudMaskParams()):
        self.params = params

    def _get_water_mask(self) -> ee.Image:
        """Get the global water mask."""
        return ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("max_extent").eq(0)

    def _get_qa_mask(self, image: ee.Image) -> ee.Image:
        """Generate a mask based on the QA_PIXEL band."""
        # Parse the QA_PIXEL band for cloud and shadow information
        qa = image.select("QA_PIXEL")
        cloud = qa.bitwiseAnd(1 << 3).neq(0)  # Cloud bit
        cloud_shadow = qa.bitwiseAnd(1 << 4).neq(0)  # Cloud shadow bit

        # Combine cloud and shadow into a single mask
        qa_mask = cloud.Or(cloud_shadow).rename("qa_mask")
        return qa_mask

    def _add_cloud_shadow_bands(self, image: ee.Image, water_mask: ee.Image) -> ee.Image:
        """Add cloud and shadow detection bands."""
        # Detect dark pixels (potential shadow areas)
        dark_pixels = (
            image.select("SR_B5")
            .lt(self.params.nir_dark_threshold)
            .multiply(water_mask)
            .rename("dark_pixels")
        )

        # Estimate cloud shadow projection direction
        shadow_azimuth = ee.Number(90).subtract(
            ee.Number(image.get("SUN_AZIMUTH"))
        )

        cloud_proj = (
            image.select("qa_mask")
            .directionalDistanceTransform(
                shadow_azimuth, self.params.cloud_proj_distance
            )
            .reproject(
                crs=image.select("SR_B4").projection(), scale=self.params.mask_resolution
            )
            .select("distance")
            .mask()
            .rename("cloud_transform")
        )

        shadows = cloud_proj.multiply(dark_pixels).rename("shadows")
        return image.addBands([dark_pixels, cloud_proj, shadows])

    def process_image(self, image: ee.Image) -> ee.Image:
        """Process a single Landsat 8 image with cloud and shadow masking."""
        projection = image.select("SR_B4").projection()
        water_mask = self._get_water_mask().reproject(projection)

        # Apply QA mask
        qa_mask = self._get_qa_mask(image)
        image_with_mask = image.addBands(qa_mask)

        # Add cloud and shadow bands
        with_cloud_shadows = self._add_cloud_shadow_bands(image_with_mask, water_mask)

        # Combine the masks
        final_mask = (
            with_cloud_shadows.select("qa_mask")
            .Or(with_cloud_shadows.select("shadows"))
            .Not()
        )
        masked = with_cloud_shadows.updateMask(final_mask)

        # Add time band
        date = ee.Date(image.get("system:time_start"))
        years = date.difference(ee.Date("1970-01-01"), "year")
        time_band = ee.Image(years).float().rename("t").reproject(projection)

        # Add constant band
        constant_band = ee.Image.constant(1).reproject(projection)

        return (
            masked.addBands([time_band, constant_band])
            .reproject(projection)
            .set("CLOUD_COVER", image.get("CLOUD_COVER_LAND"))
        )


def load_landsat_data(
    years_range: tuple[int, int],
    aoi: ee.Geometry,
    cloud_params: Optional[CloudMaskParams] = None,
) -> ee.ImageCollection:
    """Load and process Landsat data with cloud filtering."""

    start_date = ee.Date.fromYMD(years_range[0], 1, 1)
    end_date = ee.Date.fromYMD(years_range[1], 12, 31)

    collection_to_use = (
        "LANDSAT/LC08/C02/T1_L2"
    )

    l8_collection = (
        ee.ImageCollection(collection_to_use)
        .filterDate(start_date, end_date)
        .filterBounds(aoi)
        .filter(
            ee.Filter.lt(
                "CLOUD_COVER",
                cloud_params.cloud_filter if cloud_params else 60,
            )
        )
    )
    processor = LandsatProcessor(cloud_params or CloudMaskParams())
    return l8_collection.map(processor.process_image).set("sensor_id", 0)

class SentinelProcessor:
    """Sentinel-2 data processor with cloud masking."""

    def __init__(self, params: CloudMaskParams = CloudMaskParams()):
        self.params = params

    def _get_water_mask(self) -> ee.Image:
        """Get the global water mask."""
        return ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("max_extent").eq(0)

    def _add_cloud_bands(self, image: ee.Image) -> ee.Image:
        """Add cloud probability and initial mask bands."""
        # Cloud probability from s2cloudless
        cloud_prob = ee.Image(image.get("s2cloudless")).select("probability")
        is_cloud = cloud_prob.gt(self.params.cloud_prob_threshold)

        # Calculate NDVI
        ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")

        # Create combined cloud mask
        cloud_mask = (
            ee.Image(1)
            # QA60 filter
            .where(
                image.select("QA60").lt(1024),
                # B1 filter
                ee.Image(1).where(image.select("B1").gt(0), 0),
            )
            # NDVI filter
            .where(ndvi.gt(self.params.ndvi_threshold), 1)
            # Combine with cloud probability
            .Or(is_cloud).rename("clouds")
        )

        return image.addBands(cloud_prob.rename("probability")).addBands(cloud_mask)

    def _add_shadow_bands(self, image: ee.Image, water_mask: ee.Image) -> ee.Image:
        """Add cloud shadow detection bands."""
        dark_pixels = (
            image.select("B8")
            .lt(self.params.nir_dark_threshold)
            .multiply(water_mask)
            .rename("dark_pixels")
        )

        shadow_azimuth = ee.Number(90).subtract(
            ee.Number(image.get("MEAN_SOLAR_AZIMUTH_ANGLE"))
        )

        cloud_proj = (
            image.select("clouds")
            .directionalDistanceTransform(
                shadow_azimuth, self.params.cloud_proj_distance
            )
            .reproject(
                crs=image.select(0).projection(), scale=self.params.mask_resolution
            )
            .select("distance")
            .mask()
            .rename("cloud_transform")
        )

        shadows = cloud_proj.multiply(dark_pixels).rename("shadows")
        return image.addBands([dark_pixels, cloud_proj, shadows])

    def process_image(self, image: ee.Image) -> ee.Image:
        """Process a single Sentinel-2 image with cloud masking."""
        projection = image.select("B3").projection()
        water_mask = self._get_water_mask().reproject(projection)

        with_clouds = self._add_cloud_bands(image)
        with_shadows = self._add_shadow_bands(with_clouds, water_mask)

        final_mask = (
            with_shadows.select("clouds").Or(with_shadows.select("shadows")).Not()
        )
        masked = with_shadows.updateMask(final_mask)

        # Add time band
        date = ee.Date(image.get("system:time_start"))
        years = date.difference(ee.Date("1970-01-01"), "year")
        time_band = ee.Image(years).float().rename("t").reproject(projection)

        # Add constant band
        constant_band = ee.Image.constant(1).reproject(projection)

        return (
            masked.addBands([time_band, constant_band])
            .reproject(projection)
            .set("CLOUD_COVER", image.get("CLOUDY_PIXEL_PERCENTAGE"))
        )

def load_sentinel2_data(
    years_range: tuple[int, int],
    aoi: ee.Geometry,
    cloud_params: Optional[CloudMaskParams] = None,
    use_SR: bool = False,
) -> ee.ImageCollection:
    """Load and process Sentinel-2 data with cloud filtering."""
    processor = SentinelProcessor(cloud_params or CloudMaskParams())

    start_date = ee.Date.fromYMD(years_range[0], 1, 1)
    end_date = ee.Date.fromYMD(years_range[1], 12, 31)

    collection_to_use = (
        "COPERNICUS/S2_SR_HARMONIZED" if use_SR else "COPERNICUS/S2_HARMONIZED"
    )

    s2_collection = (
        ee.ImageCollection(collection_to_use)
        .filterDate(start_date, end_date)
        .filterBounds(aoi)
        .filter(
            ee.Filter.lt(
                "CLOUDY_PIXEL_PERCENTAGE",
                cloud_params.cloud_filter if cloud_params else 60,
            )
        )
    )

    s2_cloudless = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
    )

    joined = ee.ImageCollection(
        ee.Join.saveFirst("s2cloudless").apply(
            primary=s2_collection,
            secondary=s2_cloudless,
            condition=ee.Filter.equals(
                leftField="system:index", rightField="system:index"
            ),
        )
    )

    return joined.map(processor.process_image).set("sensor_id", 0)
