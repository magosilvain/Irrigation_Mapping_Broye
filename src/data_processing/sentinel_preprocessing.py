# File: /src/gee_processing/sentinel_cloud_mask.py

import ee
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Callable


@dataclass
class CloudMaskParams:
    """Parameters for cloud masking configuration."""

    cloud_prob_threshold: int = 50
    cloud_filter: int = 40
    nir_dark_threshold: float = 0.15
    cloud_proj_distance: int = 2
    buffer: int = 100
    mask_resolution: int = 60


class SentinelProcessor:
    """Enhanced Sentinel-2 data processor with improved cloud masking."""

    def __init__(self, params: CloudMaskParams = CloudMaskParams()):
        self.params = params

    def _get_water_mask(self) -> ee.Image:
        """Get the global water mask."""
        return ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("max_extent").eq(0)

    def _apply_quality_mask(self, image: ee.Image) -> ee.Image:
        """Apply QA60 bitmask for basic quality filtering."""
        qa_mask = image.select("QA60")
        cloud_bitmask = 1 << 10  # Bit 10: Opaque clouds
        cirrus_bitmask = 1 << 11  # Bit 11: Cirrus clouds
        return image.updateMask(
            qa_mask.bitwiseAnd(cloud_bitmask)
            .eq(0)
            .And(qa_mask.bitwiseAnd(cirrus_bitmask).eq(0))
        )

    def _add_cloud_bands(self, image: ee.Image) -> ee.Image:
        """Enhanced cloud probability and mask bands."""
        # Get cloud probability from s2cloudless
        cloud_prob = ee.Image(image.get("s2cloudless")).select("probability")

        # Create basic cloud mask
        is_cloud = cloud_prob.gt(self.params.cloud_prob_threshold).rename("clouds")

        # Additional check for bright pixels in blue band
        bright_pixels = image.select("B2").gt(2500)

        # Combine cloud probability with bright pixels
        combined_cloud = is_cloud.Or(bright_pixels).rename("clouds")

        return image.addBands([cloud_prob.rename("probability"), combined_cloud])

    def _add_shadow_bands(self, image: ee.Image, water_mask: ee.Image) -> ee.Image:
        """Enhanced cloud shadow detection."""
        # More conservative NIR threshold for shadows
        dark_pixels = (
            image.select("B8")
            .lt(self.params.nir_dark_threshold * 0.8)
            .And(image.select("B3").lt(self.params.nir_dark_threshold * 0.8))
            .multiply(water_mask)
            .rename("dark_pixels")
        )

        # Calculate shadow direction based on solar angle
        shadow_azimuth = ee.Number(90).subtract(
            ee.Number(image.get("MEAN_SOLAR_AZIMUTH_ANGLE"))
        )

        # Project clouds to find shadows
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

        # Combine cloud projection with dark pixels
        shadows = cloud_proj.multiply(dark_pixels).rename("shadows")

        return image.addBands([dark_pixels, cloud_proj, shadows])

    def _add_cloud_shadow_mask(self, image: ee.Image, water_mask: ee.Image) -> ee.Image:
        """Create final cloud and shadow mask with enhanced filtering."""
        # Apply basic quality mask first
        image = self._apply_quality_mask(image)

        # Add cloud and shadow bands
        with_clouds = self._add_cloud_bands(image)
        with_shadows = self._add_shadow_bands(with_clouds, water_mask)

        # Combine clouds and shadows
        is_cld_shdw = with_shadows.select("clouds").Or(with_shadows.select("shadows"))

        # Apply morphological operations for cleaner mask
        final_mask = (
            is_cld_shdw.focal_min(2)
            .focal_max(self.params.buffer * 2 / self.params.mask_resolution)
            .reproject(
                crs=image.select([0]).projection(), scale=self.params.mask_resolution
            )
            .rename("cloudmask")
        )

        # Return image with all masks
        return with_shadows.addBands(final_mask)

    def process_image(self, image: ee.Image) -> ee.Image:
        """Process a single Sentinel-2 image with enhanced cloud masking."""
        projection = image.select("B3").projection()
        water_mask = self._get_water_mask().reproject(projection)

        # Apply all masks
        masked = self._add_cloud_shadow_mask(image, water_mask)

        # Only keep pixels that pass all masks
        final_mask = masked.select("cloudmask").Not()
        masked = masked.updateMask(final_mask)

        return masked.reproject(projection)


def load_sentinel2_data(
    years_range: tuple[int, int],
    aoi: ee.Geometry,
    cloud_params: Optional[CloudMaskParams] = None,
    apply_water_mask: bool = True,
) -> ee.ImageCollection:
    """
    Load and process Sentinel-2 data with enhanced cloud filtering for a range of years.

    Args:
        years_range (tuple[int, int]): Start and end years (inclusive) for data loading.
        aoi (ee.Geometry): The area of interest.
        cloud_params (Optional[CloudMaskParams]): Custom cloud masking parameters.
        apply_water_mask (bool): Whether to apply water masking.

    Returns:
        ee.ImageCollection: Processed Sentinel-2 image collection.
    """
    processor = SentinelProcessor(cloud_params or CloudMaskParams())

    # Set up date range
    start_date = ee.Date.fromYMD(years_range[0], 1, 1)
    end_date = ee.Date.fromYMD(years_range[1], 12, 31)

    # Load and filter collection with stricter initial cloud filter
    s2_collection = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filterDate(start_date, end_date)
        .filterBounds(aoi)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
    )

    # Get cloud probability collection
    s2_cloudless = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
    )

    # Join collections
    join = ee.Join.saveFirst("s2cloudless")
    joined = ee.ImageCollection(
        join.apply(
            primary=s2_collection,
            secondary=s2_cloudless,
            condition=ee.Filter.equals(
                leftField="system:index", rightField="system:index"
            ),
        )
    )

    # Process each image
    processed = joined.map(processor.process_image)

    # Additional post-processing filter for remaining cloudy images
    return processed.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)).set(
        "sensor_id", 0
    )
