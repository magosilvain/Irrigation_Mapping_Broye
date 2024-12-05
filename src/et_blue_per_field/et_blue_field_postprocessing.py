import ee
from typing import Optional, Dict, Any, List


def compute_regional_stats(
    image: ee.Image,
    geometry: ee.Geometry,
    reducer: ee.Reducer,
    band_name: str,
    scale: float,
    max_pixels: int,
    projection: ee.Projection,
) -> ee.Dictionary:
    """Compute regional statistics for an image."""
    return ee.Dictionary(
        image.reduceRegion(
            reducer=reducer,
            geometry=geometry,
            scale=scale,
            maxPixels=max_pixels,
            crs=projection,
        )
    )


def compute_field_et_stats(
    et_image: ee.Image,
    fields: ee.FeatureCollection,
    date: str,
    et_band_name: str = "ET",
    scale: int = 10,
    max_pixels: int = int(1e9),
    prc: int = 50
) -> ee.FeatureCollection:
    """
    Compute ET statistics for each field in a feature collection.

    Args:
        et_image: Input ET image
        fields: Feature collection of field boundaries
        date: Date of the image
        et_band_name: Name of the ET band in the image
        scale: Scale in meters for computation. If None, uses native scale of the image
        max_pixels: Maximum number of pixels to process in reduction operations

    Returns:
        FeatureCollection with added properties:
        - median_et_blue: median ET value for each field
        - mean_et_nonzero: mean ET value excluding zero pixels
        - std_dev_et_nonzero: standard deviation of non-zero ET values
        - zero_fraction: fraction of pixels with 0 value in each field
    """

    et_image = et_image.unmask(-99)
    projection = et_image.projection()
    if scale is None:
        scale = projection.nominalScale()

    et_image = et_image.setDefaultProjection(projection)

    def compute_feature_stats(feature: ee.Feature) -> ee.Feature:
        geometry = feature.geometry()

        # Create masks
        zero_mask = et_image.select(et_band_name).eq(0)
        nonzero_et = et_image.select(et_band_name).updateMask(zero_mask.Not())

        # # Compute statistics
        # median_stats = compute_regional_stats(
        #     et_image.select(et_band_name),
        #     geometry,
        #     ee.Reducer.median(),
        #     et_band_name,
        #     scale,
        #     max_pixels,
        #     projection,
        # )
        # Compute statistics
        median_stats = compute_regional_stats(
            et_image.select(et_band_name),
            geometry,
            ee.Reducer.percentile([100 - prc]),
            et_band_name,
            scale,
            max_pixels,
            projection,
        )

        median_stats_gt0 = compute_regional_stats(
            et_image.select(et_band_name).updateMask(et_image.select(et_band_name).gt(0)),
            geometry,
            ee.Reducer.median(),
            et_band_name,
            scale,
            max_pixels,
            projection,
        )

        zero_stats = compute_regional_stats(
            zero_mask,
            geometry,
            ee.Reducer.mean(),
            et_band_name,
            scale,
            max_pixels,
            projection,
        )

        # Extract and set properties
        return feature.set(
            {
                f"median_et_blue_{date}": median_stats.get(et_band_name),
                f"median_et_blue_gt0_{date}": median_stats_gt0.get(et_band_name),
                # f"median_et_blue_{date}": median_stats.values().get(0),
                f"zero_fraction_{date}": zero_stats.get(et_band_name),
            }
        )

    return fields.map(compute_feature_stats)


def compute_et_volume(fields: ee.FeatureCollection, date: str) -> ee.FeatureCollection:
    """
    Compute ET volume in cubic meters for each field.

    Args:
        fields: FeatureCollection with median_et_nonzero property
        date: Date of the image

    Returns:
        FeatureCollection with new et_blue_m3 property
    """

    def add_volume(feature: ee.Feature, date: str) -> ee.Feature:
        area = feature.geometry().area()
        # et_mm = ee.Number(feature.get(f"median_et_blue_{date}"))
        et_mm = ee.Number(ee.Algorithms.If(ee.Number(feature.get(f"median_et_blue_{date}")).gt(0),
                          feature.get(f"median_et_blue_gt0_{date}"),
                          0))
        et_volume = et_mm.multiply(area).divide(1000)

        return feature.set({f"et_blue_m3_{date}": et_volume})

    return fields.map(lambda feature: add_volume(feature, date))


def threshold_et_volume(
    fields: ee.FeatureCollection, date: str, threshold: float
) -> ee.FeatureCollection:
    """
    Set et_blue_m3 to 0 if below threshold, otherwise keep current value.

    Args:
        fields: FeatureCollection with et_blue_m3 property
        date: Date of the image
        threshold: Minimum volume threshold in cubic meters

    Returns:
        FeatureCollection with thresholded et_blue_m3 property
    """

    def apply_threshold(feature: ee.Feature) -> ee.Feature:
        volume = ee.Number(feature.get(f"et_blue_m3_{date}"))
        new_volume = ee.Number(ee.Algorithms.If(volume.lt(threshold), 0, volume))
        return feature.set({f"et_blue_m3_{date}": new_volume})

    return fields.map(apply_threshold)
