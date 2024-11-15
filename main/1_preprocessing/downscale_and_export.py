import sys
from pathlib import Path

sys.path.append(str(Path().absolute().parent))

import ee
import geemap

ee.Initialize(project="thurgau-irrigation")

from src.data_processing.downscaling import resample_image
from src.data_processing.sentinel_preprocessing import load_sentinel2_data

from utils.date_utils import (
    set_to_first_of_month,
    print_collection_dates,
    create_centered_date_ranges,
)
from utils.ee_utils import harmonized_ts, export_image_to_asset, back_to_int
from utils.harmonic_regressor import HarmonicRegressor, add_temporal_bands

from typing import List, Callable


# ---- END OF IMPORTS ----


def compute_vegetation_indexes(image: ee.Image) -> ee.Image:
    """
    Compute vegetation indexes for a given image

    Args:
        image (ee.Image): The image to compute the vegetation indexes for

    Returns:
        ee.Image: The input image with the vegetation indexes

    """
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")
    ndbi = image.normalizedDifference(["B11", "B8"]).rename("NDBI")
    return image.addBands(ndvi).addBands(ndwi).addBands(ndbi)


def fill_gaps(
    img: ee.Image, source_band: str, fill_band: str, output_name: str
) -> ee.Image:
    """Fill gaps in a band with values from another band.

    Args:
        img (ee.Image): Input image containing both bands
        source_band (str): Name of band containing gaps to fill
        fill_band (str): Name of band to use for filling gaps
        output_name (str): Name for the output gap-filled band

    Returns:
        ee.Image: Image with gap-filled band
    """
    # Create mask where the source band is invalid (gaps)
    gap_mask = img.select(source_band).mask().Not()

    # Get the source band and fill band
    source = img.select(source_band)
    fill = img.select(fill_band)

    # Fill gaps: use source band where available, fill band where there are gaps
    filled = source.unmask().where(gap_mask, fill).rename(output_name)

    return filled


def apply_gap_filling(img: ee.Image, indexes: List[str]) -> ee.Image:
    """Apply gap filling to multiple bands.

    Args:
        img (ee.Image): Input image
        indexes (list[str]): List of index names to process (e.g., ['NDVI', 'NDWI', 'NDBI'])

    Returns:
        ee.Image: Original image with added gap-filled bands
    """
    result = img

    for index in indexes:
        filled_band = fill_gaps(
            img=img,
            source_band=index,
            fill_band=f"fitted_{index}",
            output_name=f"gap_filled_{index}",
        )
        result = result.addBands(filled_band)

    return result


def process_collection(
    collection: ee.ImageCollection, indexes: List[str]
) -> ee.ImageCollection:
    """Process entire collection by applying gap filling to each image.

    Args:
        collection (ee.ImageCollection): Input collection
        indexes (List[str]): List of index names to process

    Returns:
        ee.ImageCollection: Processed collection with gap-filled bands
    """
    return collection.map(lambda img: apply_gap_filling(img, indexes))


def create_timesteps(year: str, time_step_type: str = "dekadal") -> List[dict]:
    """Generate timestep information for the year."""
    steps = 36 if time_step_type == "dekadal" else 12
    timesteps = []

    for i in range(steps):
        if time_step_type == "dekadal":
            dekad = i % 3 + 1
            month = i // 3 + 1
            date = ee.Date.fromYMD(int(year), month, dekad * 10 - 9)
            time_label = f"{month:02d}_D{dekad}"
        else:
            month = i + 1
            date = ee.Date.fromYMD(int(year), month, 1)
            time_label = f"{month:02d}"

        timesteps.append({"date": date, "label": time_label, "index": i})

    return timesteps


def process_and_export_downscaled_ET(
    downscaler: "Downscaler",
    s2_indices: ee.ImageCollection,
    independent_vars: ee.ImageCollection,
    dependent_vars: ee.ImageCollection,
    aoi: ee.Geometry,
    year: str,
    asset_id_template: Callable[[str, str], str],
    scale_coarse: float,
    scale_fine: float = 10,
    time_step_type: str = "dekadal",
    crs: str = "EPSG:32632",
) -> List[ee.batch.Task]:
    """
    Process and export downscaled ET images to Earth Engine assets.

    Args:
        downscaler: The Downscaler object for downscaling images
        s2_indices: Sentinel-2 indices ImageCollection
        independent_vars: Resampled independent variables ImageCollection
        dependent_vars: Dependent variables ImageCollection
        aoi: Area of interest geometry
        year: Processing year
        asset_id_template: Function that takes (year, time_label) and returns full asset path
        scale_coarse: Scale before downscaling
        scale_fine: Scale after downscaling (default: 10m)
        time_step_type: "dekadal" or "monthly" (default: "dekadal")
        crs: Output coordinate reference system (default: "EPSG:32632")

    Returns:
        List of export tasks
    """
    # Convert collections to lists for indexed access
    s2_indices_list = s2_indices.toList(s2_indices.size())
    independent_vars_list = independent_vars.toList(independent_vars.size())
    dependent_vars_list = dependent_vars.toList(dependent_vars.size())

    # Generate timesteps
    timesteps = create_timesteps(year, time_step_type)

    tasks = []
    for step in timesteps:
        # Get images for current timestep
        s2_index = ee.Image(s2_indices_list.get(step["index"]))
        ind_vars = ee.Image(independent_vars_list.get(step["index"]))
        dep_vars = ee.Image(dependent_vars_list.get(step["index"]))

        # Perform downscaling
        et_downscaled = downscaler.downscale(
            coarse_independent_vars=ind_vars,
            coarse_dependent_var=dep_vars,
            fine_independent_vars=s2_index,
            geometry=aoi,
            resolution=scale_coarse,
        )

        # Post-process
        et_downscaled = back_to_int(et_downscaled, 100)

        # Get asset ID from template function
        asset_id = asset_id_template(year, step["label"])

        # Export
        task = export_image_to_asset(
            et_downscaled,
            asset_id=asset_id,
            description=f"ET_downscaled_{year}_{step['label']}",
            year=year,
            aoi=aoi,
            crs=crs,
            scale=scale_fine,
        )
        tasks.append(task)

    return tasks


if __name__ == "__main__":

    # ---- CONSTANTS ----

    PATH_TO_AOI = "projects/thurgau-irrigation/assets/Thurgau/thrugau_borders_2024"
    PATH_TO_ET_PRODUCT = "projects/thurgau-irrigation/assets/ETlandsatmonthly"
    START_YEAR = "2022"
    END_YEAR = "2022"
    BUFFER_DAYS = 15
    BAND_TO_RESAMPLE = "ET"
    TARGET_RESAMPLE_SCALE = 100
    BANDS_TO_HARMONIZE = ["B3", "B4", "B8", "B11", "B12"]
    AGGREGATION_OPTIONS = {
        "agg_type": "mosaic",
        "mosaic_type": "least_cloudy",
        "band_name": "NDVI",
    }
    INDEXES_FOR_HARMONIZATION = ["NDVI", "NDWI", "NDBI"]

    # ---- END OF CONSTANTS ----

    aoi_feature_collection = ee.FeatureCollection(PATH_TO_AOI)
    aoi_geometry = aoi_feature_collection.geometry().simplify(500)

    aoi = aoi_geometry.buffer(100)

    landsat_ET_30m = ee.ImageCollection("PATH_TO_ET_PRODUCT").filterDate(
        f"{START_YEAR}-01-01", f"{END_YEAR}-12-31"
    )

    landsat_ET_30m = set_to_first_of_month(landsat_ET_30m)

    landsat_ET_100m = landsat_ET_30m.map(
        lambda img: resample_image(img, TARGET_RESAMPLE_SCALE, [BAND_TO_RESAMPLE])
    )
    landsat_ET_100m_list = landsat_ET_100m.toList(landsat_ET_100m.size())

    s2_collection = load_sentinel2_data(years_range=(START_YEAR, END_YEAR), aoi=aoi)

    time_intervals = create_centered_date_ranges(
        landsat_ET_100m_list, buffer_days=BUFFER_DAYS
    )

    s2_harmonized = harmonized_ts(
        masked_collection=s2_collection,
        band_list=BANDS_TO_HARMONIZE,
        time_intervals=time_intervals,
        options=AGGREGATION_OPTIONS,
    )

    s2_harmonized = add_temporal_bands(s2_harmonized)

    s2_harmonized_w_vegetation_indexes = s2_harmonized.map(compute_vegetation_indexes)

    s2_harmonized_gaps_filled = s2_harmonized_w_vegetation_indexes

    for index in INDEXES_FOR_HARMONIZATION:
        regressor = HarmonicRegressor(
            omega=1, max_harmonic_order=1, band_to_harmonize=index
        )

        regressor.fit(s2_harmonized_w_vegetation_indexes)
        fitted_collection = regressor.predict(s2_harmonized_w_vegetation_indexes)

        fitted_collection = fitted_collection.map(
            lambda img: img.select(["fitted"]).rename(f"fitted_{index}")
        )

        s2_harmonized_gaps_filled = s2_harmonized_gaps_filled.map(
            lambda img: img.addBands(
                fitted_collection.filterDate(img.date())
                .first()
                .select([f"fitted_{index}"])
            )
        )

    s2_harmonized_gaps_filled = process_collection(
        s2_harmonized_gaps_filled, INDEXES_FOR_HARMONIZATION
    )

    independent_bands = ["gap_filled_NDVI", "gap_filled_NDBI", "gap_filled_NDWI"]
    dependent_band = ["ET"]

    s2_indices = s2_harmonized_gaps_filled.select(independent_bands)

    independent_vars = s2_indices.map(
        lambda img: resample_image(
            image=img,
            target_scale=TARGET_RESAMPLE_SCALE,
            bands_to_resample=independent_bands,
        )
    ).select([f"resampled_{band}" for band in independent_bands])

    dependent_vars = landsat_ET_100m.select(dependent_band)

    # Get the scale from the coarse resolution product
    scale = landsat_ET_100m.first().projection().nominalScale().getInfo()

    # Initialize the Downscaler with resampled band names
    from src.data_processing.downscaling import Downscaler

    downscaler = Downscaler(
        independent_vars=[f"resampled_{band}" for band in independent_bands],
        dependent_var=dependent_band[0],
    )

    # Configure asset path template
    def asset_template(year: str, label: str) -> str:
        return (
            f"projects/thurgau-irrigation/assets/"
            f"Landsat_ET_monthly_downscaled_10m_Thurgau_{year}/"
            f"Downscaled_Landsat_ET_gap_filled_monthly_10m_Thurgau_{year}_{label}"
        )

    # Process and export
    tasks = process_and_export_downscaled_ET(
        downscaler=downscaler,
        s2_indices=s2_indices,
        independent_vars=independent_vars,
        dependent_vars=dependent_vars,
        aoi=aoi,
        year=START_YEAR,
        asset_id_template=asset_template,
        scale_coarse=scale,
        scale_fine=10,
        time_step_type="monthly",
        crs="EPSG:32632",
    )

    print(f"Started {len(tasks)} export tasks.")

    # Optional: Monitor task status
    for i, task in enumerate(tasks):
        print(f"Task {i + 1}: {task.status()}")
