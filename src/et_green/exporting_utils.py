from typing import List, Tuple, Set
import ee
from src.et_green.compute_et_green import compute_et_green
from src.et_green.filter_nutzungsflaechen import (
    get_crops_to_exclude,
    get_rainfed_reference_crops,
    create_crop_filters,
    filter_crops,
    add_double_cropping_info,
    get_unique_nutzung,
)
from utils.ee_utils import back_to_int, export_image_to_asset


def get_time_step_pattern(date: ee.Date, time_step_type: str) -> str:
    """
    Get formatted time step pattern from a date based on type.

    Args:
        date (ee.Date): The date to process
        time_step_type (str): Either 'dekadal' or 'monthly'

    Returns:
        str: Formatted time step pattern (e.g. '04_D1' for dekadal or '04' for monthly)

    Raises:
        ValueError: If time_step_type is neither 'dekadal' nor 'monthly'
    """
    if time_step_type not in ["dekadal", "monthly"]:
        raise ValueError("time_step_type must be either 'dekadal' or 'monthly'")

    # Add 1 to month since GEE uses 0-based months
    month = date.get("month").getInfo()
    month_str = f"{month:02d}"

    if time_step_type == "monthly":
        return month_str

    # For dekadal, determine which 10-day period
    day = date.get("day").getInfo()
    dekadal = ((day - 1) // 10) + 1
    return f"{month_str}_D{dekadal}"


def prepare_rainfed_fields(
    landuse_collection: ee.FeatureCollection,
    double_cropping_image: ee.Image,
    not_irrigated_crops: Set[str],
    rainfed_crops: Set[str],
) -> ee.FeatureCollection:
    """
    Prepare rainfed fields by filtering and adding double cropping information.

    Args:
        landuse_collection (ee.FeatureCollection): Collection of land use features
        double_cropping_image (ee.Image): Image containing double cropping information
        not_irrigated_crops (List[str]): List of crop types that are not irrigated
        rainfed_crops (List[str]): List of rainfed reference crops

    Returns:
        ee.FeatureCollection: Filtered rainfed fields
    """

    exclude_filter, rainfed_filter = create_crop_filters(
        not_irrigated_crops, rainfed_crops
    )

    nutzung_with_double_crop = add_double_cropping_info(
        landuse_collection, double_cropping_image
    )
    _, rainfed_fields = filter_crops(
        nutzung_with_double_crop, exclude_filter, rainfed_filter
    )

    return rainfed_fields


def generate_export_task(
    et_green: ee.Image,
    asset_path: str,
    task_name: str,
    year: int,
    aoi: ee.Geometry,
    resolution: int = 10,
) -> ee.batch.Task:
    """
    Generate an export task for an ET green image.

    Args:
        et_green (ee.Image): ET green image to export
        asset_path (str): Base path for the asset
        task_name (str): Name of the export task
        year (int): Year being processed
        aoi (ee.Geometry): Area of interest
        resolution (int): Export resolution in meters

    Returns:
        ee.batch.Task: Export task
    """
    asset_id = f"{asset_path}/{task_name}"
    crs = et_green.projection().crs()

    task = export_image_to_asset(
        image=et_green,
        asset_id=asset_id,
        task_name=task_name,
        aoi=aoi,
        crs=crs,
        scale=resolution,
        year=year,
    )

    return task


def process_et_green(
    et_collection_list: ee.List,
    landuse_collection: ee.FeatureCollection,
    jurisdictions: ee.FeatureCollection,
    double_cropping_image: ee.Image,
    year: int,
    aoi: ee.Geometry,
    asset_path: str,
    et_band_name: str = "downscaled",
    time_step_type: str = "dekadal",
    resolution: int = 10,
    not_irrigated_crops: List[str] = None,
    rainfed_crops: List[str] = None,
) -> None:
    """
    Process and export ET green images for a given year.

    Args:
        et_collection_list (ee.List): List of ET images
        landuse_collection (ee.FeatureCollection): Collection of land use features
        jurisdictions (ee.FeatureCollection): Collection of jurisdiction boundaries
        double_cropping_image (ee.Image): Double cropping classification image
        year (int): Year to process
        aoi (ee.Geometry): Area of interest
        asset_path (str): Base path for asset export
        et_band_name (str): Name of the ET band to process
        time_step_type (str): Type of time step ("dekadal" or "monthly")
        resolution (int): Export resolution in meters
        not_irrigated_crops (List[str]): List of crops to exclude
        rainfed_crops (List[str]): List of rainfed reference crops
    """
    # Use default crop lists if none provided
    if not_irrigated_crops is None:
        not_irrigated_crops = get_crops_to_exclude()
    if rainfed_crops is None:
        rainfed_crops = get_rainfed_reference_crops()

    # Prepare rainfed fields
    rainfed_fields = prepare_rainfed_fields(
        landuse_collection, double_cropping_image, not_irrigated_crops, rainfed_crops
    )

    tasks = []
    collection_size = ee.List(et_collection_list).size().getInfo()

    for i in range(collection_size):
        # Process ET image
        et_image = ee.Image(et_collection_list.get(i))

        # Get time step pattern from image date
        date = ee.Date(et_image.get("system:time_start"))
        time_step_pattern = get_time_step_pattern(date, time_step_type)

        et_green = compute_et_green(
            et_image, rainfed_fields, jurisdictions, et_band_name=et_band_name
        )

        # Convert to integer
        et_green = back_to_int(et_green, 100)

        # Create export task
        task_name = f"ET_green_{time_step_type}_{year}_{time_step_pattern}"
        task = generate_export_task(
            et_green, asset_path, task_name, year, aoi, resolution
        )
        tasks.append(task)

    print(f"Generated {len(tasks)} export tasks for year {year}")
