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


def get_time_step_info(index: int, time_step_type: str) -> Tuple[str, int]:
    """
    Get formatted time step information based on index and type.

    Args:
        index (int): Time step index
        time_step_type (str): Either 'dekadal' or 'monthly'

    Returns:
        Tuple[str, int]: Formatted time step name and month number
    """
    if time_step_type == "dekadal":
        dekadal = index % 3 + 1
        month = index // 3 + 1
        time_step_name = f"{month:02d}_D{dekadal}"
    elif time_step_type == "monthly":
        month = index + 1
        time_step_name = f"{month:02d}"
    else:
        raise ValueError("time_step_type must be either 'dekadal' or 'monthly'")

    return time_step_name, month


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
    time_steps: int = 36,
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
        time_steps (int): Number of time steps
        time_step_type (str): Type of time step ("dekadal" or "monthly")
        not_irrigated_crops (List[str]): List of crops to exclude
        rainfed_crops (List[str]): List of rainfed reference crops

    Returns:
        List[ee.batch.Task]: List of export tasks
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
    for i in range(time_steps):
        # Get time step information
        time_step_name, _ = get_time_step_info(i, time_step_type)

        # Process ET image
        et_image = ee.Image(et_collection_list.get(i))
        et_green = compute_et_green(
            et_image, rainfed_fields, jurisdictions, et_band_name=et_band_name
        )

        # Convert to integer (assuming back_to_int is a utility function)
        et_green = back_to_int(et_green, 100)

        # Create export task
        task_name = f"ET_green_{time_step_type}_{year}_{time_step_name}"
        task = generate_export_task(
            et_green, asset_path, task_name, year, aoi, resolution
        )
        tasks.append(task)

    print(f"Generated {len(tasks)} export tasks for year {year}")
