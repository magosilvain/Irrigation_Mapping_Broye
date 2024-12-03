from typing import List, Tuple, Set
import ee
from src.et_green.compute_et_green import compute_et_green, compute_et_green_std
from src.et_green.filter_nutzungsflaechen import (
    get_crops_to_exclude,
    get_rainfed_reference_crops,
    create_crop_filters,
    filter_crops,
    add_double_cropping_info,
    get_unique_nutzung,
)
from utils.ee_utils import back_to_int, export_image_to_asset, normalize_string_server


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


def normalize_feature(feature: ee.Feature, property: str = "nutzung") -> ee.Feature:
    """Normalizes a property's string value in an Earth Engine Feature by replacing special characters.

    Adds a new property with suffix '_normalized' containing the normalized string value.
    For example, if property is "nutzung", creates "nutzung_normalized".

    Args:
        feature (ee.Feature): The Earth Engine Feature containing the property to normalize.
        property (str, optional): Name of the property to normalize. Defaults to "nutzung".

    Returns:
        ee.Feature: The input feature with an additional normalized property.
    """
    prop_value = ee.String(feature.get(property))

    normalized_prop_name = ee.String(property).cat("_normalized")

    normalized = normalize_string_server(prop_value)

    return feature.set(normalized_prop_name, normalized)


def prepare_rainfed_fields(
    landuse_collection: ee.FeatureCollection,
    double_cropping_image: ee.Image,
    not_irrigated_crops: Set[str],
    rainfed_crops: Set[str],
    minimum_field_size: int,
) -> ee.FeatureCollection:
    """
    Prepare rainfed fields by filtering and adding double cropping information.

    Args:
        landuse_collection (ee.FeatureCollection): Collection of land use features
        double_cropping_image (ee.Image): Image containing double cropping information
        not_irrigated_crops (List[str]): List of crop types that are not irrigated
        rainfed_crops (List[str]): List of rainfed reference crops
        minimum_field_size (int): Minimum field size in m^2

    Returns:
        ee.FeatureCollection: Filtered rainfed fields
    """
    landuse_collection = landuse_collection.map(normalize_feature)

    exclude_filter, rainfed_filter = create_crop_filters(
        not_irrigated_crops, rainfed_crops
    )

    nutzung_with_double_crop = add_double_cropping_info(
        landuse_collection, double_cropping_image
    )
    _, rainfed_fields = filter_crops(
        nutzung_with_double_crop, exclude_filter, rainfed_filter
    )

    # Add area property if not present
    rainfed_fields = rainfed_fields.map(
        lambda feature: feature.set("area", feature.geometry().area().divide(1).round())
    )

    # Drop all rainfed fields whose area is below minimum_field_size
    rainfed_fields = rainfed_fields.filter(ee.Filter.gte("area", minimum_field_size))

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
    minimum_field_size=1000,
    export_band_name: str = "ET_green",
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
        minimum_field_size (int): Minimum field size in m^2, defaults to 1000 (1 ha)
    """
    # Use default crop lists if none provided
    if not_irrigated_crops is None:
        not_irrigated_crops = get_crops_to_exclude()
    if rainfed_crops is None:
        rainfed_crops = get_rainfed_reference_crops()

    # Prepare rainfed fields
    rainfed_fields = prepare_rainfed_fields(
        landuse_collection,
        double_cropping_image,
        not_irrigated_crops,
        rainfed_crops,
        minimum_field_size,
    )

    tasks = []
    collection_size = ee.List(et_collection_list).size().getInfo()

    for i in range(collection_size):
        # Process ET image
        et_image = ee.Image(et_collection_list.get(i))

        # Get time step pattern from image date
        date = ee.Date(et_image.get("system:time_start"))
        time_step_pattern = get_time_step_pattern(date, time_step_type)

        et_green_2bands = compute_et_green_std(
            et_image, rainfed_fields, jurisdictions, et_band_name=et_band_name
        )
        # Convert to integer
        et_green = back_to_int(et_green_2bands.select('ET_green'), 100)
        et_green_std = back_to_int(et_green_2bands.select('ET_green_std'), 100)
        et_green_std = et_green_std.rename(f"{export_band_name}_std")
        et_green = et_green.addBands(et_green_std)

        # Create export task
        task_name = f"{export_band_name}_{time_step_type}_{year}_{time_step_pattern}"
        # task_name = f"ET_green_{time_step_type}_{year}_{time_step_pattern}"
        task = generate_export_task(
            et_green, asset_path, task_name, year, aoi, resolution
        )
        tasks.append(task)

    print(f"Generated {len(tasks)} export tasks for year {year}")


def process_et_green_std(
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
    minimum_field_size=1000,
    export_band_name: str = "ET_green",
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
        minimum_field_size (int): Minimum field size in m^2, defaults to 1000 (1 ha)
    """
    # Use default crop lists if none provided
    if not_irrigated_crops is None:
        not_irrigated_crops = get_crops_to_exclude()
    if rainfed_crops is None:
        rainfed_crops = get_rainfed_reference_crops()

    # Prepare rainfed fields
    rainfed_fields = prepare_rainfed_fields(
        landuse_collection,
        double_cropping_image,
        not_irrigated_crops,
        rainfed_crops,
        minimum_field_size,
    )

    tasks = []
    collection_size = ee.List(et_collection_list).size().getInfo()

    for i in range(collection_size):
        # Process ET image
        et_image = ee.Image(et_collection_list.get(i))

        # Get time step pattern from image date
        date = ee.Date(et_image.get("system:time_start"))
        time_step_pattern = get_time_step_pattern(date, time_step_type)

        et_green = compute_et_green_std(
            et_image, rainfed_fields, jurisdictions, et_band_name=et_band_name
        )

        # Convert to integer
        et_green = back_to_int(et_green, 100)

        # Create export task
        task_name = f"{export_band_name}_{time_step_type}_{year}_{time_step_pattern}"
        # task_name = f"ET_green_{time_step_type}_{year}_{time_step_pattern}"
        task = generate_export_task(
            et_green, asset_path, task_name, year, aoi, resolution
        )
        tasks.append(task)

    print(f"Generated {len(tasks)} export tasks for year {year}")