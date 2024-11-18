# File: src/et_blue/processing_utils.py

from typing import List, Tuple
import ee
from utils.ee_utils import back_to_int, export_image_to_asset
from src.et_blue.compute_et_blue import compute_et_blue
from src.et_green.exporting_utils import get_time_step_info, generate_export_task


def process_et_blue(
    et_collection_list: ee.List,
    et_green_list: ee.List,
    year: int,
    aoi: ee.Geometry,
    asset_path: str,
    time_steps: int = 12,
    time_step_type: str = "monthly",
    resolution: int = 10,
) -> None:
    """
    Process and export ET blue images for a given year.

    Args:
        wapor_et_list (ee.List): List of WaPOR ET images
        et_green_list (ee.List): List of ET green images
        year (int): Year to process
        aoi (ee.Geometry): Area of interest
        asset_path (str): Base path for asset export
        time_steps (int): Number of time steps
        time_step_type (str): Type of time step ("monthly" or "dekadal")
        resolution (int): Export resolution in meters
    """
    if time_steps not in [12, 36]:
        raise ValueError("time_steps must be either 12 or 36")

    tasks = []
    for i in range(time_steps):
        # Get time step information
        time_step_name, _ = get_time_step_info(i, time_step_type)

        # Process ET images
        et_image = ee.Image(et_collection_list.get(i))
        et_green = ee.Image(et_green_list.get(i))
        et_blue = compute_et_blue(et_image, et_green)
        et_blue = back_to_int(et_blue, 100)

        # Create export task
        task_name = f"ET_blue_raw_{time_step_type}_{year}_{time_step_name}"
        task = generate_export_task(
            et_blue, asset_path, task_name, year, aoi, resolution
        )
        tasks.append(task)

    print(f"Generated {len(tasks)} export tasks for year {year}")
