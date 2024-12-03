import ee
import pandas as pd
from datetime import datetime
from typing import Optional, Union


def print_collection_dates(collection: ee.ImageCollection) -> None:
    """
    Print the dates of all images in an ImageCollection.

    Args:
        collection (ee.ImageCollection): The input image collection.

    Returns:
        None: This function prints the dates to the console.
    """
    # Get a list of all image dates
    dates = collection.aggregate_array("system:time_start")

    # Convert to ee.Date objects and format as strings
    formatted_dates = dates.map(lambda d: ee.Date(d).format("YYYY-MM-dd"))

    # Get the list of formatted dates
    date_list = formatted_dates.getInfo()

    print("Dates of images in the collection:")
    for date in date_list:
        print(date)


def store_collection_dates(collection: ee.ImageCollection) -> pd.DataFrame:
    """
    Store the dates of all images in an ImageCollection in a pandas DataFrame.

    Args:
        collection (ee.ImageCollection): The input image collection.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the dates in datetime format.
    """
    dates = collection.aggregate_array("system:time_start")
    formatted_dates = dates.map(lambda d: ee.Date(d).format("YYYY-MM-dd"))
    date_list = formatted_dates.getInfo()

    date_df = pd.DataFrame({"date": pd.to_datetime(date_list)})

    return date_df


def update_image_timestamp(
    collection: ee.ImageCollection, image_id: str, date_str: str
) -> ee.ImageCollection:
    """
    Update the 'system:time_start' for a specific image in the collection.

    Args:
        collection (ee.ImageCollection): The original image collection.
        image_id (str): The ID of the image to update.
        date_str (str): The date string in 'YYYY-MM-DD' format.

    Returns:
        ee.ImageCollection: Updated image collection.
    """
    # Convert the date string to a timestamp
    date = datetime.strptime(date_str, "%Y-%m-%d")
    timestamp = int(date.timestamp() * 1000)  # Convert to milliseconds

    # Function to update the image if it matches the ID
    def update_image(image):
        return ee.Algorithms.If(
            ee.String(image.get("system:index")).equals(image_id),
            image.set("system:time_start", timestamp),
            image,
        )

    # Map the update function over the collection
    updated_collection = collection.map(update_image)

    return updated_collection


def create_centered_date_ranges(image_list: ee.List, buffer_days: int = 5) -> ee.List:
    """
    Creates date ranges centered around the timestamps of a list of Earth Engine images.

    Args:
        image_list (ee.List): A list of Earth Engine images.
        buffer_days (int): Number of days to buffer before and after the center date. Defaults to 5.

    Returns:
        ee.List: A list of lists, where each inner list contains two ee.Date objects
                 representing the start and end of a date range, centered around the image timestamp.
    """

    def create_centered_range(image, buffer_days):
        center_date = ee.Date(ee.Image(image).get("system:time_start"))
        start_date = center_date.advance(-buffer_days, "day")
        end_date = center_date.advance(buffer_days, "day")
        return ee.List([start_date, end_date])

    return image_list.map(lambda img: create_centered_range(img, buffer_days))


def create_forward_date_ranges(image_list: ee.List, window_days: int = 10) -> ee.List:
    """
    Creates forward-looking date ranges starting from the timestamps of a list of Earth Engine images.

    Args:
        image_list (ee.List): A list of Earth Engine images.
        window_days (int): Number of days to look forward from the start date. Defaults to 10.

    Returns:
        ee.List: A list of lists, where each inner list contains two ee.Date objects
                representing the start and end of a date range. The range starts at
                the image timestamp and extends forward by window_days.
    """

    def create_forward_range(image, days):
        # Get the start date from the image timestamp
        start_date = ee.Date(ee.Image(image).get("system:time_start"))
        # Create end date by advancing forward by the specified number of days
        end_date = start_date.advance(days, "day")
        # Return as a list of [start_date, end_date]
        return ee.List([start_date, end_date])

    return image_list.map(lambda img: create_forward_range(img, window_days))


def set_to_first_of_month(collection: ee.ImageCollection) -> ee.ImageCollection:
    """
    Updates the dates of all images in a collection to the first day of their respective months.

    Args:
        collection (ee.ImageCollection): Input image collection

    Returns:
        ee.ImageCollection: Collection with updated dates
    """

    def update_date(image):
        # Get the current timestamp
        date = ee.Date(image.get("system:time_start"))

        # Create new date for first of the month
        new_date = ee.Date.fromYMD(date.get("year"), date.get("month"), 1)

        # Update the image with new timestamp
        return image.set("system:time_start", new_date.millis())

    return collection.map(update_date)


def get_days_in_month(date: ee.Date) -> ee.Number:
    """
    Get the number of days in a month for a given Earth Engine Date object.
    Handles all months correctly, including February in leap years.

    Args:
        date (ee.Date): The input date to get the days in month for

    Returns:
        ee.Number: The number of days in the month
    """
    # Get the year and month
    year = date.get("year")
    month = date.get("month")

    # Create start of next month
    next_month = ee.Date.fromYMD(
        ee.Algorithms.If(month.eq(12), year.add(1), year),
        ee.Algorithms.If(month.eq(12), 1, month.add(1)),
        1,
    )

    # Calculate days by differencing dates
    days = next_month.difference(date.update(day=1), "day")

    return days


# def merge_same_date_images(
#     collection: ee.ImageCollection,
#     padding_size: Optional[Union[int, float]] = None,
# ) -> ee.ImageCollection:
#     """
#     Merges images from the same date in a collection, with zero padding around valid pixels.

#     Args:
#         collection (ee.ImageCollection): Input collection with potentially duplicate dates
#         padding_size (Optional[Union[int, float]]): Padding size in meters. If None, no padding is applied

#     Returns:
#         ee.ImageCollection: Collection with one image per unique date, with zero padding buffer
#     """
#     dates = collection.aggregate_array("system:time_start").distinct()

#     def merge_date_images(date):
#         date_num = ee.Number(date)
#         date_imgs = collection.filter(ee.Filter.eq("system:time_start", date_num))

#         def add_padding(img):
#             """Add a buffer of zeros around the image"""
#             if padding_size is None:
#                 return img

#             # Get original valid pixels mask
#             original_mask = img.mask()

#             # Create padded mask by growing the original mask
#             padded_mask = original_mask.focal_max(radius=padding_size, units="meters")

#             return img.updateMask(original_mask).unmask(0).updateMask(padded_mask)

#         if padding_size is not None:
#             date_imgs = date_imgs.map(add_padding)

#         merged = date_imgs.mosaic()
#         first = date_imgs.first()

#         return merged.set(
#             {
#                 "system:time_start": date_num,
#                 "system:footprint": first.get("system:footprint"),
#             }
#         )

#     merged_list = dates.map(merge_date_images)
#     return ee.ImageCollection(merged_list)

def merge_same_date_images(collection: ee.ImageCollection) -> ee.ImageCollection:
    """
    Merges images from the same date in a collection, handling edge effects by taking
    the mean value in overlapping areas.

    Args:
        collection (ee.ImageCollection): Input collection with potentially duplicate dates

    Returns:
        ee.ImageCollection: Collection with one image per unique date, with smooth transitions at edges
    """
    dates = collection.aggregate_array("system:time_start").distinct()

    def merge_date_images(date):
        date_num = ee.Number(date)
        date_imgs = collection.filter(ee.Filter.eq("system:time_start", date_num))
        
        # Calculate the mean and count of valid pixels
        mean_img = date_imgs.mean()
        count_img = date_imgs.count()
        
        # Use mosaic only where we have single image coverage
        mosaic_img = date_imgs.mosaic()
        
        # Combine: use mean where count > 1, mosaic elsewhere
        final_img = ee.Image(ee.Algorithms.If(
            count_img.gt(1),
            mean_img,
            mosaic_img
        ))
        
        first = date_imgs.first()
        return final_img.set({
            "system:time_start": date_num,
            "system:footprint": first.get("system:footprint"),
        })

    merged_list = dates.map(merge_date_images)
    return ee.ImageCollection(merged_list)