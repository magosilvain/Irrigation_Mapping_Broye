import ee
import pandas as pd
from datetime import datetime


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
