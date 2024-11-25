import ee

# Constants for base URLs as ee.String objects
WAPOR_DEKADAL_BASE_URL = ee.String(
    "gs://fao-gismgr-wapor-3-data/DATA/WAPOR-3/MAPSET/L1-AETI-D/WAPOR-3.L1-AETI-D."
)
WAPOR_MONTHLY_BASE_URL = ee.String(
    "gs://fao-gismgr-wapor-3-data/DATA/WAPOR-3/MAPSET/L1-AETI-M/WAPOR-3.L1-AETI-M."
)


def load_wapor_et_data(
    first_year: int, last_year: int, frequency: str = "dekadal"
) -> ee.ImageCollection:
    """
    Load and process WAPOR ET data for a range of years with specified frequency.

    Args:
        first_year (int): The first year to process.
        last_year (int): The last year to process.
        frequency (str): The frequency of data to load. Either "dekadal" or "monthly".

    Returns:
        ee.ImageCollection: Processed WAPOR ET data.
    """

    if first_year > last_year:
        raise ValueError("first_year must be less than or equal to last_year")

    def build_url(
        freq: str, yr: ee.Number, month: ee.Number, dekad: ee.Number = None
    ) -> ee.String:
        """
        Constructs the URL for the GeoTIFF based on frequency and date parameters.

        Args:
            freq (str): Frequency type, either "dekadal" or "monthly".
            yr (ee.Number): Year.
            month (ee.Number): Month.
            dekad (ee.Number, optional): Dekad number. Required if freq is "dekadal".

        Returns:
            ee.String: Constructed URL.
        """
        if freq == "dekadal":
            if dekad is None:
                raise ValueError("Dekad number must be provided for dekadal frequency.")
            return (
                WAPOR_DEKADAL_BASE_URL.cat(yr.format("%04d"))
                .cat("-")
                .cat(month.format("%02d"))
                .cat("-D")
                .cat(dekad.format("%d"))
                .cat(".tif")
            )
        elif freq == "monthly":
            return (
                WAPOR_MONTHLY_BASE_URL.cat(yr.format("%04d"))
                .cat("-")
                .cat(month.format("%02d"))
                .cat(".tif")
            )
        else:
            raise ValueError("Invalid frequency. Choose 'dekadal' or 'monthly'.")

    def process_image(url: ee.String, time_start: ee.Date) -> ee.Image:
        """
        Loads and processes the GeoTIFF image.

        Args:
            url (ee.String): URL of the GeoTIFF.
            time_start (ee.Date): Start date for the image.

        Returns:
            ee.Image: Processed image with metadata.
        """
        return (
            ee.Image.loadGeoTIFF(url)
            .multiply(0.1)
            .int()
            .set("system:time_start", time_start.millis())
            .rename("ET")
        )

    if frequency == "dekadal":
        # Each month has 3 dekads
        collection = ee.ImageCollection(
            ee.List.sequence(first_year, last_year)
            .map(
                lambda yr: ee.List.sequence(1, 12)
                .map(
                    lambda month: ee.List.sequence(1, 3).map(
                        lambda dekad: process_image(
                            build_url(
                                "dekadal",
                                ee.Number(yr),
                                ee.Number(month),
                                ee.Number(dekad),
                            ),
                            ee.Date.fromYMD(
                                yr,
                                month,
                                (ee.Number(dekad).subtract(1).multiply(10)).add(1).add(5),
                            ),
                        )
                        .set("Month", month)
                        .set("Year", yr)
                    )
                )
                .flatten()
            )
            .flatten()
        )
    elif frequency == "monthly":
        collection = ee.ImageCollection(
            ee.List.sequence(first_year, last_year)
            .map(
                lambda yr: ee.List.sequence(1, 12).map(
                    lambda month: process_image(
                        build_url("monthly", ee.Number(yr), ee.Number(month)),
                        ee.Date.fromYMD(yr, month, 16),
                    )
                    .set("Month", month)
                    .set("Year", yr)
                )
            )
            .flatten()
        )
    else:
        raise ValueError("Frequency must be either 'dekadal' or 'monthly'")

    return collection.sort("system:time_start")
