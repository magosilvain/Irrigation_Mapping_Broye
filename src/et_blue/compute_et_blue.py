import ee


def compute_et_blue(et_total: ee.Image, et_green: ee.Image) -> ee.Image:
    """
    Compute ET blue by subtracting ET green from total ET.
    Apply a threshold to ET blue values.

    Args:
        et_total (ee.Image): Image containing total ET values.
        et_green (ee.Image): Image containing ET green values.
    Returns:
        ee.Image: Image containing ET blue values above the threshold.
    """

    date = et_total.get("system:time_start")

    et_blue = et_total.subtract(et_green).rename("ET_blue")

    return et_blue.set("system:time_start", date)


def compute_volumetric_et_blue(et_blue: ee.Image) -> ee.Image:
    """
    Convert ET blue from mm to cubic meters.

    Args:
        et_blue (ee.Image): Image containing ET blue values in mm.

    Returns:
        ee.Image: Image containing ET blue values in cubic meters.
    """

    date = et_blue.get("system:time_start")
    # Convert mm to m (divide by 1000) and multiply by pixel area
    return (
        et_blue.multiply(0.001)
        .multiply(ee.Image.pixelArea())
        .rename("ET_blue_m3")
        .set("system:time_start", date)
    )


def postprocess_et_blue(
    et_blue_image_present: ee.Image,
    et_blue_image_past: ee.Image,
    threshold: ee.Image,
) -> ee.Image:
    """
    Postprocess ET blue images based on current and past values and a threshold.

    Args:
        et_blue_image_present (ee.Image): Current ET blue image
        et_blue_image_past (ee.Image): Past ET blue image
        threshold (ee.Image): Threshold value for ET blue

    Returns:
        ee.Image: Postprocessed ET blue image with values set to 0 where conditions not met
    """
    # Get the date from present image
    date = et_blue_image_present.get("system:time_start")

    # Calculate the minimum of past image and 0
    past_negative = et_blue_image_past.min(ee.Image.constant(0))

    # # Build combined condition using ee.Algorithms.If()
    # condition = ee.Algorithms.If(
    #     ee.Algorithms.IsEqual(threshold, None),
    #     ee.Image.constant(0),
    #     et_blue_image_present.gte(threshold).And(
    #         et_blue_image_present.add(past_negative).gt(0)
    #     ),
    # )

    condition=et_blue_image_present.gte(threshold).And(
            et_blue_image_present.add(past_negative).gt(0)
        )

    # Apply mask and return
    return (
        et_blue_image_present.updateMask(condition)
        .unmask(0)
        .rename("ET_blue")
        .set("system:time_start", date)
    )
