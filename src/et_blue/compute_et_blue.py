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
    et_blue_image_present: ee.Image, et_blue_image_past: ee.Image, threshold: float
) -> ee.Image:
    """
    Postprocess ET blue images based on current and past values and a threshold.

    Keeps the current ET blue value only if:
    1. The current value is >= threshold AND
    2. The current value plus any negative value from previous month is > 0
    Otherwise sets the pixel to 0.

    Args:
        et_blue_image_present (ee.Image): Current ET blue image.
        et_blue_image_past (ee.Image): Past ET blue image.
        threshold (float): Threshold value for ET blue.

    Returns:
        ee.Image: Postprocessed ET blue image.
    """
    date = et_blue_image_present.get("system:time_start")

    # Create condition mask:
    condition = (
        # Condition 1: Current value >= threshold
        et_blue_image_present.gte(threshold).And(
            # Condition 2: Current value + min(past_value, 0) > 0
            et_blue_image_present.add(et_blue_image_past.min(0)).gt(0)
        )
    )

    # Where condition is false, set to 0. Keep original values where condition is true
    return (
        et_blue_image_present.where(condition.Not(), 0)
        .rename("ET_blue")
        .set("system:time_start", date)
    )
