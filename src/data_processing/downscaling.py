import ee
from typing import Dict, List, Union, Optional
import logging


def resample_image(
    image: ee.Image,
    target_scale: int,
    bands_to_resample: List[str] = None,
    preserve_band_properties: bool = False,
) -> ee.Image:
    """
    Resample specific bands of an image to a target scale and add them as new bands
    with 'resampled_' prefix.

    Args:
        image (ee.Image): The input image to be resampled.
        target_scale (int): Target scale in meters.
        bands_to_resample (List[str], optional): List of band names to resample.
        preserve_band_properties (bool): Whether to resample bands separately to preserve
            band-specific properties. Defaults to False.

    Returns:
        ee.Image: Original image with additional resampled bands (prefixed with 'resampled_').
    """
    # Store original metadata
    original_projection = image.projection()
    original_scale = original_projection.nominalScale()

    if preserve_band_properties:
        # Resample each specified band separately
        def resample_band(band_name: str) -> ee.Image:
            band = image.select([band_name])
            resampled = band.reproject(
                crs=original_projection, scale=target_scale
            ).setDefaultProjection(crs=original_projection, scale=target_scale)
            # Rename the band with 'resampled_' prefix
            return resampled

        # Create a list of resampled band images
        resampled_bands = [resample_band(band) for band in bands_to_resample]

        # Combine all resampled bands into one image
        resampled = ee.Image.cat(resampled_bands)

    else:
        # Resample selected bands at once
        selected = image.select(bands_to_resample)
        resampled = selected.reproject(crs=original_projection, scale=target_scale)
        # Rename all bands with 'resampled_' prefix
        resampled = resampled


    return resampled.copyProperties(image).set(
        {
            "system:time_start": image.get("system:time_start"),
            "resampled": True,
            "original_scale": original_scale,
            "target_scale": target_scale,
            "original_projection": original_projection.wkt(),
            "resampled_bands": bands_to_resample,
        }
    )


class Downscaler:
    """
    A class to perform downscaling of Earth Engine images using regression-based methods.
    """

    def __init__(self, independent_bands: List[str], dependent_band: str):
        """
        Initialize the Downscaler with variable configurations.

        Args:
            independent_vars (List[str]): List of names for independent variables (e.g., ['NDVI', 'NDBI', 'NDWI']).
            dependent_var (str): Name of the dependent variable (e.g., 'ET').
            coefficients (Optional[Dict[str, float]]): Dictionary to store regression coefficients.
        """
        self.independent_bands = independent_bands
        self.dependent_band = dependent_band
        self.coefficients: Optional[Dict[str, float]] = None
        logging.basicConfig(level=logging.INFO)

    def compute_residuals(
        self, original_image: ee.Image, modeled_image: ee.Image
    ) -> ee.Image:
        """
        Computes the residuals between the original and the modeled image.

        Args:
            original_image (ee.Image): Original image.
            modeled_image (ee.Image): Modeled image based on regression.

        Returns:
            ee.Image: Residuals image.
        """
        return original_image.subtract(modeled_image).rename("residuals")

    def apply_gaussian_smoothing(self, image: ee.Image, radius: float = 1) -> ee.Image:
        """
        Applies Gaussian smoothing to an image.

        Args:
            image (ee.Image): Input image to smooth.
            radius (float): Radius of the Gaussian kernel in pixels.

        Returns:
            ee.Image: Smoothed image.
        """
        gaussian_kernel = ee.Kernel.gaussian(radius=radius, units="pixels")
        return image.resample("bicubic").convolve(gaussian_kernel)

    def perform_regression(
        self,
        independent_vars: ee.Image,
        dependent_var: ee.Image,
        geometry: ee.Geometry,
        scale: float,
    ) -> ee.Dictionary:
        """
        Performs linear regression using independent variables to predict the dependent variable.

        Args:
            independent_vars (ee.Image): Image containing bands of independent variables.
            dependent_var (ee.Image): Single-band image of the dependent variable.
            geometry (ee.Geometry): The geometry over which to perform the regression.
            scale (float): The scale at which to perform the regression.

        Returns:
            ee.Dictionary: The result of the linear regression.
        """
        independent_vars = independent_vars.select(self.independent_bands)
        independent_vars = ee.Image.constant(1).addBands(independent_vars)
        dependent_var = dependent_var.select([self.dependent_band])

        all_vars = independent_vars.addBands(dependent_var)
        numX = ee.List(independent_vars.bandNames()).length()

        try:
            regression = all_vars.reduceRegion(
                reducer=ee.Reducer.linearRegression(numX=numX, numY=1),
                geometry=geometry,
                scale=scale,
                maxPixels=1e13,
                tileScale=16,
            )
            return regression
        except ee.EEException as e:
            logging.error(f"Error in performing regression: {str(e)}")
            raise

    def extract_coefficients(self, regression_result: ee.Dictionary) -> None:
        """
        Extracts coefficients from the regression result and stores them in the class.

        Args:
            regression_result (ee.Dictionary): The result of the linear regression.
        """
        try:
            coefficients = ee.Array(regression_result.get("coefficients")).toList()
            self.coefficients = {
                "intercept": ee.Number(ee.List(coefficients.get(0)).get(0)),
                **{
                    f"slope_{var}": ee.Number(ee.List(coefficients.get(i + 1)).get(0))
                    for i, var in enumerate(self.independent_bands)
                },
            }
        except ee.EEException as e:
            logging.error(f"Error in extracting coefficients: {str(e)}")
            raise

    def apply_regression(self, independent_vars: ee.Image) -> ee.Image:
        """
        Applies the regression coefficients to the independent variables to predict the dependent variable.

        Args:
            independent_vars (ee.Image): Image containing bands of independent variables.

        Returns:
            ee.Image: The predicted dependent variable.
        """
        if not self.coefficients:
            raise ValueError(
                "Coefficients have not been extracted. Run extract_coefficients first."
            )

        try:
            predicted = ee.Image(self.coefficients["intercept"])
            for var in self.independent_bands:
                slope = self.coefficients[f"slope_{var}"]
                predicted = predicted.add(independent_vars.select(var).multiply(slope))

            return predicted.rename("predicted")
        except ee.EEException as e:
            logging.error(f"Error in applying regression: {str(e)}")
            raise

    def downscale(
        self,
        coarse_independent_vars: ee.Image,
        coarse_dependent_var: ee.Image,
        fine_independent_vars: ee.Image,
        geometry: ee.Geometry,
        resolution: int,
    ) -> ee.Image:
        """
        Performs the downscaling process with explicit projection handling.

        Args:
            coarse_independent_vars (ee.Image): Coarse resolution image with independent variables.
            coarse_dependent_var (ee.Image): Coarse resolution image with dependent variable.
            fine_independent_vars (ee.Image): Fine resolution image with independent variables.
            geometry (ee.Geometry): The geometry over which to perform the downscaling.
            resolution (int): The resolution of the coarse image.

        Returns:
            ee.Image: The downscaled image with consistent projection information.
        """
        try:
            # Get fine resolution properties
            fine_projection = fine_independent_vars.projection()
            fine_scale = fine_projection.nominalScale()
            fine_date = fine_independent_vars.date()

            # Store original coarse projection for reference
            coarse_projection = coarse_dependent_var.projection()
            coarse_scale = coarse_projection.nominalScale()

            # Perform regression at coarse resolution
            regression_result = self.perform_regression(
                coarse_independent_vars, coarse_dependent_var, geometry, resolution
            )
            self.extract_coefficients(regression_result)

            # Calculate residuals at coarse resolution
            coarse_modeled = self.apply_regression(
                coarse_independent_vars
            ).setDefaultProjection(crs=coarse_projection, scale=coarse_scale)

            residuals = self.compute_residuals(
                coarse_dependent_var, coarse_modeled
            ).setDefaultProjection(crs=coarse_projection, scale=coarse_scale)

            # Smooth residuals while maintaining coarse projection
            smoothed_residuals = self.apply_gaussian_smoothing(
                residuals
            ).setDefaultProjection(crs=coarse_projection, scale=coarse_scale)

            # Perform downscaling at fine resolution
            fine_downscaled = self.apply_regression(
                fine_independent_vars
            ).setDefaultProjection(crs=fine_projection, scale=fine_scale)

            # Reproject smoothed residuals to fine resolution
            smoothed_residuals_reprojected = smoothed_residuals.reproject(
                crs=fine_projection, scale=fine_scale
            )

            # Combine downscaled result with reprojected residuals
            final_downscaled = fine_downscaled.add(smoothed_residuals_reprojected)

            # Set final projection and metadata
            return (
                final_downscaled.rename("downscaled")
                .set(
                    {
                        "system:time_start": fine_date.millis(),
                        "original_coarse_scale": coarse_scale,
                        "final_scale": fine_scale,
                        "original_coarse_projection": coarse_projection.wkt(),
                        "final_projection": fine_projection.wkt(),
                    }
                )
                # .setDefaultProjection(crs=fine_projection, scale=fine_scale)
                # .reproject(crs=fine_projection, scale=fine_scale)
            )

        except Exception as e:
            logging.error(f"Error in downscaling process: {str(e)}")
            raise
