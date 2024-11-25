import ee
from typing import List, Dict, Any
import math


def add_temporal_bands(collection: ee.ImageCollection) -> ee.ImageCollection:
    """Add temporal and constant bands to each image in the collection. This is for harmonic regression.

    Args:
        collection: The input image collection.

    Returns:
        ee.ImageCollection: The collection with added bands."""

    def _add_bands(image: ee.Image) -> ee.Image:
        date = ee.Date(image.get("system:time_start"))
        years = date.difference(ee.Date("1970-01-01"), "year")

        projection = image.select([0]).projection()
        time_band = ee.Image(years).float().rename("t")
        constant_band = ee.Image.constant(1).rename("constant")

        return image.addBands(
            [
                time_band.setDefaultProjection(projection),
                constant_band.setDefaultProjection(projection),
            ]
        )

    return collection.map(_add_bands)


class HarmonicRegressor:
    def __init__(
        self,
        omega: float = 1.5,
        max_harmonic_order: int = 2,
        band_to_harmonize: str = "NDVI",
        parallel_scale: int = 2,
    ):
        self.omega = omega
        self.max_harmonic_order = max_harmonic_order
        self.band_to_harmonize = band_to_harmonize
        self.parallel_scale = parallel_scale
        self._regression_coefficients = None
        self._fitted_data = None

    @property
    def harmonic_component_names(self) -> List[str]:
        """Generate harmonic component names based on the max harmonic order.

        Returns:
            List[str]: List of harmonic component names.
        """
        return ["constant", "t"] + [
            f"{trig}{i}"
            for i in range(1, self.max_harmonic_order + 1)
            for trig in ["cos", "sin"]
        ]

    def fit(self, image_collection: ee.ImageCollection) -> "HarmonicRegressor":
        """
        Fit the harmonic regression model to the input image collection.

        Args:
            image_collection (ee.ImageCollection): Input image collection.

        Returns:
            HarmonicRegressor: Fitted model.

        Raises:
            TypeError: If image_collection is not an ee.ImageCollection.
            ValueError: If required bands are missing from the image collection.
        """
        if not isinstance(image_collection, ee.ImageCollection):
            raise TypeError("image_collection must be an ee.ImageCollection.")

        first_image = image_collection.first()
        required_bands = ["t", self.band_to_harmonize]
        missing_bands = [
            band
            for band in required_bands
            if not first_image.bandNames().contains(band).getInfo()
        ]
        if missing_bands:
            raise ValueError(
                f"Input ImageCollection is missing required bands: {missing_bands}"
            )

        harmonic_collection = self._prepare_harmonic_collection(image_collection)
        self._regression_coefficients = self._compute_regression_coefficients(
            harmonic_collection
        )
        self._fitted_data = self._compute_fitted_values(
            harmonic_collection, self._regression_coefficients
        )
        return self
    
    def fit2(self, image_collection: ee.ImageCollection) -> "HarmonicRegressor":
        """
        Fit the harmonic regression model to the input image collection.

        Args:
            image_collection (ee.ImageCollection): Input image collection.

        Returns:
            HarmonicRegressor: Fitted model.

        Raises:
            TypeError: If image_collection is not an ee.ImageCollection.
        """
        if not isinstance(image_collection, ee.ImageCollection):
            raise TypeError("image_collection must be an ee.ImageCollection.")

        # Prepare the harmonic collection and compute regression coefficients
        harmonic_collection = self._prepare_harmonic_collection(image_collection)
        self._regression_coefficients = self._compute_regression_coefficients(
            harmonic_collection
        )
        self._fitted_data = self._compute_fitted_values(
            harmonic_collection, self._regression_coefficients
        )
        return self


    def predict(self, image_collection: ee.ImageCollection) -> ee.ImageCollection:
        """
        Predict using the fitted harmonic regression model.

        Args:
            image_collection (ee.ImageCollection): Input image collection for prediction.

        Returns:
            ee.ImageCollection: Image collection with predicted values.

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        if self._regression_coefficients is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        harmonic_collection = self._prepare_harmonic_collection(image_collection)

        return self._compute_fitted_values(
            harmonic_collection, self._regression_coefficients
        )

    def get_phase_amplitude(self) -> ee.Image:
        """
        Calculate phase and amplitude from regression coefficients.

        Returns:
            ee.Image: Image with phase and amplitude bands.
        """
        if self._regression_coefficients is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self._calculate_phase_amplitude()

    def _prepare_harmonic_collection(
        self, image_collection: ee.ImageCollection
    ) -> ee.ImageCollection:
        """
        Prepare the input image collection for harmonic regression.

        Args:
            image_collection (ee.ImageCollection): Input image collection.

        Returns:
            ee.ImageCollection: Image collection with harmonic components added.
        """
        return image_collection.map(self._add_harmonic_components)

    def _add_harmonic_components(self, image: ee.Image) -> ee.Image:
        """Add harmonic component bands to the image.

        Args:
            image (ee.Image): Input image.

        Returns:
            ee.Image: Image with harmonic components added.
        """
        for i in range(1, self.max_harmonic_order + 1):
            omega_i = 2 * i * self.omega * math.pi
            time_radians = image.select("t").multiply(omega_i)
            cos_band = time_radians.cos().rename(f"cos{i}")
            sin_band = time_radians.sin().rename(f"sin{i}")
            image = image.addBands(cos_band).addBands(sin_band)
        return image

    def _compute_regression_coefficients(
        self, harmonic_collection: ee.ImageCollection
    ) -> ee.Image:
        """Compute regression coefficients using Earth Engine's linearRegression reducer.

        Args:
            harmonic_collection (ee.ImageCollection): Image collection with harmonic components.

        Returns:
            ee.Image: Image with regression coefficients.
        """
        regression_input_bands = ee.List(self.harmonic_component_names).add(
            self.band_to_harmonize
        )
        regression_result = harmonic_collection.select(regression_input_bands).reduce(
            ee.Reducer.linearRegression(
                numX=len(self.harmonic_component_names), numY=1
            ),
            parallelScale=self.parallel_scale,
        )
        return (
            regression_result.select("coefficients")
            .arrayProject([0])
            .arrayFlatten([self.harmonic_component_names])
        )

    def _compute_fitted_values(
        self, harmonic_collection: ee.ImageCollection, coefficients: ee.Image
    ) -> ee.ImageCollection:
        """Compute fitted values using the regression coefficients.

        Args:
            harmonic_collection (ee.ImageCollection): Image collection with harmonic components.
            coefficients (ee.Image): Image with regression coefficients.

        Returns:
            ee.ImageCollection: Image collection with fitted values.
        """

        def compute_fitted(image: ee.Image) -> ee.Image:
            fitted_values = (
                image.select(self.harmonic_component_names)
                .multiply(coefficients)
                .reduce(ee.Reducer.sum())
                .rename("fitted")
            )
            return image.addBands(fitted_values)

        return harmonic_collection.map(compute_fitted)

    def _calculate_phase_amplitude(self) -> ee.Image:
        """Calculate phase and amplitude from regression coefficients.

        Returns:
            ee.Image: Image with phase and amplitude bands.
        """
        phases = []
        amplitudes = []
        for i in range(1, self.max_harmonic_order + 1):
            cos_coeff = self._regression_coefficients.select(f"cos{i}")
            sin_coeff = self._regression_coefficients.select(f"sin{i}")
            phase = sin_coeff.atan2(cos_coeff).rename(f"phase{i}")
            amplitude = sin_coeff.hypot(cos_coeff).rename(f"amplitude{i}")
            phases.append(phase)
            amplitudes.append(amplitude)
        return ee.Image.cat(phases + amplitudes)
