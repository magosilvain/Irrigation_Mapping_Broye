import ee
from typing import Set, List
from utils.ee_utils import normalize_string_client, normalize_string_server


def get_crops_to_exclude() -> Set[str]:
    """
    Returns a set of crop types to exclude from irrigation analysis.
    This includes permanent cultures, natural areas, ecological compensation areas,
    and other non-irrigated agricultural areas.

    Returns:
        Set[str]: A set of crop names to exclude.
    """
    exclude_set = {
        # Indoor production
        "Landwirtschaftliche Produktion in Gebäuden (z. B. Champignon, Brüsseler)",
        # Permanent fruit and vine cultures
        "Andere Obstanlagen (Kiwis, Holunder usw.)",
        "Baumschule von Forstpflanzen ausserhalb der Forstzone",
        "Christbäume",
        "Hochstamm-Feldobstbäume (Punkte oder Flächen)",
        "Obstanlagen (Äpfel)",
        "Obstanlagen (Birnen)",
        "Obstanlagen (Steinobst)",
        "Obstanlagen aggregiert",
        "Obstanlagen Steinobst",
        "Obstanlagen Äpfel",
        "Reben",
        "Rebflächen mit natürlicher Artenvielfalt",
        "Übrige Baumschulen (Rosen, Früchte, usw.)",
        "Baumschulen von Reben",
        "Trüffelanlagen",
        # Natural areas and water bodies
        "Wald",
        "Wassergräben, Tümpel, Teiche",
        "Wassergraben, Tümpel, Teiche",
        "Unbefestigte, natürliche Wege",
        # Hedges and ecological areas
        "Hecken-, Feld- und Ufergehölze (mit Pufferstreifen)",
        "Hecken-, Feld- und Ufergehölze (mit Krautsaum)",
        "Hecken-, Feld und Ufergehölze (mit Krautsaum)",
        "Hecken-, Feld- und Ufergehölze (mit Puf.str.)",
        "Hecken, Feld-, Ufergehölze mit Krautsaum",
        "Hecken-, Feld und Ufergehölz (reg. BFF)",
        "Hecken-, Feld- und Ufergehölze (mit Pufferstreifen) (regionsspezifische Biodiversitätsförderfläche)",
        # Permanent and extensive grasslands
        "Extensiv genutzte Wiesen (ohne Weiden)",
        "Extensiv genutzte Weiden",
        "Wenig intensiv genutzte Wiesen (ohne Weiden)",
        "Wenig intensiv gen. Wiesen (ohne Weiden)",
        "Streueflächen in der LN",
        "Weide (Heimweiden, Üb. Weide ohne Sö.weiden)",
        "Weiden (Heimweiden, übrige Weiden ohne Sömmerungsweiden)",
        "Weide (Heimweiden, üb. Weide ohne Sö.geb.)",
        "Sömmerungsweiden",
        "Heuwiesen im Sömmerungsgebiet, Übrige Wiesen",
        # Various grassland types
        "Üb. Grünfläche (Dauergrünfläche) beitragsbe.",
        "Übrige Grünfläche (Dauergrünfläche), beitragsberechtigt",
        "Üb. Grünfläche beitragsberechtigt",
        "Üb. Grünfläche nicht beitragsberechtigt",
        "Übrige Grünfläche (Dauergrünfläche), nicht beitragsberechtigt",
        "Übr. Kunstwiese bb z.B. Schweine-, Geflügelwe.",
        "Übrige Kunstwiese (Schweine-, Geflügelweide)",
        "Übrige Kunstwiese, beitragsberechtigt (z.B. Schweineweide, Geflügelweide)",
        # Ecological compensation areas
        "Buntbrache",
        "Rotationsbrache",
        "Saum auf Ackerflächen",
        "Saum auf Ackerfläche",
        "Blühstreifen für Bestäuber und andere Nützlinge",
        "Blühstreifen für Bestäuber und and. Nützlinge",
        "Ackerschonstreifen",
        "Ruderalflächen, Steinhaufen und -wälle",
        # Riparian zones
        "Uferwiese (ohne Weiden) entlang von Fliessgewässern",
        "Uferwiese (o.Wei.) entlang von Fliessgew.",
        "Uferwiesen (ohne Weiden) entlang von Fliessg.",
        "Uferwiesen entlang von Fliessgewässern (ohne Weiden)",
        "Uferwiese (ohne Weiden) entlang von Fließgewässern",
        # Special cultures
        "Ziersträucher, Ziergehölze und Zierstauden",
        "Hausgärten",
        "Mehrjährige nachwachsende Rohstoffe (Chinaschilf, usw.)",
        # Non-agricultural areas and infrastructure
        "Flächen ohne landwirtschaftliche Hauptzweckbestimmung",
        "Flächen ohne landwirtschaftliche Hauptzweckbestimmung (erschlossenes Bauland, Spiel-, Reit-, Camping-, Golf-, Flug- und Militärplätze oder ausgemarchte Bereiche von Eisenbahnen, öffentlichen Strassen und Gewässern)",
        "Fläche ohne landw. Hauptzweckbestimmung",
        "Übrige unproduktive Flächen (z.B. gemulchte Flächen, stark verunkrautete Flächen, Hecken ohne Pufferstreifen)",
        "übrige Unproduktive Flächen (z.B. gemulchte Flächen, stark verunkraute Flächen, Hecke ohne Pufferstreifen)",
        "Übrige Flächen ausserhalb der LN und SF",
        # Non-agricultural or unused areas
        "Übrige Flächen innerhalb der LN, nicht beitragsberechtigt",
        "Übrige Ackergewächse (nicht beitragsber.)",
        "übrige offene Ackerfläche, nicht beitragsberechtigt",
        "Übrige Flächen mit Dauerkulturen, beitragsberechtigt",
        "Übrige Flächen mit Dauerkulturen, nicht beitragsberechtigt",
    }

    return {normalize_string_client(crop) for crop in exclude_set}


def get_rainfed_reference_crops() -> set:
    """
    Returns a set of crop types to use as rainfed reference.
    """
    # return {"Kunstwiesen (ohne Weiden)", "Übrige Dauerwiesen (ohne Weiden)"}

    # TODO: Change when not validating
    rainfed_reference_set = {
        "Extensiv genutzte Weiden",
        "Weiden (Heimweiden, übrige Weiden ohne Sömmerungsweiden)",
        "Übrige Dauerwiesen (ohne Weiden)",
        "Übrige Grünfläche (Dauergrünfläche), beitragsberechtigt",
        "Übrige Grünfläche (Dauergrünflächen), nicht beitragsberechtigt",
        "Waldweiden (ohne bewaldete Fläche)",
    }

    return {normalize_string_client(crop) for crop in rainfed_reference_set}


def get_winter_crops() -> Set[str]:
    """
    Returns a set of winter crop types based on the agricultural classification.
    This includes explicit winter crops, traditional winter cereals, and other potential winter crops.

    Returns:
        Set[str]: A set of winter crop names.
    """
    return {
        # Explicit winter crops
        "Wintergerste",
        "Winterweizen (ohne Futterweizen der Sortenliste swiss granum)",
        "Winterraps zur Speiseölgewinnung",
        # Traditional winter cereals
        "Dinkel",
        "Emmer, Einkorn",
        "Roggen",
        "Triticale",
        # Other potential winter crops
        "Futterweizen gemäss Sortenliste swiss granum",
    }


def add_double_cropping_info(
    feature_collection: ee.FeatureCollection, double_cropping_image: ee.Image, scale=10
) -> ee.FeatureCollection:
    """
    Adds double cropping information to each feature based on the median value of pixels within the feature.

    Args:
        feature_collection (ee.FeatureCollection): The input feature collection of crop fields.
        double_cropping_image (ee.Image): Image with 'isDoubleCropping' band (1 for double-cropped, 0 for single-cropped).
        scale (int): The scale to use for reducing the image.

    Returns:
        ee.FeatureCollection: Updated feature collection with 'isDoubleCropped' property.
    """

    filled_image = double_cropping_image.unmask(0)

    def add_double_crop_property(feature):
        median_value = (
            filled_image.select("isDoubleCropping")
            .reduceRegion(
                reducer=ee.Reducer.median(),
                geometry=feature.geometry(),
                scale=scale,
            )
            .get("isDoubleCropping")
        )

        return feature.set("isDoubleCropped", median_value)

    return feature_collection.map(add_double_crop_property)


def create_crop_filters(crops_to_exclude: set, rainfed_crops: set) -> tuple:
    """
    Creates filters for excluding crops and identifying rainfed reference crops.

    Args:
        crops_to_exclude (set): Set of crop names to exclude.
        rainfed_crops (set): Set of crop names to use as rainfed reference.

    Returns:
        tuple: A tuple containing two ee.Filter objects (exclude_condition, rainfed_condition).
    """
    exclude_condition = ee.Filter.inList("nutzung_normalized", list(crops_to_exclude)).Not()
    rainfed_condition = ee.Filter.And(
        ee.Filter.inList("nutzung_normalized", list(rainfed_crops)),
        ee.Filter.eq("isDoubleCropped", 0),  # Exclude double-cropped fields
    )
    return exclude_condition, rainfed_condition


def filter_crops(
    feature_collection: ee.FeatureCollection,
    exclude_filter: ee.Filter,
    rainfed_filter: ee.Filter,
) -> tuple:
    """
    Filters a feature collection based on crop type conditions.

    Args:
        feature_collection (ee.FeatureCollection): The input feature collection.
        exclude_filter (ee.Filter): Filter for excluding certain crop types.
        rainfed_filter (ee.Filter): Filter for identifying rainfed reference crops.

    Returns:
        tuple: A tuple containing two ee.FeatureCollection objects (filtered_fields, rainfed_fields).
    """
    filtered_fields = feature_collection.filter(exclude_filter)
    rainfed_fields = feature_collection.filter(rainfed_filter)
    return filtered_fields, rainfed_fields


def get_unique_nutzung(
    feature_collection: ee.FeatureCollection, nutzung_field_name: str = "nutzung"
) -> ee.List:
    """
    Gets all unique values for the 'nutzung' attribute in a FeatureCollection.

    Args:
        feature_collection (ee.FeatureCollection): The input FeatureCollection containing 'nutzung' property.
        nutzung_field_name (str): The name of the 'nutzung' field.

    Returns:
        ee.List: A list of unique 'nutzung' values.
    """
    return feature_collection.distinct(nutzung_field_name).aggregate_array(
        nutzung_field_name
    )


# Example usage
# def main():
#     # Load your feature collection and double cropping image
#     nutzung_collection = ee.FeatureCollection("path/to/your/nutzung/collection")
#     double_cropping_image = ee.Image("path/to/your/double_cropping_image")

#     # Add double cropping information to the feature collection
#     nutzung_collection_with_double_crop = add_double_cropping_info(
#         nutzung_collection, double_cropping_image
#     )

#     crops_to_exclude = get_crops_to_exclude()
#     rainfed_crops = get_rainfed_reference_crops()

#     exclude_filter, rainfed_filter = create_crop_filters(
#         crops_to_exclude, rainfed_crops
#     )

#     filtered_fields, rainfed_fields = filter_crops(
#         nutzung_collection_with_double_crop, exclude_filter, rainfed_filter
#     )

#     print("Filtered fields count:", filtered_fields.size().getInfo())
#     print("Rainfed reference fields count:", rainfed_fields.size().getInfo())