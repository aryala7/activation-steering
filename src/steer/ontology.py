"""Supported categorical slots for controlled response corruption.

This deliberately small ontology covers hotel/restaurant area and price only.
"""

from types import MappingProxyType

SLOT_VALUES = MappingProxyType({
    "restaurant-area": ("centre", "east", "north", "south", "west"),
    "restaurant-pricerange": ("cheap", "expensive", "moderate"),
    "hotel-area": ("centre", "east", "north", "south", "west"),
    "hotel-pricerange": ("cheap", "expensive", "moderate"),
})
