"""Vehicle identifiers and configured flood clearances for this simulation."""
VEHICLE_PROFILES = {
    "ambulance": {"max_flood_cm": 20.0, "label": "Ambulance (<= 20cm flood)"},
    "high_clearance_truck": {"max_flood_cm": 60.0, "label": "Rescue Truck (<= 60cm flood)"},
    "rescue_boat": {"max_flood_cm": 999.0, "label": "Rescue Boat (Deep-water capable)"},
    "civilian_car": {"max_flood_cm": 15.0, "label": "Civilian Car (<= 15cm flood)"},
}


def normalize_vehicle_type(vehicle_type):
    vehicle_type = {"heavy_truck": "high_clearance_truck"}.get(vehicle_type, vehicle_type)
    if vehicle_type not in VEHICLE_PROFILES:
        raise ValueError(f"Unknown vehicle profile: {vehicle_type!r}")
    return vehicle_type
