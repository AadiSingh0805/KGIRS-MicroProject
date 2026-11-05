"""
Synthetic Reconnaissance Image Generator for Crisis GraphRAG
Generates drone HUD imagery with tactical overlays, thermal/optical styling,
damage indicators, and GPS telemetry for multimodal evaluation.
"""
import os
from PIL import Image, ImageDraw, ImageFont

def generate_drone_hud_image(filepath, title, status_text, severity_color, bg_gradient_start, bg_gradient_end, hazard_type):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    width, height = 640, 480
    img = Image.new("RGB", (width, height), bg_gradient_start)
    draw = ImageDraw.Draw(img)

    # Gradient background simulation
    for y in range(height):
        ratio = y / height
        r = int(bg_gradient_start[0] * (1 - ratio) + bg_gradient_end[0] * ratio)
        g = int(bg_gradient_start[1] * (1 - ratio) + bg_gradient_end[1] * ratio)
        b = int(bg_gradient_start[2] * (1 - ratio) + bg_gradient_end[2] * ratio)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # Grid lines (HUD tactical styling)
    for x in range(0, width, 40):
        draw.line([(x, 0), (x, height)], fill=(40, 60, 80), width=1)
    for y in range(0, height, 40):
        draw.line([(0, y), (width, y)], fill=(40, 60, 80), width=1)

    # Corner brackets (Drone Reticle)
    bracket_len = 35
    margin = 25
    col = (0, 240, 255) # Cyan HUD
    # Top-Left
    draw.line([(margin, margin), (margin + bracket_len, margin)], fill=col, width=3)
    draw.line([(margin, margin), (margin, margin + bracket_len)], fill=col, width=3)
    # Top-Right
    draw.line([(width - margin, margin), (width - margin - bracket_len, margin)], fill=col, width=3)
    draw.line([(width - margin, margin), (width - margin, margin + bracket_len)], fill=col, width=3)
    # Bottom-Left
    draw.line([(margin, height - margin), (margin + bracket_len, height - margin)], fill=col, width=3)
    draw.line([(margin, height - margin), (margin, height - margin - bracket_len)], fill=col, width=3)
    # Bottom-Right
    draw.line([(width - margin, height - margin), (width - margin - bracket_len, height - margin)], fill=col, width=3)
    draw.line([(width - margin, height - margin), (width - margin, height - margin - bracket_len)], fill=col, width=3)

    # Center Reticle
    cx, cy = width // 2, height // 2
    draw.line([(cx - 20, cy), (cx + 20, cy)], fill=col, width=2)
    draw.line([(cx, cy - 20), (cx, cy + 20)], fill=col, width=2)
    draw.ellipse([cx - 40, cy - 40, cx + 40, cy + 40], outline=col, width=1)

    # Specific Hazard Visuals
    if "flood" in hazard_type.lower():
        # Thermal water contours and contour lines
        for y_w in range(cy - 20, height - margin, 18):
            draw.line([(margin + 15, y_w), (width - margin - 15, y_w)], fill=(0, 160, 220), width=2)
            draw.line([(margin + 30, y_w + 8), (width - margin - 30, y_w + 8)], fill=(0, 90, 160), width=1)
        draw.text((cx - 100, cy + 40), "WATER DEPTH SENSOR: 75cm - 85cm", fill=(0, 240, 255))
        draw.text((cx - 100, cy + 60), "STATUS: COMPLETE SUBMERSION", fill=(239, 68, 68))
    elif "bridge" in hazard_type.lower() or "crack" in hazard_type.lower():
        # Diagonal crack fracture line
        points = [(cx - 120, cy + 80), (cx - 50, cy + 20), (cx, cy - 10), (cx + 40, cy - 60), (cx + 110, cy - 100)]
        draw.line(points, fill=(255, 60, 60), width=6)
        draw.line(points, fill=(255, 255, 255), width=2)
        draw.text((cx - 90, cy + 30), "EXPANSION JOINT SHEAR: 14cm", fill=(255, 60, 60))
    elif "landslide" in hazard_type.lower():
        # Debris mounds & warning
        draw.polygon([(cx - 150, height - margin), (cx - 80, cy + 30), (cx + 30, height - margin)], fill=(120, 80, 50))
        draw.polygon([(cx - 20, height - margin), (cx + 90, cy + 60), (cx + 170, height - margin)], fill=(150, 100, 60))
        draw.text((cx - 80, cy + 10), "11kV POWER CABLES DOWN", fill=(255, 220, 0))
    elif "hospital" in hazard_type.lower():
        # Hospital substation failure
        draw.rectangle([cx - 50, cy - 15, cx + 50, cy + 15], fill=(220, 50, 50))
        draw.rectangle([cx - 15, cy - 50, cx + 15, cy + 50], fill=(220, 50, 50))
        draw.text((cx - 90, cy + 60), "BASEMENT GENERATORS DROWNED", fill=(245, 158, 11))
    elif "clear" in hazard_type.lower():
        # Safe corridor green double lines
        draw.line([(cx - 80, height - margin), (cx - 40, margin + 80)], fill=(0, 255, 120), width=4)
        draw.line([(cx + 80, height - margin), (cx + 40, margin + 80)], fill=(0, 255, 120), width=4)
        draw.text((cx - 80, cy + 20), "CORRIDOR TRAVERSABLE: 100%", fill=(0, 255, 120))

    # Telemetry HUD Header & Footer
    draw.rectangle([margin, margin, width - margin, margin + 40], fill=(15, 23, 42))
    draw.text((margin + 12, margin + 8), f"RECON UAV-04 | SENSOR: THERMAL/RGB | {title}", fill=(240, 240, 255))
    draw.text((margin + 12, margin + 24), f"STATUS: {status_text}", fill=severity_color)

    draw.rectangle([margin, height - margin - 35, width - margin, height - margin], fill=(15, 23, 42))
    draw.text((margin + 12, height - margin - 28), "GPS: 19.0596 N, 72.8988 E | MUMBAI CHEMBUR | ALT: 145m | BEARING: 312° NW | ENCRYPTION: AES-256", fill=(148, 163, 184))

    img.save(filepath)
    print(f"Generated: {filepath}")

def generate_facility_satellite_image(filepath, facility_name, facility_type, lat, lon, telemetry_stats):
    """Generates satellite-styled tactical aerial reconnaissance for source and destination facilities."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    width, height = 640, 480
    
    # Earth/Urban satellite tones (deep slate, dark charcoal, asphalt and foliage texture)
    img = Image.new("RGB", (width, height), (16, 24, 36))
    draw = ImageDraw.Draw(img)

    # Simulated satellite ground terrain texture
    for y in range(0, height, 4):
        for x in range(0, width, 4):
            val = (x * 7 + y * 13) % 25
            draw.rectangle([x, y, x + 3, y + 3], fill=(20 + val // 2, 28 + val, 38 + val // 2))

    # Road network grid lines on ground
    draw.line([(0, 240), (width, 240)], fill=(50, 60, 75), width=18) # Main arterial boulevard
    draw.line([(0, 240), (width, 240)], fill=(220, 200, 100), width=2) # Centerline
    draw.line([(320, 0), (320, height)], fill=(45, 55, 70), width=12) # Cross street
    
    # Building blocks surrounding facility
    draw.rectangle([60, 60, 240, 200], fill=(28, 38, 52), outline=(60, 80, 110))
    draw.rectangle([360, 70, 560, 210], fill=(32, 42, 58), outline=(60, 80, 110))
    draw.rectangle([70, 280, 270, 410], fill=(26, 36, 48), outline=(60, 80, 110))

    # Target Facility Perimeter & Markings
    cx, cy = 440, 320
    if facility_type == "hospital":
        # Hospital compound with helipad
        draw.rectangle([330, 260, 580, 420], fill=(22, 45, 42), outline=(16, 185, 129), width=2)
        # Helipad H
        draw.ellipse([cx - 45, cy - 45, cx + 45, cy + 45], outline=(255, 255, 255), width=3)
        draw.line([(cx - 20, cy - 25), (cx - 20, cy + 25)], fill=(255, 255, 255), width=5)
        draw.line([(cx + 20, cy - 25), (cx + 20, cy + 25)], fill=(255, 255, 255), width=5)
        draw.line([(cx - 20, cy), (cx + 20, cy)], fill=(255, 255, 255), width=5)
        # Red cross indicator
        draw.rectangle([345, 275, 375, 285], fill=(239, 68, 68))
        draw.rectangle([355, 265, 365, 295], fill=(239, 68, 68))
    elif facility_type == "shelter":
        # Mega-shelter compound (stadium / club layout)
        draw.rectangle([330, 260, 580, 420], fill=(45, 35, 20), outline=(245, 158, 11), width=2)
        draw.ellipse([cx - 50, cy - 35, cx + 50, cy + 35], outline=(245, 158, 11), width=3)
        draw.text((cx - 40, cy - 8), "RELIEF ZONE", fill=(255, 220, 100))
    else:
        # Transit hub / Intersection
        draw.rectangle([330, 260, 580, 420], fill=(20, 35, 55), outline=(56, 189, 248), width=2)
        draw.ellipse([cx - 30, cy - 30, cx + 30, cy + 30], outline=(56, 189, 248), width=2)
        draw.text((cx - 35, cy - 8), "TRANSIT HUB", fill=(56, 189, 248))

    # Corner brackets (Satellite Targeting Reticle)
    col = (0, 240, 255)
    margin = 20
    b_len = 30
    draw.line([(margin, margin), (margin + b_len, margin)], fill=col, width=2)
    draw.line([(margin, margin), (margin, margin + b_len)], fill=col, width=2)
    draw.line([(width - margin, margin), (width - margin - b_len, margin)], fill=col, width=2)
    draw.line([(width - margin, margin), (width - margin, margin + b_len)], fill=col, width=2)
    draw.line([(margin, height - margin), (margin + b_len, height - margin)], fill=col, width=2)
    draw.line([(margin, height - margin), (margin, height - margin - b_len)], fill=col, width=2)
    draw.line([(width - margin, height - margin), (width - margin - b_len, height - margin)], fill=col, width=2)
    draw.line([(width - margin, height - margin), (width - margin, height - margin - b_len)], fill=col, width=2)

    # Satellite Telemetry Header
    draw.rectangle([margin, margin, width - margin, margin + 42], fill=(10, 15, 26))
    draw.text((margin + 10, margin + 6), f"SATELLITE RECON-9 | MULTISPECTRAL OPTICAL | {facility_name.upper()}", fill=(240, 245, 255))
    draw.text((margin + 10, margin + 22), f"TELEMETRY: {telemetry_stats}", fill=(56, 189, 248))

    # Satellite Telemetry Footer
    draw.rectangle([margin, height - margin - 32, width - margin, height - margin], fill=(10, 15, 26))
    draw.text((margin + 10, height - margin - 24), f"COORDINATES: {lat:.4f} N, {lon:.4f} E | CHEMBUR SECTOR | RESOLUTION: 0.35m/px", fill=(148, 163, 184))

    img.save(filepath)
    print(f"Generated satellite view: {filepath}")

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "damage_images")
    images_to_generate = [
        ("flash_flood_embarcadero.png", "SHELL COLONY LOWLAND INUNDATION", "IMPASSABLE: 75cm MONSOON FLOOD SURGE", (239, 68, 68), (10, 30, 60), (5, 15, 30), "flood"),
        ("bridge_pier_crack.png", "AMAR MAHAL FLYOVER PIER-4 INSPECTION", "CRITICAL DAMAGE: STRUCTURAL SHEAR", (239, 68, 68), (40, 20, 20), (20, 10, 10), "bridge"),
        ("hospital_submerged_basement.png", "INLAKS HOSPITAL SUBSTATION FLOOD", "CRITICAL: GENERATOR GRID DROWNED", (245, 158, 11), (30, 20, 40), (15, 10, 25), "hospital"),
        ("landslide_downed_lines.png", "SHELL-SUMAN 11kV CABLES DOWN", "ROAD BLOCK: 11kV ARCS & FLOOD DEBRIS", (239, 68, 68), (45, 30, 15), (20, 15, 10), "landslide"),
        ("viaduct_caution_debris.png", "SION-TROMBAY WATERLOGGED SPAN", "MODERATE CAUTION: 15cm WATER/DEBRIS", (245, 158, 11), (30, 35, 45), (15, 20, 25), "caution"),
        ("clear_corridor_highway.png", "EASTERN FREEWAY GREEN CORRIDOR", "NOMINAL: 100% TRAVERSABLE HIGHWAY", (16, 185, 129), (15, 45, 25), (10, 25, 15), "clear")
    ]
    for filename, title, status, color, g1, g2, hazard in images_to_generate:
        path = os.path.join(out_dir, filename)
        generate_drone_hud_image(path, title, status, color, g1, g2, hazard)

    # Generate source & destination facility satellite views
    facility_views = [
        ("source_chembur_naka.png", "Chembur Naka Transit Hub", "intersection", 19.0596, 72.8988, "Traffic Spine Normal | Elevation 14.0m | Surface Water 0cm"),
        ("dest_zen_hospital.png", "Zen Multi Speciality Hospital", "hospital", 19.0582, 72.9035, "28 Beds Available | 10 ICU Beds | Helipad Active | Power Nominal"),
        ("dest_surana_hospital.png", "Surana Sethia Hospital Trauma Centre", "hospital", 19.0542, 72.8912, "34 Beds Available | 12 ICU Beds | Generator Grid Active"),
        ("dest_rcf_shelter.png", "RCF Sports Complex Mega-Shelter", "shelter", 19.0465, 72.9095, "Capacity 750/1800 | 6 Days Food/Water | Medical Station Active"),
        ("dest_mankhurd_depot.png", "Mankhurd Logistics & Supply Depot", "supply_depot", 19.0565, 72.9315, "Heavy Fleet Staged | Boats: 14 | High-Clearance Trucks: 22")
    ]
    for filename, name, ftype, lat, lon, stats in facility_views:
        path = os.path.join(out_dir, filename)
        generate_facility_satellite_image(path, name, ftype, lat, lon, stats)
