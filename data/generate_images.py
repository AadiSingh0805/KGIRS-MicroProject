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
        # Water inundation ripple overlay
        for y_w in range(cy + 20, height - margin, 15):
            draw.line([(margin + 10, y_w), (width - margin - 10, y_w)], fill=(0, 180, 230), width=3)
        draw.rectangle([margin + 10, cy + 30, width - margin - 10, height - margin], fill=(0, 100, 180, 100), outline=(0, 220, 255))
    elif "bridge" in hazard_type.lower() or "crack" in hazard_type.lower():
        # Diagonal crack fracture line
        points = [(cx - 120, cy + 80), (cx - 50, cy + 20), (cx, cy - 10), (cx + 40, cy - 60), (cx + 110, cy - 100)]
        draw.line(points, fill=(255, 60, 60), width=6)
        draw.line(points, fill=(255, 255, 255), width=2)
    elif "landslide" in hazard_type.lower():
        # Debris mounds
        draw.polygon([(cx - 150, height - margin), (cx - 80, cy + 30), (cx + 30, height - margin)], fill=(120, 80, 50))
        draw.polygon([(cx - 20, height - margin), (cx + 90, cy + 60), (cx + 170, height - margin)], fill=(150, 100, 60))
        # High-voltage lightning bolt icon
        bolt = [(cx, cy - 70), (cx - 20, cy - 20), (cx - 5, cy - 20), (cx - 25, cy + 40), (cx + 15, cy - 10), (cx, cy - 10), (cx + 20, cy - 70)]
        draw.polygon(bolt, fill=(255, 220, 0), outline=(255, 255, 255))
    elif "hospital" in hazard_type.lower():
        # Hospital cross with warning
        draw.rectangle([cx - 60, cy - 20, cx + 60, cy + 20], fill=(220, 50, 50))
        draw.rectangle([cx - 20, cy - 60, cx + 20, cy + 60], fill=(220, 50, 50))
    elif "clear" in hazard_type.lower():
        # Safe corridor green double lines
        draw.line([(cx - 80, height - margin), (cx - 40, margin + 80)], fill=(0, 255, 120), width=4)
        draw.line([(cx + 80, height - margin), (cx + 40, margin + 80)], fill=(0, 255, 120), width=4)
        draw.line([(cx, height - margin), (cx, margin + 80)], fill=(0, 255, 120), width=2)

    # Telemetry HUD Header & Footer
    draw.rectangle([margin, margin, width - margin, margin + 40], fill=(15, 23, 42))
    draw.text((margin + 12, margin + 8), f"RECON UAV-04 | SENSOR: THERMAL/RGB | {title}", fill=(240, 240, 255))
    draw.text((margin + 12, margin + 24), f"STATUS: {status_text}", fill=severity_color)

    draw.rectangle([margin, height - margin - 35, width - margin, height - margin], fill=(15, 23, 42))
    draw.text((margin + 12, height - margin - 28), "GPS: 37.7850 N, 122.4100 W | ALT: 145m | BEARING: 312° NW | ENCRYPTION: AES-256", fill=(148, 163, 184))

    img.save(filepath)
    print(f"Generated: {filepath}")

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "damage_images")
    images_to_generate = [
        ("flash_flood_embarcadero.png", "EMBARCADERO SURGE INUNDATION", "IMPASSABLE: 75cm SEAWATER SURGE", (239, 68, 68), (10, 30, 60), (5, 15, 30), "flood"),
        ("bridge_pier_crack.png", "NORTH BAY SPAN PIER-4 INSPECTION", "CRITICAL DAMAGE: STRUCTURAL SHEAR", (239, 68, 68), (40, 20, 20), (20, 10, 10), "bridge"),
        ("hospital_submerged_basement.png", "HARBOR PAVILION BASEMENT INUNDATION", "CRITICAL: GENERATOR GRID DROWNED", (245, 158, 11), (30, 20, 40), (15, 10, 25), "hospital"),
        ("landslide_downed_lines.png", "16TH ST SLOPE COLLAPSE & ARCS", "ROAD BLOCK: 12kV ARCS & ROCKFALL", (239, 68, 68), (45, 30, 15), (20, 15, 10), "landslide"),
        ("viaduct_caution_debris.png", "CENTRAL CHANNEL VIADUCT", "MODERATE CAUTION: 12cm WATER/DEBRIS", (245, 158, 11), (30, 35, 45), (15, 20, 25), "caution"),
        ("clear_corridor_highway.png", "GEARY EXPRESSWAY GREEN CORRIDOR", "NOMINAL: 100% TRAVERSABLE HIGHWAY", (16, 185, 129), (15, 45, 25), (10, 25, 15), "clear")
    ]
    for filename, title, status, color, g1, g2, hazard in images_to_generate:
        path = os.path.join(out_dir, filename)
        generate_drone_hud_image(path, title, status, color, g1, g2, hazard)
