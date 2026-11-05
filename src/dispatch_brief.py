"""Generate a printable PDF dispatch brief from the current GraphRAG state."""

import io
import math
from datetime import datetime, timezone
from html import escape
from urllib.request import Request, urlopen

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


def _lon_to_tile(lon, zoom):
    return (lon + 180.0) / 360.0 * (1 << zoom)


def _lat_to_tile(lat, zoom):
    lat_rad = math.radians(lat)
    return (
        1.0
        - math.asinh(math.tan(lat_rad)) / math.pi
    ) / 2.0 * (1 << zoom)


def _fetch_real_map_image(coords, is_aerial=False):
    """Download and stitch an Esri satellite basemap for the route coordinates."""
    if len(coords) == 0:
        return None

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None

    lats = [item[0] for item in coords]
    lons = [item[1] for item in coords]
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)
    zoom = 14

    lat_span = max(lat_max - lat_min, 0.0025)
    lon_span = max(lon_max - lon_min, 0.0035)
    lat_min = max(-85.0511, lat_min - lat_span * 0.3)
    lat_max = min(85.0511, lat_max + lat_span * 0.3)
    lon_min -= lon_span * 0.3
    lon_max += lon_span * 0.3

    tile_x_min = math.floor(_lon_to_tile(lon_min, zoom))
    tile_x_max = math.floor(_lon_to_tile(lon_max, zoom))
    tile_y_min = math.floor(_lat_to_tile(lat_max, zoom))
    tile_y_max = math.floor(_lat_to_tile(lat_min, zoom))
    tile_count = (tile_x_max - tile_x_min + 1) * (tile_y_max - tile_y_min + 1)
    if tile_count > 16:
        zoom -= 1
        tile_x_min = math.floor(_lon_to_tile(lon_min, zoom))
        tile_x_max = math.floor(_lon_to_tile(lon_max, zoom))
        tile_y_min = math.floor(_lat_to_tile(lat_max, zoom))
        tile_y_max = math.floor(_lat_to_tile(lat_min, zoom))

    tile_size = 256
    canvas_width = (tile_x_max - tile_x_min + 1) * tile_size
    canvas_height = (tile_y_max - tile_y_min + 1) * tile_size
    if canvas_width > 1600 or canvas_height > 1600:
        return None

    stitched = Image.new("RGB", (canvas_width, canvas_height), "#10202A")

    for tile_y in range(tile_y_min, tile_y_max + 1):
        for tile_x in range(tile_x_min, tile_x_max + 1):
            url = (
                "https://server.arcgisonline.com/ArcGIS/rest/services/"
                f"World_Imagery/MapServer/tile/{zoom}/{tile_y}/{tile_x}"
            )
            request = Request(url, headers={"User-Agent": "AegisCrisisNav/1.0"})
            try:
                with urlopen(request, timeout=8) as response:
                    tile_bytes = response.read()
                tile_image = Image.open(io.BytesIO(tile_bytes)).convert("RGB")
            except Exception:
                return None
            stitched.paste(
                tile_image,
                ((tile_x - tile_x_min) * tile_size, (tile_y - tile_y_min) * tile_size),
            )

    def image_point(lat, lon):
        x = (_lon_to_tile(lon, zoom) - tile_x_min) * tile_size
        y = (_lat_to_tile(lat, zoom) - tile_y_min) * tile_size
        return int(x), int(y)

    draw = ImageDraw.Draw(stitched)
    screen_points = [image_point(lat, lon) for lat, lon, _ in coords]
    route_color = "#F59E0B" if is_aerial else "#00D7FF"
    route_width = 6

    if len(screen_points) > 1:
        draw.line(screen_points, fill=route_color, width=route_width, joint="curve")

    for index, point in enumerate(screen_points):
        radius = 8 if index in (0, len(screen_points) - 1) else 4
        fill = "#34D399" if index == len(screen_points) - 1 else "#00D7FF"
        if index == 0:
            fill = "#F8FAFC"
        draw.ellipse(
            (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
            fill=fill,
            outline="#FFFFFF",
            width=2,
        )

    output = io.BytesIO()
    stitched.save(output, format="JPEG", quality=92)
    return output.getvalue()


def _node_label(graph_store, node_id):
    if not node_id:
        return "Unknown location"
    return graph_store.graph.nodes.get(node_id, {}).get("name", node_id)


def _paragraph(canvas, text, x, y, width, size=8.5, leading=11, color="#E2E8F0"):
    from reportlab.platypus import Paragraph
    from reportlab.lib.styles import ParagraphStyle

    style = ParagraphStyle(
        name="BriefParagraph",
        fontName="Helvetica",
        fontSize=size,
        leading=leading,
        textColor=colors.HexColor(color),
        spaceAfter=0,
    )
    paragraph = Paragraph(escape(str(text)).replace("\n", "<br/>"), style)
    _, height = paragraph.wrap(width, 1000)
    paragraph.drawOn(canvas, x, y - height)
    return y - height - 5


def build_dispatch_brief(state, graph_store, selected_settings=None):
    """Build a PDF dispatch brief and return its bytes."""
    buffer = io.BytesIO()
    page_size = landscape(A4)
    pdf = canvas.Canvas(buffer, pagesize=page_size)
    pdf.setTitle("Aegis CrisisNav Dispatch Brief")
    width, height = page_size

    route = state.get("primary_route") or {}
    segments = route.get("segments") or []
    evidence = state.get("retrieved_incidents") or []
    alternatives = state.get("alternative_routes") or []
    settings = selected_settings or {}

    origin_id = state.get("extracted_origin") or route.get("origin") or route.get("source")
    destination_id = state.get("extracted_destination") or route.get("destination") or route.get("target")
    origin_label = _node_label(graph_store, origin_id)
    destination_label = _node_label(graph_store, destination_id)
    vehicle = settings.get("vehicle_type", state.get("vehicle_type", "ambulance"))
    avoid_hazards = settings.get("avoid_hazards", state.get("avoid_hazards", True))
    query = settings.get("query", state.get("user_query", "No query recorded"))
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    pdf.setFillColor(colors.HexColor("#080C14"))
    pdf.rect(0, 0, width, height, fill=1, stroke=0)

    pdf.setFillColor(colors.HexColor("#38BDF8"))
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(36, height - 42, "AEGIS CRISISNAV  |  DISPATCH BRIEF")
    pdf.setFillColor(colors.HexColor("#94A3B8"))
    pdf.setFont("Helvetica", 8)
    pdf.drawRightString(width - 36, height - 38, timestamp)

    left = 36
    right = width - 36
    map_x, map_y, map_w, map_h = 36, 190, 360, height - 270
    details_x, details_y = 420, height - 72
    details_w = right - details_x

    # Real satellite map with the selected route overlaid.
    pdf.setStrokeColor(colors.HexColor("#26384D"))
    pdf.setFillColor(colors.HexColor("#0E1726"))
    pdf.roundRect(map_x, map_y, map_w, map_h, 8, fill=1, stroke=1)
    pdf.setFillColor(colors.HexColor("#94A3B8"))
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(map_x + 12, map_y + map_h - 20, "REAL MAP  |  SELECTED ROUTE")

    node_ids = route.get("path") or []
    coords = []
    for node_id in node_ids:
        node = graph_store.graph.nodes.get(node_id, {})
        lat = node.get("lat")
        lon = node.get("lon")
        if lat is not None and lon is not None:
            coords.append((float(lat), float(lon), node_id))

    if coords:
        real_map = _fetch_real_map_image(coords, route.get("is_aerial_evacuation", False))
        if real_map:
            image = ImageReader(io.BytesIO(real_map))
            image_y = map_y + 18
            image_h = map_h - 50
            pdf.drawImage(
                image,
                map_x + 12,
                image_y,
                width=map_w - 24,
                height=image_h,
                preserveAspectRatio=True,
                anchor="c",
                mask="auto",
            )
            pdf.setFillColor(colors.HexColor("#94A3B8"))
            pdf.setFont("Helvetica", 6.5)
            pdf.drawRightString(map_x + map_w - 14, map_y + 7, "Basemap: Esri World Imagery")
        else:
            pdf.setFillColor(colors.HexColor("#F87171"))
            pdf.setFont("Helvetica-Bold", 10)
            pdf.drawCentredString(map_x + map_w / 2, map_y + map_h / 2, "REAL BASEMAP UNAVAILABLE")
            pdf.setFillColor(colors.HexColor("#94A3B8"))
            pdf.setFont("Helvetica", 8)
            pdf.drawCentredString(
                map_x + map_w / 2,
                map_y + map_h / 2 - 16,
                "Check network access and regenerate the brief.",
            )
    else:
        pdf.setFillColor(colors.HexColor("#F87171"))
        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawCentredString(map_x + map_w / 2, map_y + map_h / 2, "NO ROUTE MAP AVAILABLE")
        pdf.setFillColor(colors.HexColor("#94A3B8"))
        pdf.setFont("Helvetica", 9)
        pdf.drawCentredString(map_x + map_w / 2, map_y + map_h / 2 - 18, "No safe route or route segments were produced.")

    # Settings and briefing.
    y = details_y
    pdf.setFillColor(colors.HexColor("#F8FAFC"))
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(details_x, y, "CURRENT BRIEFING")
    y -= 18

    status = "SIMULATED AERIAL ALTERNATIVE" if route.get("is_aerial_evacuation") else route.get("status", "NO ROUTE FOUND")
    briefing_lines = [
        f"Origin: {origin_label}",
        f"Destination: {destination_label}",
        f"Vehicle: {vehicle}",
        f"Hazard avoidance: {'enabled' if avoid_hazards else 'disabled'}",
        f"Route status: {status}",
        f"Distance: {route.get('total_distance_km', 0)} km",
        f"Estimated transit: {route.get('estimated_travel_time_min', 0)} min",
        f"Supporting search results: {len(evidence)} incident report(s)",
    ]
    for line in briefing_lines:
        pdf.setFillColor(colors.HexColor("#CBD5E1"))
        pdf.setFont("Helvetica", 9)
        pdf.drawString(details_x, y, line)
        y -= 15

    y -= 3
    y = _paragraph(pdf, f"Selected request: {query}", details_x, y, details_w, size=8, leading=10)

    if evidence:
        y = _paragraph(pdf, "Supporting incident reports:", details_x, y, details_w, size=9, leading=11, color="#A78BFA")
        for report in evidence[:5]:
            title = report.get("incident_type", report.get("report_id", "Incident"))
            text = report.get("text", "No report text available.")
            y = _paragraph(pdf, f"- {title}: {text}", details_x, y, details_w, size=8, leading=10)
    else:
        y = _paragraph(
            pdf,
            "No supporting incident reports were retrieved. Routing and restrictions still use the current incident registry snapshot.",
            details_x,
            y,
            details_w,
            size=8,
            leading=10,
        )

    # Directions continue below the map across the lower half.
    lower_y = map_y - 18
    pdf.setFillColor(colors.HexColor("#F8FAFC"))
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(left, lower_y, "DIRECTIONS")
    lower_y -= 18

    if segments:
        for index, segment in enumerate(segments[:8], 1):
            from_label = _node_label(graph_store, segment.get("from_node"))
            to_label = _node_label(graph_store, segment.get("to_node"))
            text = (
                f"{index}. {segment.get('road_name', 'Unnamed corridor')} - "
                f"{from_label} to {to_label} ({segment.get('distance_km', 0)} km)"
            )
            lower_y = _paragraph(pdf, text, left, lower_y, map_w + 90, size=8, leading=10)
    else:
        lower_y = _paragraph(
            pdf,
            "No route segments are available. Review active restrictions, select another destination, or choose an operational vehicle profile.",
            left,
            lower_y,
            map_w + 90,
            size=8,
            leading=10,
        )

    if route.get("is_aerial_evacuation"):
        _paragraph(
            pdf,
            "Simulation notice: This aerial corridor is illustrative only. Aircraft availability, landing sites, and hospital notifications are not verified.",
            left,
            lower_y,
            map_w + 90,
            size=8,
            leading=10,
            color="#FBBF24",
        )

    pdf.setFillColor(colors.HexColor("#64748B"))
    pdf.setFont("Helvetica", 7)
    pdf.drawString(left, 24, "Aegis CrisisNav - demonstration dataset; not a live emergency dispatch system")
    pdf.drawRightString(right, 24, "Page 1")
    pdf.showPage()
    pdf.save()
    return buffer.getvalue()
