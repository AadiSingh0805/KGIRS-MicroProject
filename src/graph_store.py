"""
Geospatial Infrastructure Graph Store & Spatial Traversal Engine
Handles NetworkX graph representation, dynamic road condition updates,
Dijkstra/A* routing with hazard penalties, k-shortest paths, and Pyvis export.
"""

import json
import copy
import math
import os
from typing import Dict, List, Optional, Tuple, Any
import networkx as nx

class CrisisSpatialGraph:
    """
    Manages geospatial road networks, medical centers, emergency shelters,
    and dynamic hazard impact states using NetworkX.
    """
    def __init__(self, network_json_path: Optional[str] = None):
        self.graph = nx.DiGraph()
        self.metadata: Dict[str, Any] = {}
        self.edge_lookup: Dict[str, Tuple[str, str]] = {}
        self._baseline_edges = {}
        self.node_lookup: Dict[str, Dict[str, Any]] = {}
        
        if network_json_path and os.path.exists(network_json_path):
            self.load_from_json(network_json_path)

    def load_from_json(self, json_path: str) -> None:
        """Loads nodes and edges from OSM-styled JSON dataset."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.metadata = data.get("metadata", {})
        self.graph.clear()
        self.edge_lookup.clear()
        self.node_lookup.clear()

        # Add Nodes
        for node in data.get("nodes", []):
            nid = node["id"]
            self.graph.add_node(
                nid,
                name=node.get("name", nid),
                type=node.get("type", "intersection"),
                lat=node.get("lat", 0.0),
                lon=node.get("lon", 0.0),
                elevation_m=node.get("elevation_m", 10.0),
                operational_status=node.get("operational_status", "active"),
                capacity=node.get("capacity", 0),
                available_capacity=node.get("available_capacity", 0),
                icu_beds=node.get("icu_beds", 0),
                helipad=node.get("helipad", False),
                generator_power=node.get("generator_power", True)
            )
            self.node_lookup[nid] = self.graph.nodes[nid]

        # Add Edges (Bi-directional representation for 2-way roads)
        for edge in data.get("edges", []):
            eid = edge["id"]
            u = edge["source"]
            v = edge["target"]
            dist_km = edge.get("distance_km", 1.0)
            speed = edge.get("base_speed_kmh", 40.0)
            base_time_min = (dist_km / max(speed, 5.0)) * 60.0

            edge_attrs = {
                "id": eid,
                "name": edge.get("name", eid),
                "road_type": edge.get("road_type", "secondary"),
                "lanes": edge.get("lanes", 2),
                "distance_km": dist_km,
                "base_speed_kmh": speed,
                "base_time_min": base_time_min,
                "elevation_m": edge.get("elevation_m", 10.0),
                "status": edge.get("status", "open"),
                "damage_severity": float(edge.get("damage_severity", 0.0)),
                "flood_depth_cm": float(edge.get("flood_depth_cm", 0.0)),
                "hazard_risk": float(edge.get("hazard_risk", 0.0)),
                "traversable": bool(edge.get("traversable", True)),
                "geometry": edge.get("geometry", [])
            }

            # Forward edge
            self.graph.add_edge(u, v, **edge_attrs)
            self.edge_lookup[eid] = (u, v)

            # Reverse edge for two-way streets
            reverse_attrs = dict(edge_attrs)
            reverse_attrs["id"] = f"{eid}_REV"
            if reverse_attrs["geometry"]:
                reverse_attrs["geometry"] = list(reversed(reverse_attrs["geometry"]))
            self.graph.add_edge(v, u, **reverse_attrs)
            self.edge_lookup[reverse_attrs["id"]] = (v, u)

        self._baseline_edges = {(u, v): copy.deepcopy(data) for u, v, data in self.graph.edges(data=True)}

    def apply_incident_reports(self, reports):
        """Apply every active incident; resolving one cannot clear another.

        Resolved reports clear their own baseline restrictions. Reports on other
        roads leave the original network intact, and repeated runs are identical.
        """
        for (u, v), data in self._baseline_edges.items():
            self.graph[u][v].clear()
            self.graph[u][v].update(copy.deepcopy(data))
        by_edge = {}
        for report in reports:
            for edge in report.get("cleared_edges", []):
                by_edge.setdefault(edge.removesuffix("_REV"), [])
            for edge in report.get("affected_edges", []):
                by_edge.setdefault(edge.removesuffix("_REV"), []).append(report)
        updates = []
        for edge_id, related in by_edge.items():
            active = [r for r in related if r.get("incident_status", "active") != "resolved"]
            flood = max((float(r.get("flood_depth_cm", 0)) for r in active), default=0)
            damage = max((float(r.get("damage_severity", 0)) for r in active), default=0)
            closed = any(r.get("road_closed", False) for r in active) or damage >= 0.85
            status = "severely_damaged" if damage >= 0.85 else ("blocked" if closed else ("flooded" if flood > 0 else ("caution" if damage > 0.25 else "open")))
            if self.update_edge_damage(edge_id, status, damage, flood, traversable=not closed):
                for key in (edge_id, f"{edge_id}_REV"):
                    if key in self.edge_lookup:
                        u, v = self.edge_lookup[key]
                        self.graph[u][v]["hazard_risk"] = max(damage, min(flood / 100, 1))
                updates.append(f"{edge_id}: {status}, flood={flood:g}cm, damage={damage:.0%}")
        return updates

    def node_hazards(self, node_id):
        edges = [data for _, _, data in self.graph.edges(node_id, data=True)]
        return {
            "flood_depth_cm": max((e.get("flood_depth_cm", 0) for e in edges), default=0),
            "damage_severity": max((e.get("damage_severity", 0) for e in edges), default=0),
        }

    def calculate_edge_weight(
        self,
        u: str,
        v: str,
        edge_data: Dict[str, Any],
        vehicle_type: str = "ambulance",
        avoid_hazards: bool = True
    ) -> float:
        """
        Calculates effective traversal cost (minutes + risk penalty).
        Returns infinity (math.inf) if impassable for the given vehicle type.
        """
        # Absolute blockers
        if not edge_data.get("traversable", True) and avoid_hazards:
            return float("inf")
        
        status = edge_data.get("status", "open")
        damage = edge_data.get("damage_severity", 0.0)
        flood = edge_data.get("flood_depth_cm", 0.0)
        risk = edge_data.get("hazard_risk", 0.0)
        base_time = edge_data.get("base_time_min", 2.0)

        # Vehicle threshold constraints
        max_flood_tolerance = {
            "ambulance": 20.0,         # 20cm max safe water clearance
            "high_clearance_truck": 60.0,
            "rescue_boat": 999.0,
            "civilian_car": 15.0
        }.get(vehicle_type, 20.0)

        if avoid_hazards:
            if flood > max_flood_tolerance:
                return float("inf")
            if damage >= 0.85: # Structural collapse / severed span
                return float("inf")

        # Dynamic penalty multiplier
        # Cost = base_time * (1 + 8*damage + 4*risk + flood_penalty)
        flood_penalty = (flood / 10.0) * 1.5 if flood > 0 else 0.0
        multiplier = 1.0 + (damage * 8.0) + (risk * 4.0) + flood_penalty

        return base_time * multiplier

    def update_edge_damage(
        self,
        edge_id: str,
        status: str,
        damage_severity: float,
        flood_depth_cm: float,
        traversable: Optional[bool] = None
    ) -> bool:
        """Dynamically applies incident dispatch telemetry to road edges."""
        updated = False
        target_ids = [edge_id, f"{edge_id}_REV"]
        
        for u, v, data in self.graph.edges(data=True):
            if data.get("id") in target_ids:
                data["status"] = status
                data["damage_severity"] = min(max(damage_severity, 0.0), 1.0)
                data["flood_depth_cm"] = max(flood_depth_cm, 0.0)
                if traversable is not None:
                    data["traversable"] = traversable
                else:
                    data["traversable"] = False if (status in ["severely_damaged", "flooded", "blocked"] or damage_severity > 0.8) else True
                updated = True
        return updated

    def compute_route(
        self,
        source: str,
        target: str,
        vehicle_type: str = "ambulance",
        avoid_hazards: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Computes the safest fastest path between source and target node
        using Dijkstra with dynamic hazard penalty cost weights.
        """
        if source not in self.graph or target not in self.graph:
            return None

        def weight_func(u, v, d):
            return self.calculate_edge_weight(u, v, d, vehicle_type, avoid_hazards)

        try:
            path_nodes = nx.dijkstra_path(self.graph, source, target, weight=weight_func)
            cost = nx.dijkstra_path_length(self.graph, source, target, weight=weight_func)
            if cost == float("inf") or math.isinf(cost):
                return None
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

        # Build detailed route segments
        total_distance_km = 0.0
        total_base_time_min = 0.0
        hazard_warnings = []
        segments = []

        for i in range(len(path_nodes) - 1):
            u_node = path_nodes[i]
            v_node = path_nodes[i + 1]
            edge_data = self.graph.get_edge_data(u_node, v_node)
            if not edge_data:
                return None
            
            # Strict safety check: if any segment is impassable, invalidate ground route
            edge_w = self.calculate_edge_weight(u_node, v_node, edge_data, vehicle_type, avoid_hazards)
            if edge_w == float("inf") or math.isinf(edge_w):
                return None
            
            dist = edge_data.get("distance_km", 0.0)
            base_t = edge_data.get("base_time_min", 0.0)
            status = edge_data.get("status", "open")
            dmg = edge_data.get("damage_severity", 0.0)
            fld = edge_data.get("flood_depth_cm", 0.0)
            name = edge_data.get("name", f"{u_node}->{v_node}")

            total_distance_km += dist
            total_base_time_min += base_t

            if dmg > 0.25 or fld > 5.0 or status != "open":
                hazard_warnings.append(
                    f"{name}: {status.upper()} (Damage: {int(dmg*100)}%, Flood: {fld:.0f}cm)"
                )

            segments.append({
                "from_node": u_node,
                "to_node": v_node,
                "edge_id": edge_data.get("id"),
                "road_name": name,
                "distance_km": dist,
                "status": status,
                "damage_severity": dmg,
                "flood_depth_cm": fld,
                "geometry": edge_data.get("geometry", [])
            })

        return {
            "source": source,
            "origin": source,
            "target": target,
            "destination": target,
            "path": path_nodes,
            "effective_cost_score": round(cost, 2),
            "total_distance_km": round(total_distance_km, 2),
            "estimated_travel_time_min": round(cost, 1),
            "nominal_travel_time_min": round(total_base_time_min, 1),
            "hazard_warnings": hazard_warnings,
            "segments": segments,
            "status": "CLEAR" if len(hazard_warnings) == 0 else "CAUTION"
        }

    def find_k_shortest_paths(
        self,
        source: str,
        target: str,
        k: int = 3,
        vehicle_type: str = "ambulance"
    ) -> List[Dict[str, Any]]:
        """Computes K alternate candidate paths for disaster contingency."""
        if source not in self.graph or target not in self.graph:
            return []

        def weight_func(u, v, d):
            return self.calculate_edge_weight(u, v, d, vehicle_type, avoid_hazards=True)

        results = []
        try:
            paths_gen = nx.shortest_simple_paths(self.graph, source, target, weight=weight_func)
            for idx, p in enumerate(paths_gen):
                if idx >= k:
                    break
                # Validate traversability
                is_valid = True
                cost = 0.0
                dist = 0.0
                for i in range(len(p) - 1):
                    ed = self.graph.get_edge_data(p[i], p[i+1])
                    w = weight_func(p[i], p[i+1], ed)
                    if math.isinf(w):
                        is_valid = False
                        break
                    cost += w
                    dist += ed.get("distance_km", 0.0)

                if is_valid:
                    results.append({
                        "rank": idx + 1,
                        "path": p,
                        "cost_score": round(cost, 2),
                        "distance_km": round(dist, 2),
                        "eta_min": round(cost, 1)
                    })
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
        return results

    def find_nearest_facility(
        self,
        source: str,
        facility_type: str = "hospital",
        min_capacity: int = 1,
        vehicle_type: str = "ambulance"
    ) -> Optional[Dict[str, Any]]:
        """
        Locates the optimal reachable facility of a given type based on
        path traversal time, status, and bed/shelter availability.
        """
        if source not in self.graph:
            return None

        candidates = []
        for nid, data in self.graph.nodes(data=True):
            if data.get("type") == facility_type:
                avail = data.get("available_capacity", 0)
                status = data.get("operational_status", "active")
                if avail >= min_capacity and status != "closed":
                    candidates.append((nid, data))

        best_choice = None
        min_eta = float("inf")

        for nid, data in candidates:
            route = self.compute_route(source, nid, vehicle_type=vehicle_type, avoid_hazards=True)
            if route and route["effective_cost_score"] < min_eta:
                min_eta = route["effective_cost_score"]
                best_choice = {
                    "facility_id": nid,
                    "facility_name": data.get("name"),
                    "facility_type": facility_type,
                    "available_capacity": data.get("available_capacity"),
                    "total_capacity": data.get("capacity"),
                    "operational_status": data.get("operational_status"),
                    "icu_beds": data.get("icu_beds", 0),
                    "helipad": data.get("helipad", False),
                    "route": route
                }

        return best_choice

    def export_pyvis_network(
        self,
        highlight_path: Optional[List[str]] = None,
        is_aerial: bool = False,
        height: str = "580px",
        width: str = "100%"
    ) -> str:
        """
        Generates Pyvis interactive HTML string with Dark Mode First /
        Tech Forward styling (obsidian backdrop, glowing nodes, neon corridors, and active Air Force vectors).
        """
        try:
            from pyvis.network import Network
        except ImportError:
            return "<div style='color:white;padding:20px;'>Pyvis is not installed</div>"

        net = Network(height=height, width=width, bgcolor="#0B0F19", font_color="#F8FAFC", directed=True)

        path_set = set(highlight_path) if highlight_path else set()
        active_origin = highlight_path[0] if highlight_path else None
        active_target = highlight_path[-1] if highlight_path else None
        path_edges = set()
        if highlight_path and len(highlight_path) > 1 and not is_aerial:
            for i in range(len(highlight_path) - 1):
                path_edges.add((highlight_path[i], highlight_path[i+1]))

        # Add Nodes with vibrant, tactical styling strictly matching Folium
        for nid, data in self.graph.nodes(data=True):
            ntype = data.get("type", "intersection")
            full_name = data.get("name", nid)
            words = full_name.split()
            short_label = " ".join(words[:2]) if len(words) > 2 else full_name
            status = data.get("operational_status", "active")
            is_origin = bool(active_origin and nid == active_origin)
            is_target = bool(active_target and nid == active_target)

            # Node role identification & palette
            if is_origin:
                color = "#00F0FF" # Cyan Extraction Origin Beacon
                border_color = "#FFFFFF"
                size = 32
                short_label = f"[ORIGIN] {short_label}"
            elif is_target:
                color = "#10B981" # Emerald Target Arrival Beacon
                border_color = "#FEF08A"
                size = 34
                short_label = f"[TARGET] {short_label}"
            elif self.node_hazards(nid)["flood_depth_cm"] > 20:
                color = "#0284C7" # Water Inundation Hazard
                border_color = "#67E8F9"
                size = 28
                short_label += " [FLOOD]"
            elif self.node_hazards(nid)["damage_severity"] >= 0.85:
                color = "#EF4444" # Critical Structural Shear Failure
                border_color = "#FCA5A5"
                size = 28
                short_label = "Suman Nagar [SEVERED]"
            elif ntype == "campus":
                color = "#A855F7" # Royal Purple (VESIT Campus)
                border_color = "#C084FC"
                size = 26
                short_label = "VESIT Campus"
            elif ntype == "hospital":
                color = "#059669" # Emerald Medical Triage Center
                border_color = "#34D399"
                size = 26
            elif ntype == "shelter":
                color = "#D97706" # Sunburst Amber Community Shelter
                border_color = "#FCD34D"
                size = 24
            elif ntype == "bridge":
                color = "#6366F1" # Cyber Indigo Elevated Span (Amar Mahal / Panjarpol)
                border_color = "#A5B4FC"
                size = 24
            elif ntype == "transit_hub" or nid == "N_CHEMBUR_NAKA":
                color = "#0284C7" # Ocean Transit Hub
                border_color = "#38BDF8"
                size = 26
            elif ntype == "supply_depot":
                color = "#2563EB" # Deep Blue Supply Depot
                border_color = "#60A5FA"
                size = 22
            else:
                # Open Road Intersections (Shell Colony, Tilak Nagar, Sindhi Society, Govandi)
                color = "#EA580C" # Vivid Tangerine
                border_color = "#FDBA74"
                size = 20
                if "shell" in nid.lower():
                    short_label = "Shell Colony [OPEN]"

            title_html = (
                f"<b>{full_name}</b> ({nid})<br/>"
                f"Classification: {ntype.upper()}<br/>"
                f"Status: {status.upper()}<br/>"
                f"Elevation: {data.get('elevation_m', 0)}m<br/>"
                f"Intake Capacity: {data.get('available_capacity', 0)}/{data.get('capacity', 0)} beds (ICU: {data.get('icu_beds', 0)})"
            )

            net.add_node(
                nid,
                label=short_label,
                title=title_html,
                color={"background": color, "border": border_color, "highlight": {"background": "#38BDF8", "border": "#FFFFFF"}},
                size=size,
                shape="dot",
                font={"color": "#F1F5F9", "size": 13, "face": "Inter, system-ui, sans-serif", "strokeWidth": 2, "strokeColor": "#080C14"}
            )

        # Add Edges with high-contrast tactical visibility
        for u, v, data in self.graph.edges(data=True):
            # Avoid duplicate reverse visualization clutter
            if data.get("id", "").endswith("_REV"):
                continue

            status = data.get("status", "open")
            damage = data.get("damage_severity", 0.0)
            flood = data.get("flood_depth_cm", 0.0)

            is_in_path = (u, v) in path_edges or (v, u) in path_edges

            if is_in_path:
                edge_color = "#00F0FF" # Neon Cyan active corridor
                width_val = 6
                dashes = False
            elif status in ["severely_damaged", "blocked"] or damage >= 0.7:
                edge_color = "#EF4444" # Red dashed blocked
                width_val = 3.5
                dashes = [6, 6]
            elif status == "flooded" or flood >= 25:
                edge_color = "#38BDF8" # Cyan dashed flood
                width_val = 4
                dashes = [6, 6]
            elif status == "caution":
                edge_color = "#F59E0B"
                width_val = 2.5
                dashes = False
            else:
                edge_color = "#475569" # High-contrast slate-blue road edge (crisply visible)
                width_val = 2.0
                dashes = False

            title_edge = (
                f"<b>{data.get('name')}</b><br/>"
                f"Status: {status.upper()}<br/>"
                f"Distance: {data.get('distance_km')} km<br/>"
                f"Damage: {int(damage*100)}% | Flood: {flood}cm"
            )

            net.add_edge(
                u,
                v,
                title=title_edge,
                color=edge_color,
                width=width_val,
                dashes=dashes,
                arrows="to"
            )

        # Direct Tactical Aerial Flight Vector if ground routing is severed
        if is_aerial and active_origin and active_target:
            net.add_edge(
                active_origin,
                active_target,
                title="<b>IAF Rotary Tactical Air Corridor</b><br/>Squadron: IAF 121 Helicopter Unit<br/>Asset: Mil Mi-17 V5 / HAL Dhruv ALH<br/>Status: ACTIVE ROTARY AIR-LIFT",
                color="#F59E0B",
                width=6,
                dashes=[10, 8],
                arrows="to",
                label="IAF ROTARY AIR VECTOR"
            )

        # Fine-tune ForceAtlas2 physics for stable, non-overlapping node distribution
        options = {
            "physics": {
                "forceAtlas2Based": {
                    "gravitationalConstant": -110,
                    "centralGravity": 0.015,
                    "springLength": 155,
                    "springStrength": 0.08,
                    "damping": 0.45,
                    "avoidOverlap": 0.55
                },
                "minVelocity": 0.75,
                "solver": "forceAtlas2Based",
                "stabilization": {"enabled": True, "iterations": 120}
            },
            "nodes": {
                "font": {
                    "size": 13,
                    "color": "#F8FAFC",
                    "face": "Inter, -apple-system, sans-serif"
                },
                "borderWidth": 2,
                "shadow": {"enabled": True, "color": "rgba(0,0,0,0.6)", "size": 8}
            },
            "edges": {
                "smooth": {"type": "continuous", "roundness": 0.15},
                "font": {"size": 10, "color": "#94A3B8", "strokeWidth": 2, "strokeColor": "#080C14"}
            },
            "interaction": {
                "hover": True,
                "tooltipDelay": 80,
                "navigationButtons": True,
                "keyboard": False
            }
        }
        net.set_options(json.dumps(options))

        raw_html = net.generate_html()
        # Clean Pyvis default white styling and border artifacts
        clean_html = raw_html.replace("border: 1px solid lightgray;", "border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px;")
        clean_html = clean_html.replace("background-color: white;", "background-color: #0B0F19;")
        clean_html = clean_html.replace("<div class=\"card-body\">", "<div style=\"background-color: #0B0F19; border-radius: 12px; overflow: hidden;\">")
        return clean_html

    def generate_neo4j_cypher(self) -> str:
        """Generates ready-to-run Neo4j Cypher statements for graph database import."""
        statements = [
            "// --- CRISIS INFRASTRUCTURE NEOSPACE SCHEMA ---",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE;"
        ]
        # Nodes
        for nid, data in self.graph.nodes(data=True):
            ntype = data.get("type", "Intersection").capitalize()
            statements.append(
                f"MERGE (n:{ntype}:GeoEntity {{id: '{nid}'}}) "
                f"SET n.name = '{data.get('name')}', n.lat = {data.get('lat')}, "
                f"n.lon = {data.get('lon')}, n.elevation = {data.get('elevation_m')}, "
                f"n.capacity = {data.get('capacity')}, n.available = {data.get('available_capacity')}, "
                f"n.status = '{data.get('operational_status')}';"
            )
        # Edges
        for u, v, data in self.graph.edges(data=True):
            if data.get("id", "").endswith("_REV"):
                continue
            statements.append(
                f"MATCH (u:GeoEntity {{id: '{u}'}}), (v:GeoEntity {{id: '{v}'}}) "
                f"MERGE (u)-[r:ROAD_CONNECTS {{id: '{data.get('id')}'}}]->(v) "
                f"SET r.name = '{data.get('name')}', r.distance_km = {data.get('distance_km')}, "
                f"r.status = '{data.get('status')}', r.damage = {data.get('damage_severity')}, "
                f"r.flood_depth_cm = {data.get('flood_depth_cm')};"
            )
        return "\n".join(statements)

