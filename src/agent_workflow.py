"""
Agentic Workflow & Graph Traversal Pipeline (Module 6.1)
Implements query decomposition, multimodal damage assessment, spatial graph traversal,
and tactical synthesis using an extensible StateGraph machine.
"""

import os
import json
import re
import difflib
import urllib.request
from typing import Dict, Any, List, Optional, TypedDict, Tuple
from .graph_store import CrisisSpatialGraph
from .multimodal_retriever import MultimodalRetriever

try:
    import google.generativeai as genai
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False


class CrisisAgentState(TypedDict, total=False):
    """LangGraph compatible State Schema for Crisis Agent Workflow."""
    user_query: str
    query_image_path: Optional[str]
    vehicle_type: str
    avoid_hazards: bool
    subtasks: List[str]
    extracted_origin: str
    extracted_destination: str
    facility_type_target: Optional[str]
    retrieved_incidents: List[Dict[str, Any]]
    retrieved_images: List[Dict[str, Any]]
    graph_updates: List[str]
    primary_route: Optional[Dict[str, Any]]
    alternative_routes: List[Dict[str, Any]]
    target_facility_info: Optional[Dict[str, Any]]
    reasoning_trace: List[Dict[str, Any]]
    final_dispatch_order: str
    confidence_score: float


class CrisisResponseAgent:
    """
    Coordinates Agentic Decomposition & Graph Traversal (Module 6.1).
    Performs multi-step reasoning: Decompose -> Retrieve -> Update -> Traverse -> Synthesize.
    """
    def __init__(
        self,
        graph_store: CrisisSpatialGraph,
        retriever: MultimodalRetriever,
        api_key: Optional[str] = None
    ):
        self.graph_store = graph_store
        self.retriever = retriever
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.candidate_models = ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-flash-latest", "gemini-3.8-flash"]
        self.model = None

        if self.api_key and _GENAI_AVAILABLE:
            try:
                genai.configure(api_key=self.api_key)
            except Exception:
                pass

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Calls Gemini API via direct REST or SDK with graceful fallback."""
        if not self.api_key:
            return None

        # 1. Direct REST via standard library (robust across environments)
        for m_name in self.candidate_models:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{m_name}:generateContent?key={self.api_key}"
                payload = {"contents": [{"parts": [{"text": prompt}]}]}
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode("utf-8"))
                        cands = data.get("candidates", [])
                        if cands:
                            parts = cands[0].get("content", {}).get("parts", [])
                            if parts and parts[0].get("text"):
                                return parts[0]["text"].strip()
            except Exception:
                continue

        # 2. Try google.generativeai SDK if available
        if _GENAI_AVAILABLE:
            for m_name in self.candidate_models:
                try:
                    m = genai.GenerativeModel(m_name)
                    response = m.generate_content(prompt)
                    if response and response.text:
                        self.model = m
                        return response.text.strip()
                except Exception:
                    continue

        return None

    def _resolve_landmark(self, text: str) -> Optional[str]:
        """
        Dynamically resolves natural language landmark strings to canonical graph node IDs
        using alias matching, token overlap, and Levenshtein similarity.
        Handles variations, typos (e.g. 'Sunita' -> 'Surana'), and real Chembur landmarks.
        """
        if not text:
            return None
        
        t_clean = re.sub(r"[^\w\s]", " ", text.lower()).strip()
        if not t_clean:
            return None

        # 0. High-Priority Direct Canonical Fast-Paths
        if "vesit" in t_clean or "vivekanand" in t_clean or "collector" in t_clean or "hashu advani" in t_clean:
            return "N_CAMPUS_VESIT"
        if "zen" in t_clean:
            return "N_HOSP_ZEN"
        if "surana" in t_clean or "sunita" in t_clean or "sethia" in t_clean:
            return "N_HOSP_SURANA"
        if "inlaks" in t_clean or "postal colony hosp" in t_clean:
            return "N_HOSP_INLAKS"
        if "rcf" in t_clean:
            return "N_SHELTER_RCF"
        if "acres" in t_clean:
            return "N_SHELTER_ACRES"
        if "amar mahal" in t_clean:
            return "N_BRIDGE_AMAR_MAHAL"
        if "freeway" in t_clean or "panjarpol" in t_clean:
            return "N_BRIDGE_FREEWAY"
        if "shell colony" in t_clean or t_clean == "shell":
            return "N_INT_SHELL_COLONY"
        if "tilak nagar" in t_clean or t_clean == "tilak":
            return "N_INT_TILAK_NAGAR"
        if "suman nagar" in t_clean or t_clean == "suman":
            return "N_INT_SUMAN_NAGAR"
        if "sindhi society" in t_clean or t_clean == "sindhi":
            return "N_INT_SINDHI_SOC"
        if "vn purav" in t_clean or "purav" in t_clean:
            return "N_INT_VN_PURAV"
        if "govandi" in t_clean:
            return "N_INT_GOVANDI"
        if "mankhurd" in t_clean:
            return "N_DEPOT_MANKHURD"
        if "naka" in t_clean or "chembur naka" in t_clean or t_clean == "chembur":
            return "N_CHEMBUR_NAKA"

        # 1. Alias & Phonetic Mapping for Chembur Landmarks
        aliases = {
            "N_CAMPUS_VESIT": [
                "vesit", "vivekanand education society", "vivekanand", "ves", 
                "collector colony campus", "collectors colony", "hashu advani",
                "engineering college", "ves campus", "vivekanand institute", "vesit campus"
            ],
            "N_HOSP_SURANA": [
                "surana", "surana sethia", "surana hospital", "sunita hospital", "sunita",
                "suman nagar hospital", "surana trauma centre", "surana sethia hospital"
            ],
            "N_HOSP_ZEN": [
                "zen", "zen hospital", "zen multi speciality", "zen speciality", "zen chembur", "zen multi"
            ],
            "N_HOSP_INLAKS": [
                "inlaks", "inlaks hospital", "inlaks general", "postal colony hospital"
            ],
            "N_CHEMBUR_NAKA": [
                "chembur naka", "chembur circle", "naka", "chembur junction", "chembur", "downtown", "naka hub"
            ],
            "N_BRIDGE_AMAR_MAHAL": [
                "amar mahal", "amar mahal flyover", "amar mahal junction", "amar flyover", "amar bridge"
            ],
            "N_BRIDGE_FREEWAY": [
                "eastern freeway", "panjarpol", "freeway ramp", "panjarpol elevated", "elevated span"
            ],
            "N_SHELTER_ACRES": [
                "acres", "the acres club", "acres club", "sindhi society gymkhana", "civic center", "acres mega shelter"
            ],
            "N_SHELTER_RCF": [
                "rcf", "rcf sports club", "rcf shelter", "rcf ground", "rcf community shelter", "rcf sports complex"
            ],
            "N_INT_SUMAN_NAGAR": [
                "suman nagar", "suman nagar junction", "suman nagar flyover", "suman"
            ],
            "N_INT_TILAK_NAGAR": [
                "tilak nagar", "tilak nagar station", "tilak nagar cross", "tilak"
            ],
            "N_INT_SINDHI_SOC": [
                "sindhi society", "sindhi society central", "sindhi society cross", "sindhi cross"
            ],
            "N_INT_SHELL_COLONY": [
                "shell colony", "shell colony ground", "shell colony postal cross", "shell"
            ],
            "N_INT_VN_PURAV": [
                "vn purav", "v n purav", "vn purav marg", "purav marg", "vn purav corridor"
            ],
            "N_INT_GOVANDI": [
                "govandi", "govandi station", "govandi west", "deonar depot link"
            ],
            "N_DEPOT_MANKHURD": [
                "mankhurd", "mankhurd depot", "mankhurd logistics", "supply depot", "mankhurd supply"
            ]
        }

        # Check exact alias match
        for node_id, alias_list in aliases.items():
            for al in alias_list:
                if al == t_clean or (len(al) > 3 and al in t_clean):
                    return node_id

        # 2. Check Node Name substrings in graph
        for node_id, node_data in self.graph_store.graph.nodes(data=True):
            n_name = node_data.get("name", "").lower()
            if t_clean in n_name or (len(t_clean) > 3 and n_name in t_clean):
                return node_id

        # 3. Fuzzy matching using SequenceMatcher
        best_score = 0.0
        best_node = None

        for node_id, alias_list in aliases.items():
            for al in alias_list:
                ratio = difflib.SequenceMatcher(None, t_clean, al).ratio()
                if ratio > best_score:
                    best_score = ratio
                    best_node = node_id

        for node_id, node_data in self.graph_store.graph.nodes(data=True):
            n_name = node_data.get("name", "").lower()
            ratio = difflib.SequenceMatcher(None, t_clean, n_name).ratio()
            if ratio > best_score:
                best_score = ratio
                best_node = node_id

        if best_score >= 0.55:
            return best_node

        return None

    # --- Node 1: Decompose Query ---
    def decompose_query_node(self, state: CrisisAgentState) -> CrisisAgentState:
        """
        Deconstructs complex user dispatch request into atomic subtasks
        and extracts spatial entities (origin, destination or facility type)
        dynamically using linguistic clause parsing and landmark resolution.
        """
        query = state.get("user_query", "").strip()
        trace = state.setdefault("reasoning_trace", [])

        # Subtask decomposition
        subtasks = [
            "Decompose crisis directive into origin, destination, and vehicle constraints",
            "Retrieve multimodal flood/damage satellite telemetry and update graph edge traversability",
            "Execute spatial graph traversal with hazard-penalized Dijkstra / A*",
            "Compute K-shortest alternate contingency corridors",
            "Synthesize tactical field order and explain decision rationale"
        ]
        state["subtasks"] = subtasks

        q_lower = query.lower()

        extracted_origin = None
        extracted_dest = None
        facility_type = None

        # Natural language regex patterns for origin and destination clauses
        patterns = [
            r"from\s+(.+?)\s+to\s+(.+?)(?:\s+(?:avoiding|via|with|for|\.|$)|$)",
            r"(?:route|corridor|path)\s+(?:from|between)\s+(.+?)\s+(?:to|and)\s+(.+?)(?:\s+(?:avoiding|via|with|for|\.|$)|$)",
            r"(?:to|towards)\s+(.+?)\s+(?:from|starting at)\s+(.+?)(?:\s+(?:avoiding|via|with|for|\.|$)|$)"
        ]

        for p_idx, pat in enumerate(patterns):
            m = re.search(pat, q_lower)
            if m:
                if p_idx == 2:
                    cand_dest_raw = m.group(1).strip()
                    cand_orig_raw = m.group(2).strip()
                else:
                    cand_orig_raw = m.group(1).strip()
                    cand_dest_raw = m.group(2).strip()

                extracted_origin = self._resolve_landmark(cand_orig_raw)

                # Check if destination is generic vs explicit location
                if "nearest hospital" in cand_dest_raw or "closest hospital" in cand_dest_raw or cand_dest_raw in ["hospital", "nearest"]:
                    facility_type = "hospital"
                    extracted_dest = None
                elif "nearest shelter" in cand_dest_raw or "closest shelter" in cand_dest_raw or cand_dest_raw in ["shelter", "relief shelter"]:
                    facility_type = "shelter"
                    extracted_dest = None
                else:
                    extracted_dest = self._resolve_landmark(cand_dest_raw)
                break

        # Fallback clause scanning across key landmark tokens if pattern didn't capture both
        if not extracted_origin:
            for word in ["vesit", "vivekanand", "sunita", "surana", "chembur naka", "chembur", "amar mahal", "acres", "rcf", "suman", "tilak", "inlaks", "govandi", "mankhurd"]:
                if word in q_lower:
                    extracted_origin = self._resolve_landmark(word)
                    break

        if not extracted_dest and not facility_type:
            if "nearest hospital" in q_lower or "closest hospital" in q_lower:
                facility_type = "hospital"
            elif "nearest shelter" in q_lower or "closest shelter" in q_lower:
                facility_type = "shelter"
            else:
                for word in ["zen", "surana", "sunita", "inlaks", "rcf", "acres", "vesit", "amar mahal", "mankhurd"]:
                    if word in q_lower:
                        cand = self._resolve_landmark(word)
                        if cand != extracted_origin:
                            extracted_dest = cand
                            break

        origin = extracted_origin or "N_CHEMBUR_NAKA"
        destination = extracted_dest or ("N_HOSP_ZEN" if not facility_type else None)

        # Dynamic vehicle type extraction from text
        if any(w in q_lower for w in ["boat", "zodiac", "raft"]):
            state["vehicle_type"] = "rescue_boat"
        elif any(w in q_lower for w in ["truck", "heavy vehicle", "unimog"]):
            state["vehicle_type"] = "heavy_truck"
        elif any(w in q_lower for w in ["car", "sedan", "civilian"]):
            state["vehicle_type"] = "civilian_car"
        elif "ambulance" in q_lower:
            state["vehicle_type"] = "ambulance"

        # By default in crisis navigation, all emergency dispatch avoids impassable hazards
        if any(w in q_lower for w in ["ignore hazard", "unconstrained", "force direct", "allow flood"]):
            state["avoid_hazards"] = False
        else:
            state["avoid_hazards"] = True

        state["extracted_origin"] = origin
        state["extracted_destination"] = destination
        state["facility_type_target"] = facility_type

        origin_name = self.graph_store.graph.nodes.get(origin, {}).get("name", origin)
        dest_name = self.graph_store.graph.nodes.get(destination, {}).get("name", destination) if destination else f"Nearest {facility_type.title()}"

        trace.append({
            "step": "Query Decomposition",
            "module": "Agentic Decomposition",
            "details": f"Decomposed query '{query}'. Identified Origin: [{origin_name}], Destination: [{dest_name}], Facility Intent: [{facility_type or 'Explicit Corridor'}]."
        })
        return state

    # --- Node 2: Assess Hazards & Dynamic Graph Update ---
    def assess_hazards_node(self, state: CrisisAgentState) -> CrisisAgentState:
        """
        Retrieves multimodal incident reports and reconnaissance images,
        and synchronizes dynamic road damages onto the graph store.
        """
        query = state.get("user_query", "")
        img_path = state.get("query_image_path")
        trace = state.get("reasoning_trace", [])

        # Retrieve incident reports
        matched_reports = self.retriever.retrieve_reports(query, top_k=3)
        state["retrieved_incidents"] = [m["report"] for m in matched_reports]

        # Retrieve images if applicable
        matched_imgs = self.retriever.retrieve_images(query_text=query, query_image_path=img_path, top_k=2)
        state["retrieved_images"] = matched_imgs

        graph_updates = []
        for match in matched_reports:
            rep = match["report"]
            for eid in rep.get("affected_edges", []):
                status = "severely_damaged" if rep.get("damage_severity", 0) > 0.8 else ("flooded" if rep.get("flood_depth_cm", 0) > 30 else "caution")
                updated = self.graph_store.update_edge_damage(
                    edge_id=eid,
                    status=status,
                    damage_severity=rep.get("damage_severity", 0.5),
                    flood_depth_cm=rep.get("flood_depth_cm", 0.0)
                )
                if updated:
                    graph_updates.append(f"Updated Edge {eid}: Status={status}, Dmg={rep.get('damage_severity')}, Flood={rep.get('flood_depth_cm')}cm")

        state["graph_updates"] = graph_updates

        trace.append({
            "step": "Multimodal Hazard Assessment",
            "module": "Dynamic Graph Synchronization",
            "details": f"Retrieved {len(matched_reports)} incident dispatches and {len(matched_imgs)} aerial imagery matches. Dynamically updated {len(graph_updates)} graph edges with real-time hazard constraints."
        })
        return state

    # --- Node 3: Spatial Graph Traversal ---
    def spatial_traverse_node(self, state: CrisisAgentState) -> CrisisAgentState:
        """
        Executes Dijkstra/A* routing over the hazard-penalized graph.
        If user requested nearest facility, identifies reachable facility with optimal capacity.
        """
        origin = state.get("extracted_origin", "N_CHEMBUR_NAKA")
        destination = state.get("extracted_destination", "N_HOSP_ZEN")
        facility_type = state.get("facility_type_target")
        vehicle_type = state.get("vehicle_type", "ambulance")
        avoid_hazards = state.get("avoid_hazards", True)
        trace = state.get("reasoning_trace", [])

        primary_route = None
        facility_info = None

        if facility_type and not destination:
            # Multi-facility capacity & reachability search ONLY when destination is unspecified
            facility_choice = self.graph_store.find_nearest_facility(
                source=origin,
                facility_type=facility_type,
                min_capacity=2,
                vehicle_type=vehicle_type
            )
            if facility_choice:
                destination = facility_choice["facility_id"]
                state["extracted_destination"] = destination
                primary_route = facility_choice["route"]
                facility_info = facility_choice
        elif destination:
            primary_route = self.graph_store.compute_route(
                source=origin,
                target=destination,
                vehicle_type=vehicle_type,
                avoid_hazards=avoid_hazards
            )
            if destination in self.graph_store.graph.nodes:
                d_node = self.graph_store.graph.nodes[destination]
                facility_info = {
                    "facility_id": destination,
                    "facility_name": d_node.get("name", destination),
                    "type": d_node.get("type", "facility"),
                    "available_capacity": d_node.get("available_capacity", 0),
                    "total_capacity": d_node.get("capacity", 0),
                    "icu_beds": d_node.get("icu_beds", 0),
                    "operational_status": d_node.get("operational_status", "active")
                }

        if not primary_route and destination:
            primary_route = self.graph_store.compute_route(
                source=origin,
                target=destination,
                vehicle_type=vehicle_type,
                avoid_hazards=avoid_hazards
            )

        # Candidate contingency paths (K-shortest)
        alt_paths = self.graph_store.find_k_shortest_paths(
            source=origin,
            target=destination,
            k=3,
            vehicle_type=vehicle_type
        )

        state["primary_route"] = primary_route
        state["alternative_routes"] = alt_paths
        state["target_facility_info"] = facility_info

        route_status = primary_route.get("status") if primary_route else "NO_ROUTE_FOUND"
        trace.append({
            "step": "Spatial Graph Traversal",
            "module": "Resilient Traversal Engine",
            "details": f"Computed primary path [{origin} -> {destination}] with status [{route_status}]. Evaluated {len(alt_paths)} viable contingency corridors."
        })
        return state

    # --- Node 4: Tactical Dispatch Synthesis ---
    def synthesize_dispatch_node(self, state: CrisisAgentState) -> CrisisAgentState:
        """
        Synthesizes operational instructions, turn-by-turn guidance, and safety warnings.
        """
        route = state.get("primary_route")
        origin = state.get("extracted_origin")
        dest = state.get("extracted_destination")
        facility_info = state.get("target_facility_info")
        trace = state.get("reasoning_trace", [])

        if not route:
            synthesis = (
                f"CRITICAL DISPATCH HALT: No safe traversable path exists from [{origin}] to [{dest}] "
                f"under active flood surge and structural damage constraints. Recommend aerial evacuation via helipad."
            )
            state["final_dispatch_order"] = synthesis
            state["confidence_score"] = 0.20
            return state

        path_nodes = " -> ".join(route["path"])
        dist = route["total_distance_km"]
        eta = route["estimated_travel_time_min"]
        warnings = "\n".join([f"  • [CAUTION] {w}" for w in route.get("hazard_warnings", [])]) or "  • All corridor segments nominal green."

        facility_context = ""
        if facility_info:
            facility_context = (
                f"\nTARGET FACILITY METRICS:\n"
                f"  • Name: {facility_info.get('facility_name')} ({facility_info.get('facility_id')})\n"
                f"  • Available Capacity: {facility_info.get('available_capacity')}/{facility_info.get('total_capacity')} beds\n"
                f"  • ICU Beds Open: {facility_info.get('icu_beds', 'N/A')} | Helipad: {'Active' if facility_info.get('helipad') else 'None'}"
            )

        # Prepare prompt for LLM or formulate structured dispatch
        llm_prompt = f"""
You are the Tactical Dispatch AI in a Crisis Operations Center.
Synthesize an authoritative, concise emergency dispatch directive based on these verified GraphRAG routing metrics:

Origin: {origin}
Destination: {dest}
Selected Safe Corridor: {path_nodes}
Total Distance: {dist} km
Estimated Response Time: {eta} mins
Active Hazard Warnings:
{warnings}
Facility Telemetry:
{facility_context}

Format with:
1. MISSION DIRECTIVE SUMMARY
2. CLEAR CORRIDOR ROUTING INSTRUCTIONS
3. HAZARD ADVISORIES & BYPASS RATIONALE
4. CONTINGENCY FALLBACK
Keep it high-contrast, tactical, and strictly grounded in the verified data.
"""
        llm_response = self._call_llm(llm_prompt)

        # Resolve human-readable node names
        origin_node_data = self.graph_store.graph.nodes.get(origin, {})
        dest_node_data = self.graph_store.graph.nodes.get(dest, {})
        origin_name = origin_node_data.get("name", origin)
        dest_name = dest_node_data.get("name", dest)
        path_names = [self.graph_store.graph.nodes.get(n, {}).get("name", n) for n in route.get("path", [])]
        path_display = " -> ".join(path_names)

        if not llm_response:
            # Deterministic Tactical Dispatch Fallback - Clean, Human-Readable
            warnings_text = "\n".join([f"• {w}" for w in route.get("hazard_warnings", [])]) if route.get("hazard_warnings") else "• All segments verified clear and traversable above flood waterlines."
            
            facility_line = ""
            if facility_info:
                facility_line = f"• Destination Capacity: {facility_info.get('available_capacity')}/{facility_info.get('total_capacity')} emergency beds open (ICU beds: {facility_info.get('icu_beds', 'N/A')})."
            
            llm_response = (
                f"MISSION DIRECTIVE: Resilient Route Active\n"
                f"From: {origin_name} to {dest_name}\n"
                f"Status: {route.get('status', 'CLEAR')} | Estimated Transit: {eta} min | Distance: {dist} km\n\n"
                f"PRIMARY CORRIDOR:\n"
                f"{path_display}\n\n"
                f"OPERATIONAL INSTRUCTIONS:\n"
                f"• Proceed along designated arterial corridor; adhere to emergency convoy speed.\n"
                f"{warnings_text}\n"
                f"{facility_line}\n"
                f"• Contingency fallback corridors pre-computed in case of secondary waterlogging."
            )

        state["final_dispatch_order"] = llm_response
        state["confidence_score"] = 0.94 if route.get("status") == "CLEAR" else 0.78

        trace.append({
            "step": "Tactical Dispatch Synthesis",
            "module": "Mission Order Generation",
            "details": f"Generated final dispatch order with confidence {state['confidence_score']:.2f}. ETA: {eta} min over {dist} km."
        })
        return state

    def run_pipeline(
        self,
        query: str,
        image_path: Optional[str] = None,
        vehicle_type: str = "ambulance",
        avoid_hazards: bool = True
    ) -> CrisisAgentState:
        """Executes the full agentic GraphRAG pipeline end-to-end."""
        state: CrisisAgentState = {
            "user_query": query,
            "query_image_path": image_path,
            "vehicle_type": vehicle_type,
            "avoid_hazards": avoid_hazards,
            "reasoning_trace": []
        }

        # Step 1: Decompose
        state = self.decompose_query_node(state)
        # Step 2: Assess Hazards
        state = self.assess_hazards_node(state)
        # Step 3: Spatial Traverse
        state = self.spatial_traverse_node(state)
        # Step 4: Synthesize Dispatch
        state = self.synthesize_dispatch_node(state)

        return state

