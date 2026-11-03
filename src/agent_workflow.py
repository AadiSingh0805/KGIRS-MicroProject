"""
Agentic Workflow & Graph Traversal Pipeline (Module 6.1)
Implements query decomposition, multimodal damage assessment, spatial graph traversal,
and tactical synthesis using an extensible StateGraph machine.
"""

import os
import json
from typing import Dict, Any, List, Optional, TypedDict
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
        self.model = None

        if self.api_key and _GENAI_AVAILABLE:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel("gemini-2.5-flash")
            except Exception:
                try:
                    self.model = genai.GenerativeModel("gemini-1.5-flash")
                except Exception:
                    self.model = None

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Calls Gemini API with graceful error handling."""
        if self.model:
            try:
                response = self.model.generate_content(prompt)
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                print(f"LLM API Call notice: {e}. Switching to deterministic engine.")
        return None

    # --- Node 1: Decompose Query ---
    def decompose_query_node(self, state: CrisisAgentState) -> CrisisAgentState:
        """
        Deconstructs complex user dispatch request into atomic subtasks
        and extracts spatial entities (origin, destination or facility type).
        """
        query = state.get("user_query", "").strip()
        trace = state.setdefault("reasoning_trace", [])

        # Subtask decomposition
        subtasks = [
            "Decompose crisis directive into origin, destination, and vehicle constraints",
            "Retrieve multimodal flood/damage telemetry and update graph edge traversability",
            "Execute spatial graph traversal with hazard-penalized Dijkstra / A*",
            "Compute K-shortest alternate contingency corridors",
            "Synthesize tactical field order and explain decision rationale"
        ]
        state["subtasks"] = subtasks

        # Extract entities using LLM or rule-based parser
        origin = "N_CHEMBUR_NAKA"
        destination = "N_HOSP_ZEN"
        facility_type = None

        q_lower = query.lower()

        # Origin heuristic
        if "chembur" in q_lower or "naka" in q_lower or "downtown" in q_lower:
            origin = "N_CHEMBUR_NAKA"
        elif "acres" in q_lower:
            origin = "N_SHELTER_ACRES"
        elif "rcf" in q_lower:
            origin = "N_SHELTER_RCF"
        elif "suman" in q_lower or "surana" in q_lower:
            origin = "N_INT_SUMAN_NAGAR"
        elif "tilak" in q_lower:
            origin = "N_INT_TILAK_NAGAR"
        elif "civic" in q_lower:
            origin = "N_SHELTER_ACRES"

        # Destination / Facility heuristic
        if "surana" in q_lower:
            destination = "N_HOSP_SURANA"
            facility_type = "hospital"
        elif "zen" in q_lower:
            destination = "N_HOSP_ZEN"
            facility_type = "hospital"
        elif "inlaks" in q_lower:
            destination = "N_HOSP_INLAKS"
            facility_type = "hospital"
        elif "nearest hospital" in q_lower or "hospital" in q_lower:
            facility_type = "hospital"
            destination = "N_HOSP_ZEN"
        elif "acres" in q_lower or "nearest shelter" in q_lower:
            facility_type = "shelter"
            destination = "N_SHELTER_ACRES"
        elif "rcf" in q_lower or "shelter" in q_lower:
            facility_type = "shelter"
            destination = "N_SHELTER_RCF"
        elif "depot" in q_lower or "mankhurd" in q_lower:
            destination = "N_DEPOT_MANKHURD"
        elif "amar mahal" in q_lower or "bridge" in q_lower or "bay span" in q_lower:
            destination = "N_BRIDGE_AMAR_MAHAL"

        state["extracted_origin"] = origin
        state["extracted_destination"] = destination
        state["facility_type_target"] = facility_type

        trace.append({
            "step": "Query Decomposition",
            "module": "Agentic Decomposition",
            "details": f"Decomposed query '{query}'. Identified Origin: [{origin}], Destination: [{destination}], Facility Intent: [{facility_type}]."
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

        if facility_type:
            # Multi-facility capacity & reachability search
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

        if not primary_route:
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

        path_nodes = " ➔ ".join(route["path"])
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
        path_display = " ➔ ".join(path_names)

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

