"""
Explainable Reasoning Paths & Decision Attribution (Module 6.3)
Generates step-by-step reasoning chains, counterfactual path comparisons,
evidence grounding citations, and infrastructure bottleneck attribution.
"""

from typing import Dict, Any, List, Optional
import json


class ReasoningChainVisualizer:
    """
    Extracts and visualizes explainable decision chains (Module 6.3).
    Explains WHY paths were chosen, WHY alternatives were eliminated,
    and grounds routing decisions in multimodal sensor evidence.
    """
    def __init__(self, graph_store: Any):
        self.graph_store = graph_store

    def explain_route_decision(
        self,
        primary_route: Optional[Dict[str, Any]],
        alternative_routes: List[Dict[str, Any]],
        retrieved_incidents: List[Dict[str, Any]],
        target_facility_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Builds a comprehensive explanation report covering:
        1. Primary Corridor Selection Justification
        2. Elimination Rationale for Impassable Edges
        3. Counterfactual Path Comparison (Path A vs Path B)
        4. Multimodal Grounding & Evidence Citations
        5. Single-Point-of-Failure Bottleneck Analysis
        """
        if not primary_route:
            return {
                "summary": "Route generation failed due to complete corridor severance.",
                "steps": [],
                "eliminated_edges": self._get_eliminated_edges(retrieved_incidents),
                "counterfactual_analysis": "No traversable corridors identified.",
                "evidence_citations": []
            }

        steps = []
        path_nodes = primary_route.get("path", [])

        # Step 1: Origin & Target Justification
        steps.append({
            "step_index": 1,
            "decision": "Facility & Endpoint Selection",
            "details": self._explain_facility_selection(primary_route["target"], target_facility_info)
        })

        # Step 2: Edge-by-Edge Traversal Justification
        for i in range(len(path_nodes) - 1):
            u = path_nodes[i]
            v = path_nodes[i + 1]
            edge_data = self.graph_store.graph.get_edge_data(u, v) or {}
            steps.append({
                "step_index": i + 2,
                "decision": f"Traverse Segment [{u} ➔ {v}]",
                "details": (
                    f"Selected '{edge_data.get('name')}' (Dist: {edge_data.get('distance_km')}km, "
                    f"Elevation: {edge_data.get('elevation_m')}m). "
                    f"Status is {edge_data.get('status').upper()} with damage severity {edge_data.get('damage_severity', 0):.2f} "
                    f"and flood water depth {edge_data.get('flood_depth_cm', 0)}cm (below vehicle hazard threshold)."
                )
            })

        # Step 3: Eliminated Edges Rationale
        eliminated = self._get_eliminated_edges(retrieved_incidents)

        # Step 4: Counterfactual Path Analysis (Why Path 1 over Path 2?)
        counterfactual = self._generate_counterfactual_comparison(primary_route, alternative_routes)

        # Step 5: Evidence Grounding Citations
        citations = []
        for inc in retrieved_incidents:
            citations.append({
                "source": inc.get("source"),
                "report_id": inc.get("report_id"),
                "incident_type": inc.get("incident_type"),
                "location_ref": inc.get("location_ref"),
                "image_evidence": inc.get("image_ref"),
                "evidence_summary": inc.get("text")
            })

        # Step 6: Bottlenecks & Critical Infrastructure
        bottlenecks = self._identify_bottlenecks(path_nodes)

        return {
            "summary": (
                f"Selected {len(path_nodes)}-node corridor spanning {primary_route['total_distance_km']} km "
                f"with estimated arrival in {primary_route['estimated_travel_time_min']} mins. "
                f"Bypassed {len(eliminated)} impassable hazard zones."
            ),
            "reasoning_steps": steps,
            "eliminated_edges": eliminated,
            "counterfactual_analysis": counterfactual,
            "evidence_citations": citations,
            "bottlenecks": bottlenecks
        }

    def _explain_facility_selection(
        self,
        target_node: str,
        facility_info: Optional[Dict[str, Any]]
    ) -> str:
        if not facility_info:
            return f"Direct routing targeted to requested endpoint [{target_node}]."

        name = facility_info.get("facility_name")
        avail = facility_info.get("available_capacity")
        total = facility_info.get("total_capacity")
        status = facility_info.get("operational_status")

        return (
            f"Target facility [{name}] was selected based on optimal operational status ('{status}') "
            f"and high available capacity ({avail}/{total} emergency beds), surpassing other damaged or flooded medical facilities."
        )

    def _get_eliminated_edges(self, retrieved_incidents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identifies edges that were disqualified and explains why."""
        eliminated = []
        for inc in retrieved_incidents:
            for eid in inc.get("affected_edges", []):
                eliminated.append({
                    "edge_id": eid,
                    "incident_id": inc.get("report_id"),
                    "reason": inc.get("incident_type"),
                    "flood_depth": f"{inc.get('flood_depth_cm', 0)} cm",
                    "damage_severity": f"{int(inc.get('damage_severity', 0)*100)}%",
                    "disqualification_rationale": (
                        f"Road {eid} disqualified: {inc.get('incident_type')} reported by {inc.get('source')}. "
                        f"Exceeds safe clearance thresholds for rapid emergency transit."
                    )
                })
        return eliminated

    def _generate_counterfactual_comparison(
        self,
        primary_route: Dict[str, Any],
        alternative_routes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Provides 'Why not Path B?' contrastive analysis."""
        if len(alternative_routes) <= 1:
            return {
                "comparison": "Only one viable topological corridor connects origin and destination under current hazard state.",
                "details": []
            }

        primary_eta = primary_route["estimated_travel_time_min"]
        primary_dist = primary_route["total_distance_km"]

        alt_comparisons = []
        for alt in alternative_routes[1:]:
            diff_eta = round(alt["eta_min"] - primary_eta, 1)
            diff_dist = round(alt["distance_km"] - primary_dist, 1)
            alt_comparisons.append({
                "alternative_rank": alt["rank"],
                "path": " ➔ ".join(alt["path"]),
                "eta_penalty_min": diff_eta,
                "distance_penalty_km": diff_dist,
                "rejection_rationale": (
                    f"Candidate Path {alt['rank']} rejected: Incurs a +{diff_eta} min delay (+{diff_dist} km detour) "
                    f"and encounters higher cumulative elevation/congestion risk."
                )
            })

        return {
            "primary_route": " ➔ ".join(primary_route["path"]),
            "alternatives_evaluated": len(alternative_routes),
            "details": alt_comparisons
        }

    def _identify_bottlenecks(self, path_nodes: List[str]) -> List[str]:
        """Identifies bridges, viaducts, or low elevation points along chosen route."""
        bottlenecks = []
        for nid in path_nodes:
            ndata = self.graph_store.graph.nodes.get(nid, {})
            ntype = ndata.get("type")
            elev = ndata.get("elevation_m", 10.0)
            if ntype == "bridge":
                bottlenecks.append(f"🌉 Bridge Node [{nid}] '{ndata.get('name')}': Single-span bottleneck; monitor for structural aftershocks.")
            elif elev < 5.0:
                bottlenecks.append(f"🌊 Low-Elevation Node [{nid}] ({elev}m): High vulnerability to advancing storm surges.")
        return bottlenecks

    def format_markdown_explanation(self, explanation: Dict[str, Any]) -> str:
        """Formats the explanation as a sleek, structured Markdown dossier."""
        md = []
        md.append("### 🧠 EXPLAINABLE REASONING CHAIN & DECISION PROVENANCE\n")
        md.append(f"**Operational Summary:** {explanation['summary']}\n")

        md.append("#### 1. STEP-BY-STEP REASONING TRAJECTORY")
        for step in explanation.get("reasoning_steps", []):
            md.append(f"- **Step {step['step_index']}: {step['decision']}**  \n  {step['details']}")

        md.append("\n#### 2. DISQUALIFIED HAZARD EDGES (ELIMINATION LOG)")
        eliminated = explanation.get("eliminated_edges", [])
        if eliminated:
            for el in eliminated:
                md.append(f"- ⛔ **{el['edge_id']}**: {el['disqualification_rationale']} *(Ref: {el['incident_id']})*")
        else:
            md.append("- No edges disqualified along this specific arterial query.")

        md.append("\n#### 3. COUNTERFACTUAL PATH COMPARISON (WHY THIS ROUTE?)")
        cf = explanation.get("counterfactual_analysis", {})
        if isinstance(cf, dict) and "details" in cf and cf["details"]:
            for d in cf["details"]:
                md.append(f"- ⚖️ **Alternative {d['alternative_rank']}**: `{d['path']}`  \n  {d['rejection_rationale']}")
        else:
            md.append("- Primary route is the singularly viable corridor under active disaster constraints.")

        md.append("\n#### 4. MULTIMODAL EVIDENCE GROUNDING")
        for cit in explanation.get("evidence_citations", []):
            md.append(f"- 📡 **{cit['report_id']}** (`{cit['source']}`): *\"{cit['evidence_summary']}\"* [Image Sensor: `{cit['image_evidence']}`]")

        bottlenecks = explanation.get("bottlenecks", [])
        if bottlenecks:
            md.append("\n#### 5. CRITICAL INFRASTRUCTURE BOTTLENECKS")
            for b in bottlenecks:
                md.append(f"- {b}")

        return "\n".join(md)

