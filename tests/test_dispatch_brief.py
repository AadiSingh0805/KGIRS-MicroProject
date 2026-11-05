import copy
import unittest
from pathlib import Path

from src.dispatch_brief import build_dispatch_brief
from src.graph_store import CrisisSpatialGraph


ROOT = Path(__file__).resolve().parents[1]


class DispatchBriefTests(unittest.TestCase):
    def setUp(self):
        self.graph = CrisisSpatialGraph(str(ROOT / "data/sample_osm_network.json"))

    def test_pdf_contains_route_details_and_schematic_for_normal_route(self):
        route = self.graph.compute_route("N_CHEMBUR_NAKA", "N_HOSP_ZEN", "ambulance")
        state = {
            "extracted_origin": "N_CHEMBUR_NAKA",
            "extracted_destination": "N_HOSP_ZEN",
            "vehicle_type": "ambulance",
            "avoid_hazards": True,
            "user_query": "Find the safest ambulance route",
            "primary_route": route,
            "retrieved_incidents": [{
                "report_id": "TEST-1",
                "incident_type": "Flood",
                "text": "Water depth reported on the approach",
            }],
        }
        pdf = build_dispatch_brief(state, self.graph, {"query": state["user_query"]})
        self.assertTrue(pdf.startswith(b"%PDF-"))
        self.assertGreater(len(pdf), 1000)

    def test_pdf_handles_missing_route_and_no_supporting_reports(self):
        state = {
            "extracted_origin": "N_CHEMBUR_NAKA",
            "extracted_destination": "N_HOSP_ZEN",
            "vehicle_type": "ambulance",
            "primary_route": None,
            "retrieved_incidents": [],
        }
        pdf = build_dispatch_brief(state, self.graph)
        self.assertTrue(pdf.startswith(b"%PDF-"))
        self.assertGreater(len(pdf), 1000)

    def test_disconnected_graph_produces_no_safe_ground_route(self):
        graph = copy.deepcopy(self.graph)
        blocked = graph.graph.nodes["N_CHEMBUR_NAKA"]
        blocked["operational_status"] = "closed"
        for neighbor in list(graph.graph.neighbors("N_CHEMBUR_NAKA")):
            graph.graph.edges["N_CHEMBUR_NAKA", neighbor]["traversable"] = False
        route = graph.compute_route("N_CHEMBUR_NAKA", "N_HOSP_ZEN", "ambulance")
        self.assertIsNone(route)


if __name__ == "__main__":
    unittest.main()
