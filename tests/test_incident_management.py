"""Incident persistence, live routing, Qdrant, and form workflow tests."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from qdrant_client import QdrantClient
from streamlit.testing.v1 import AppTest

from src.graph_store import CrisisSpatialGraph
from src.incident_repository import IncidentRepository
from src.multimodal_retriever import MultimodalRetriever
from src.agent_workflow import CrisisResponseAgent

ROOT = Path(__file__).resolve().parents[1]


class IncidentManagementTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "incidents.json"
        self.graph = CrisisSpatialGraph(str(ROOT / "data/sample_osm_network.json"))
        self.roads = {data["id"]: {"name": data["name"], "nodes": (u, v)}
                      for u, v, data in self.graph.graph.edges(data=True)
                      if not data["id"].endswith("_REV")}
        self.repository = IncidentRepository(self.path, ROOT / "data/disaster_reports.json", self.roads)
        self.client = QdrantClient(":memory:")
        self.env = patch.dict("os.environ", {"GEMINI_API_KEY": "", "EMBEDDING_PROVIDER": "hash", "EMBEDDING_DIM": "128"})
        self.env.start()
        self.retriever = MultimodalRetriever(128, qdrant_client=self.client)

    def tearDown(self):
        self.client.close()
        self.env.stop()
        self.directory.cleanup()

    def new_report(self, **changes):
        report = {"report_id": "TEST-1", "incident_type": "Flood at Sindhi Society",
                  "text": "Water on the road", "source": "Operator", "severity": "HIGH",
                  "incident_status": "active", "affected_edges": ["E_CHEMBUR_TO_SINDHI"],
                  "flood_depth_cm": 80, "damage_severity": 0}
        report.update(changes)
        return report

    def test_create_update_persist_and_reject_stale_edit(self):
        rows, revision = self.repository.load()
        saved = self.repository.save(self.new_report(), revision)
        self.assertEqual(saved["affected_nodes"], ["N_CHEMBUR_NAKA", "N_INT_SINDHI_SOC"])
        rows, new_revision = self.repository.load()
        self.assertEqual(len(rows), 7)
        with self.assertRaisesRegex(ValueError, "changed while"):
            self.repository.save(self.new_report(), revision)
        self.repository.save(dict(saved, incident_status="resolved"), new_revision)
        reopened = IncidentRepository(self.path, self.repository.seed_path, self.roads)
        self.assertEqual(reopened.load()[0][-1]["incident_status"], "resolved")

    def test_moving_seed_report_clears_its_previous_road(self):
        rows, revision = self.repository.load()
        self.repository.save(dict(rows[0], affected_edges=["E_CHEMBUR_TO_SINDHI"]), revision)
        updated, _ = self.repository.load()
        self.graph.apply_incident_reports(updated)
        self.assertTrue(self.graph.graph["N_CHEMBUR_NAKA"]["N_INT_VN_PURAV"]["traversable"])
        self.assertFalse(self.graph.graph["N_CHEMBUR_NAKA"]["N_INT_SINDHI_SOC"]["traversable"])

    def test_validation_does_not_write_invalid_reports(self):
        _, revision = self.repository.load()
        for changes in ({"affected_edges": ["missing"]}, {"text": " "},
                        {"flood_depth_cm": -1}, {"damage_severity": float("nan")},
                        {"incident_status": "unknown"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.repository.save(self.new_report(**changes), revision)
        self.assertFalse(self.path.exists())

    def test_incident_closes_route_and_resolution_restores_it(self):
        rows, _ = self.repository.load()
        self.graph.apply_incident_reports(rows)
        original = self.graph.compute_route("N_CHEMBUR_NAKA", "N_HOSP_ZEN")
        incident = self.new_report()
        self.graph.apply_incident_reports(rows + [incident])
        detour = self.graph.compute_route("N_CHEMBUR_NAKA", "N_HOSP_ZEN")
        self.assertNotEqual(detour["path"], original["path"])
        self.assertNotEqual(detour["path"][1], "N_INT_SINDHI_SOC")
        self.graph.apply_incident_reports(rows + [dict(incident, incident_status="resolved")])
        self.assertEqual(self.graph.compute_route("N_CHEMBUR_NAKA", "N_HOSP_ZEN")["path"], original["path"])

    def test_resolving_seed_hazard_reopens_original_junction(self):
        rows, _ = self.repository.load()
        rows[0] = dict(rows[0], incident_status="resolved")
        self.graph.apply_incident_reports(rows)
        route = self.graph.compute_route("N_CHEMBUR_NAKA", "N_HOSP_ZEN")
        self.assertEqual(route["path"], ["N_CHEMBUR_NAKA", "N_INT_VN_PURAV", "N_HOSP_ZEN"])
        self.assertEqual(self.graph.graph["N_INT_VN_PURAV"]["N_CHEMBUR_NAKA"]["flood_depth_cm"], 0)

    def test_resolving_one_report_does_not_clear_another_active_hazard(self):
        first = self.new_report(incident_status="resolved")
        second = self.new_report(report_id="TEST-2", flood_depth_cm=40)
        self.graph.apply_incident_reports([first, second])
        edge = self.graph.graph["N_CHEMBUR_NAKA"]["N_INT_SINDHI_SOC"]
        self.assertEqual(edge["flood_depth_cm"], 40)
        self.assertEqual(self.graph.calculate_edge_weight("N_CHEMBUR_NAKA", "N_INT_SINDHI_SOC", edge), float("inf"))
        self.assertNotEqual(self.graph.calculate_edge_weight("N_CHEMBUR_NAKA", "N_INT_SINDHI_SOC", edge, "high_clearance_truck"), float("inf"))
        self.graph.apply_incident_reports([dict(second, road_closed=True)])
        self.assertFalse(self.graph.graph["N_CHEMBUR_NAKA"]["N_INT_SINDHI_SOC"]["traversable"])

    def test_qdrant_history_and_query_independent_hazard_application(self):
        rows, _ = self.repository.load()
        incident = self.new_report(text="zzz unrelated dispatch wording")
        self.retriever.index_reports(rows + [incident])
        agent = CrisisResponseAgent(self.graph, self.retriever)
        # Even an incident absent from the top search results must constrain routing.
        with patch.object(self.retriever, "retrieve_reports", return_value=[]):
            state = agent.run_pipeline("from Chembur Naka to Zen Hospital")
        self.assertNotEqual(state["primary_route"]["path"][1], "N_INT_SINDHI_SOC")
        self.retriever.index_reports(rows + [dict(incident, incident_status="resolved")])
        self.assertEqual(len(self.retriever.store.reports_snapshot()), 7)
        self.assertNotIn("TEST-1", [hit["report"]["report_id"] for hit in self.retriever.retrieve_reports("flood", 20)])
        with patch.object(self.retriever, "retrieve_reports", return_value=[]):
            restored = agent.run_pipeline("from Chembur Naka to Zen Hospital")
        self.assertEqual(restored["primary_route"]["path"][1], "N_INT_SINDHI_SOC")

    def test_truck_query_uses_supported_vehicle_profile(self):
        agent = CrisisResponseAgent(self.graph, self.retriever)
        state = agent.decompose_query_node({"user_query": "truck from Chembur Naka to Zen Hospital"})
        self.assertEqual(state["vehicle_type"], "high_clearance_truck")

    def test_full_dashboard_recalculates_after_create_and_resolve(self):
        with patch.dict("os.environ", {"QDRANT_URL": "", "QDRANT_PATH": str(self.path.parent / "qdrant"),
                                       "INCIDENT_REPORTS_PATH": str(self.path)}):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20).run()
            self.assertFalse(app.exception)
            original = app.session_state["current_state"]["primary_route"]["path"]
            app.text_input[0].set_value("Full dashboard flood test")
            app.multiselect[0].select("E_CHEMBUR_TO_SINDHI")
            app.text_area[0].set_value("Road inundated, use an alternate corridor")
            app.number_input[0].set_value(80)
            next(button for button in app.button if button.label == "Save incident").click().run()
            self.assertFalse(app.exception)
            detour = app.session_state["current_state"]["primary_route"]["path"]
            self.assertNotEqual(original, detour)
            report_id = self.repository.load()[0][-1]["report_id"]
            app.selectbox[0].select(report_id).run()
            app.selectbox[1].select("resolved")
            next(button for button in app.button if button.label == "Save incident").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["current_state"]["primary_route"]["path"], original)
            from src.qdrant_store import configured_client
            client = configured_client()
            self.assertEqual(client.count(MultimodalRetriever(128, qdrant_client=client).store.collections["reports"]).count, 7)
            client.close()

    def test_editor_create_and_resolve_flow(self):
        code = f"""
from src.incident_ui import render_incident_manager
from src.incident_repository import IncidentRepository
repository = IncidentRepository({str(self.path)!r}, {str(self.repository.seed_path)!r}, {self.roads!r})
render_incident_manager(repository)
"""
        app = AppTest.from_string(code).run()
        self.assertFalse(app.exception)
        app.text_input[0].set_value("Operator flood test")
        app.multiselect[0].select("E_CHEMBUR_TO_SINDHI")
        app.text_area[0].set_value("Flood observed on Sindhi Society approach")
        app.number_input[0].set_value(80)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.success)
        reports, _ = self.repository.load()
        report_id = reports[-1]["report_id"]
        self.assertEqual(reports[-1]["flood_depth_cm"], 80)
        app.selectbox[0].select(report_id).run()
        self.assertFalse(app.exception)
        app.selectbox[1].select("resolved")
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(self.repository.load()[0][-1]["incident_status"], "resolved")


if __name__ == "__main__":
    unittest.main()
