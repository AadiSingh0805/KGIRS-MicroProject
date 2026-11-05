"""Provider failures and mixed vector spaces must never be silent."""
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from qdrant_client import QdrantClient, models

from src.embeddings import EmbeddingError, TextEmbedder
from src.multimodal_retriever import MultimodalRetriever
from src.graph_store import CrisisSpatialGraph
from src.agent_workflow import CrisisResponseAgent
from src.vehicle_profiles import VEHICLE_PROFILES, normalize_vehicle_type

ROOT = Path(__file__).resolve().parents[1]


def response(values):
    return io.BytesIO(json.dumps({"embedding": {"values": values}}).encode())


class EmbeddingConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict("os.environ", {"EMBEDDING_PROVIDER": "hash", "EMBEDDING_DIM": "", "GEMINI_API_KEY": ""})
        self.env.start()
        self.client = QdrantClient(":memory:")
        self.reports = [{"report_id": "one", "incident_type": "Flood", "text": "Flood on road"}]

    def tearDown(self):
        self.client.close()
        self.env.stop()

    def retriever(self, provider="hash", dimension=128):
        return MultimodalRetriever(dimension, api_key="test-key", qdrant_client=self.client, embedding_provider=provider)

    def test_key_alone_does_not_change_provider_or_collection(self):
        first = self.retriever()
        with patch.dict("os.environ", {"GEMINI_API_KEY": "another-key"}):
            second = MultimodalRetriever(128, qdrant_client=self.client)
        self.assertEqual(second.embedder.provider, "hash")
        self.assertEqual(first.store.collections, second.store.collections)
        self.assertTrue(np.array_equal(first.compute_text_embedding("flood"), second.compute_text_embedding("flood")))

    def test_configuration_requires_valid_provider_dimension_and_key(self):
        for arguments in ((0, "hash"), (1.5, "hash"), (128, "unknown"), (4096, "gemini")):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                TextEmbedder(*arguments, api_key="test-key")
        with self.assertRaisesRegex(ValueError, "GEMINI_API_KEY"):
            TextEmbedder(provider="gemini")
        with patch.dict("os.environ", {"EMBEDDING_DIM": "invalid"}), self.assertRaises(ValueError):
            TextEmbedder()
        with patch.dict("os.environ", {"EMBEDDING_DIM": ""}):
            self.assertEqual(TextEmbedder().dimension, 128)
            self.assertEqual(TextEmbedder(provider="gemini", api_key="test-key").dimension, 768)

    def test_gemini_requests_matching_dimension_and_correct_tasks(self):
        retriever = self.retriever("gemini", 3)
        with patch("src.embeddings.urllib.request.urlopen", side_effect=[response([3, 4, 0]), response([3, 4, 0])]) as api:
            retriever.index_reports(self.reports)
            self.assertEqual(retriever.retrieve_reports("flood")[0]["report"]["report_id"], "one")
        documents, query = [json.loads(call.args[0].data) for call in api.call_args_list]
        self.assertEqual(documents["taskType"], "RETRIEVAL_DOCUMENT")
        self.assertEqual(query["taskType"], "RETRIEVAL_QUERY")
        self.assertEqual(documents["model"], query["model"])
        self.assertEqual(documents["outputDimensionality"], 3)
        self.assertNotIn("test-key", api.call_args.args[0].full_url)
        stored = self.client.scroll(retriever.store.collections["reports"], with_vectors=True)[0][0]
        np.testing.assert_allclose(stored.vector, [0.6, 0.8, 0], atol=1e-6)
        self.assertEqual(stored.payload["embedding_profile"]["model"], "gemini-embedding-001")

    def test_api_failure_never_falls_back_and_preserves_snapshot(self):
        retriever = self.retriever("gemini", 3)
        with patch("src.embeddings.urllib.request.urlopen", return_value=response([1, 0, 0])):
            retriever.index_reports(self.reports)
        with patch("src.embeddings.urllib.request.urlopen", side_effect=[response([0, 1, 0]), OSError("offline")]):
            with self.assertRaises(EmbeddingError):
                retriever.index_reports(self.reports + [{"report_id": "two", "text": "bridge"}])
        points = self.client.scroll(retriever.store.collections["reports"], with_vectors=True)[0]
        self.assertEqual(len(points), 1)
        np.testing.assert_allclose(points[0].vector, [1, 0, 0])
        with patch("src.embeddings.urllib.request.urlopen", side_effect=OSError("offline")):
            with self.assertRaises(EmbeddingError):
                retriever.retrieve_reports("flood")

    def test_invalid_api_vectors_are_not_cropped_padded_or_accepted(self):
        embedder = TextEmbedder(3, "gemini", "test-key")
        for vector in ([1, 2], [1, 2, 3, 4], [], [0, 0, 0], [1, float("nan"), 2], [1, float("inf"), 2]):
            with self.subTest(vector=vector), patch("src.embeddings.urllib.request.urlopen", return_value=response(vector)):
                with self.assertRaises(EmbeddingError):
                    embedder.embed("flood")

    def test_provider_profiles_and_legacy_collections_are_isolated(self):
        self.client.create_collection("crisisnav_reports_128", vectors_config=models.VectorParams(size=128, distance=models.Distance.COSINE))
        self.client.upsert("crisisnav_reports_128", [models.PointStruct(id=1, vector=[1.0] * 128, payload={"report": self.reports[0]})])
        hashed = self.retriever()
        gemini = self.retriever("gemini")
        self.assertNotEqual(hashed.store.collections, gemini.store.collections)
        self.assertEqual(hashed.retrieve_reports("flood"), [])
        hashed.index_reports(self.reports)
        self.assertEqual(len(hashed.retrieve_reports("flood")), 1)
        self.assertEqual(gemini.store.reports_snapshot(), [])
        self.assertEqual(self.client.count("crisisnav_reports_128").count, 1)

    def test_stored_profile_mismatch_is_rejected(self):
        retriever = self.retriever()
        retriever.index_reports(self.reports)
        collection = retriever.store.collections["reports"]
        self.client.set_payload(collection, {"embedding_profile": {"provider": "wrong"}}, points=models.Filter())
        with self.assertRaisesRegex(ValueError, "incompatible"):
            retriever.retrieve_reports("flood")
        with self.assertRaisesRegex(ValueError, "incompatible"):
            retriever.store.reports_snapshot()
        retriever.index_reports(self.reports)
        self.assertEqual(len(retriever.retrieve_reports("flood")), 1)

    def test_store_validates_whole_vector_batch_before_writing(self):
        retriever = self.retriever()
        retriever.index_reports(self.reports)
        for vector in (np.ones(127), np.full(128, np.nan)):
            with self.assertRaises(ValueError):
                retriever.store.replace("reports", [{"id": "bad", "embedding": vector, "payload": {}}])
        self.assertEqual(retriever.store.reports_snapshot(), self.reports)

    def test_image_provider_failure_is_not_swallowed(self):
        retriever = self.retriever("gemini", 3)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "flood.png")
            Image.new("RGB", (64, 64), "blue").save(path)
            with patch("src.embeddings.urllib.request.urlopen", side_effect=OSError("offline")):
                with self.assertRaises(EmbeddingError):
                    retriever.index_images([{"image_path": path}])
        self.assertEqual(self.client.count(retriever.store.collections["images"]).count, 0)

    def test_persistence_across_instances_retains_same_profile(self):
        first = self.retriever()
        first.index_reports(self.reports)
        second = self.retriever()
        self.assertEqual(first.store.collections, second.store.collections)
        self.assertEqual(first.retrieve_reports("flood"), second.retrieve_reports("flood"))

    def test_vehicle_boundaries_and_aliases_share_configuration(self):
        graph = CrisisSpatialGraph(str(ROOT / "data/sample_osm_network.json"))
        edge = {"traversable": True, "damage_severity": 0, "hazard_risk": 0, "base_time_min": 1}
        for name, profile in VEHICLE_PROFILES.items():
            edge["flood_depth_cm"] = profile["max_flood_cm"]
            self.assertNotEqual(graph.calculate_edge_weight("a", "b", edge, name), float("inf"))
            edge["flood_depth_cm"] += 0.1
            self.assertEqual(graph.calculate_edge_weight("a", "b", edge, name), float("inf"))
        self.assertEqual(normalize_vehicle_type("heavy_truck"), "high_clearance_truck")
        with self.assertRaises(ValueError):
            graph.calculate_edge_weight("a", "b", edge, "typo")

    def test_cargo_word_does_not_override_explicit_vehicle_selection(self):
        graph = CrisisSpatialGraph(str(ROOT / "data/sample_osm_network.json"))
        agent = CrisisResponseAgent(graph, self.retriever())
        state = agent.decompose_query_node({"user_query": "cargo from Chembur Naka to Zen Hospital", "vehicle_type": "high_clearance_truck"})
        self.assertEqual(state["vehicle_type"], "high_clearance_truck")


if __name__ == "__main__":
    unittest.main()
