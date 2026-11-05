"""Integration tests against real embedded Qdrant, without network or API keys."""
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from qdrant_client import QdrantClient, models

from src.multimodal_retriever import MultimodalRetriever
from src.qdrant_store import configured_client

ROOT = Path(__file__).resolve().parents[1]


class QdrantRetrieverTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict("os.environ", {"GEMINI_API_KEY": "", "EMBEDDING_PROVIDER": "hash", "EMBEDDING_DIM": "128"})
        self.env.start()
        self.client = QdrantClient(":memory:")
        self.r = MultimodalRetriever(128, qdrant_client=self.client, collection_prefix="test")
        self.reports = json.loads((ROOT / "data/disaster_reports.json").read_text())

    def tearDown(self):
        self.client.close()
        self.env.stop()

    def test_search_matches_previous_cosine_ranking(self):
        self.r.index_reports(self.reports)
        query = "monsoon flood inundation at Shell Colony"
        vector = self.r.compute_text_embedding(query)
        expected = []
        for report in self.reports:
            text = f"{report.get('incident_type', '')} {report.get('text', '')} {report.get('action_required', '')} {' '.join(report.get('affected_edges', []))}"
            emb = self.r.compute_text_embedding(text)
            score = float(np.dot(vector, emb) / (np.linalg.norm(vector) * np.linalg.norm(emb)))
            expected.append((score, report["report_id"]))
        expected.sort(reverse=True)
        actual = self.r.retrieve_reports(query, 3)
        self.assertEqual([r["report"]["report_id"] for r in actual], [r[1] for r in expected[:3]])
        for result, (score, _) in zip(actual, expected):
            self.assertAlmostEqual(result["score"], score, places=3)

    def test_reindex_updates_removes_and_clears(self):
        self.r.index_reports(self.reports)
        self.r.index_reports(self.reports)
        name = self.r.store.collections["reports"]
        self.assertEqual(self.client.count(name).count, len(self.reports))
        updated = dict(self.reports[0], text="updated flood report")
        self.r.index_reports([updated])
        self.assertEqual(self.client.count(name).count, 1)
        self.assertEqual(self.r.retrieve_reports("flood")[0]["report"]["text"], updated["text"])
        self.r.index_reports([])
        self.assertEqual(self.r.retrieve_reports("flood"), [])

    def test_validation_preserves_existing_data(self):
        self.r.index_reports(self.reports[:1])
        with self.assertRaises(ValueError):
            self.r.index_reports([self.reports[0], self.reports[0]])
        self.assertEqual(self.client.count(self.r.store.collections["reports"]).count, 1)
        self.assertEqual(self.r.retrieve_reports("flood", 0), [])
        self.assertEqual(self.r.retrieve_images(query_text="flood", top_k=-1), [])

    def test_persistence_after_close_and_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            client = QdrantClient(path=directory)
            r = MultimodalRetriever(128, qdrant_client=client, collection_prefix="persist")
            r.index_reports(self.reports)
            expected = r.retrieve_reports("bridge damage")
            client.close()
            client = QdrantClient(path=directory)
            try:
                reopened = MultimodalRetriever(128, qdrant_client=client, collection_prefix="persist")
                self.assertEqual(reopened.retrieve_reports("bridge damage"), expected)
            finally:
                client.close()

    def test_image_search_and_modality_isolation(self):
        self.r.index_reports(self.reports[:1])
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "flood.png")
            Image.new("RGB", (64, 64), "blue").save(path)
            meta = [{"image_id": "flood-image", "image_path": path,
                     "hazard_type": "flood", "associated_report": self.reports[0]["report_id"]}]
            self.r.index_images(meta)
            self.r.index_images(meta)
            result = self.r.retrieve_images(query_image_path=path)[0]
            self.assertEqual(result["image_id"], "flood-image")
            self.assertAlmostEqual(result["score"], 1.0, places=4)
            self.assertEqual(result["path"], path)
            self.assertTrue(self.r.retrieve_images(query_text="flood"))
            self.r.index_images([])
            self.assertEqual(self.r.retrieve_images(query_text="flood"), [])
            self.assertEqual(len(self.r.retrieve_reports("flood")), 1)

    def test_dimension_isolation_and_incompatible_collection(self):
        other = MultimodalRetriever(256, qdrant_client=self.client, collection_prefix="test")
        self.assertNotEqual(other.store.collections, self.r.store.collections)
        bad = MultimodalRetriever(128, qdrant_client=self.client, collection_prefix="bad")
        name = bad.store.collections["reports"]
        self.client.delete_collection(name)
        self.client.create_collection(name, vectors_config=models.VectorParams(size=64, distance=models.Distance.COSINE))
        with self.assertRaisesRegex(ValueError, "128-dimensional"):
            MultimodalRetriever(128, qdrant_client=self.client, collection_prefix="bad")

    def test_local_client_shared_across_retriever_instances(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict("os.environ", {"QDRANT_URL": "", "QDRANT_PATH": directory}):
                first = configured_client()
                self.assertIs(first, configured_client())
                first.close()

    def test_persistent_client_reused_from_streamlit_worker_threads(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict("os.environ", {"QDRANT_URL": "", "QDRANT_PATH": directory}):
                # Create SQLite on this thread, then recreate the retriever and
                # write from worker threads, as Streamlit does after cache clears.
                initial = MultimodalRetriever(128, collection_prefix="threads")
                initial.index_reports(self.reports)
                expected = initial.retrieve_reports("bridge flood")
                with tempfile.TemporaryDirectory() as images:
                    image_path = str(Path(images) / "bridge.png")
                    Image.new("RGB", (64, 64), "blue").save(image_path)

                    def rerun(_):
                        retriever = MultimodalRetriever(128, collection_prefix="threads")
                        self.assertIs(retriever.store.client, initial.store.client)
                        retriever.index_reports(self.reports)
                        retriever.index_images([{"image_path": image_path, "hazard_type": "flood"}])
                        result = retriever.retrieve_reports("bridge flood")
                        self.assertAlmostEqual(retriever.retrieve_images(query_image_path=image_path)[0]["score"], 1.0, places=4)
                        return result

                    try:
                        with ThreadPoolExecutor(max_workers=4) as executor:
                            results = list(executor.map(rerun, range(12)))
                        self.assertTrue(all(result == expected for result in results))
                    finally:
                        initial.store.client.close()

    def test_remote_configuration_and_errors_are_not_hidden(self):
        with patch.dict("os.environ", {"QDRANT_URL": "https://example.qdrant.io", "QDRANT_API_KEY": "test-key"}):
            with patch("src.qdrant_store._shared_client") as shared:
                configured_client()
                self.assertEqual(shared.call_args.args[:2], ("https://example.qdrant.io", "test-key"))
        with patch.object(self.client, "upsert", side_effect=RuntimeError("unavailable")):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                self.r.index_reports(self.reports)


if __name__ == "__main__":
    unittest.main()
