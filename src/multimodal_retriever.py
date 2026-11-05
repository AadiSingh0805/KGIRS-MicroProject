"""
Multimodal Geospatial Incident & Damage Retriever
Provides semantic vector search across incident dispatch texts and reconnaissance images.
Uses Gemini or deterministic text embeddings, visual features, and persistent Qdrant search.
"""

import os
import hashlib
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image
from .qdrant_store import QdrantVectorStore

try:
    import google.generativeai as genai
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False


class MultimodalRetriever:
    """
    Multimodal vector index and retriever for crisis dispatch reports and damage images.
    Stores embeddings and metadata in Qdrant for text, visual, and multimodal queries.
    """
    def __init__(self, embedding_dim: int = 256, api_key: Optional[str] = None,
                 qdrant_client=None, collection_prefix: Optional[str] = None):
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.store = QdrantVectorStore(embedding_dim, qdrant_client, collection_prefix)
        self.embedding_dim = embedding_dim
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self._setup_api()

    def _setup_api(self):
        if self.api_key and _GENAI_AVAILABLE:
            try:
                genai.configure(api_key=self.api_key)
            except Exception:
                pass

    def compute_text_embedding(self, text: str) -> np.ndarray:
        """
        Computes text embedding. Uses Gemini models/text-embedding-004 if available,
        falling back to deterministic semantic hashing and vocabulary term vector.
        """
        if self.api_key and _GENAI_AVAILABLE:
            for m_name in ["models/gemini-embedding-001", "models/text-embedding-004"]:
                try:
                    response = genai.embed_content(
                        model=m_name,
                        content=text,
                        task_type="retrieval_document"
                    )
                    raw_emb = response.get("embedding", {})
                    if isinstance(raw_emb, dict) and "values" in raw_emb:
                        emb = np.array(raw_emb["values"], dtype=np.float32)
                    elif isinstance(raw_emb, list):
                        emb = np.array(raw_emb, dtype=np.float32)
                    else:
                        emb = np.array(response.get("embedding", []), dtype=np.float32)
                        
                    if len(emb) != self.embedding_dim:
                        if len(emb) > self.embedding_dim:
                            emb = emb[:self.embedding_dim]
                        else:
                            emb = np.pad(emb, (0, self.embedding_dim - len(emb)))
                    norm = np.linalg.norm(emb)
                    return emb / (norm + 1e-8)
                except Exception:
                    continue

        # Robust High-Dimension Semantic Fallback Vector
        vec = np.zeros(self.embedding_dim, dtype=np.float32)
        tokens = text.lower().replace(",", " ").replace(".", " ").replace(":", " ").split()
        for idx, token in enumerate(tokens):
            # Seeded deterministic projection
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            pos = h % self.embedding_dim
            sign = 1.0 if ((h >> 8) % 2 == 0) else -1.0
            weight = 1.0 + (1.0 / (idx + 1))
            
            # Boost emergency keywords
            if any(k in token for k in ["flood", "water", "crack", "bridge", "collapse", "hospital", "landslide", "blocked"]):
                weight *= 3.0
            vec[pos] += sign * weight

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def compute_image_embedding(self, image_path: str) -> np.ndarray:
        """
        Extracts visual damage feature vector (color histograms, edge density,
        thermal/structural indicators) calibrated for crisis damage imagery.
        """
        vec = np.zeros(self.embedding_dim, dtype=np.float32)
        if not os.path.exists(image_path):
            return vec

        try:
            with Image.open(image_path) as img:
                img_rgb = img.convert("RGB")
                img_resized = img_rgb.resize((64, 64))
                arr = np.array(img_resized, dtype=np.float32) / 255.0

                # 1. Color channel moments (Red = hazard/fire, Blue = water, Brown = landslide)
                r_mean, g_mean, b_mean = arr[:, :, 0].mean(), arr[:, :, 1].mean(), arr[:, :, 2].mean()
                r_std, g_std, b_std = arr[:, :, 0].std(), arr[:, :, 1].std(), arr[:, :, 2].std()

                # 2. Gradient / Edge magnitude (structural cracks and debris)
                dx = np.diff(arr, axis=1)
                dy = np.diff(arr, axis=0)
                edge_energy = float(np.mean(np.abs(dx)) + np.mean(np.abs(dy)))

                # 3. Spatial quadrant distribution
                quadrants = [
                    arr[:32, :32], arr[:32, 32:],
                    arr[32:, :32], arr[32:, 32:]
                ]
                quad_features = []
                for q in quadrants:
                    quad_features.extend([q[:, :, 0].mean(), q[:, :, 1].mean(), q[:, :, 2].mean()])

                # Compose feature slots
                feature_list = [r_mean, g_mean, b_mean, r_std, g_std, b_std, edge_energy] + quad_features
                for i, val in enumerate(feature_list):
                    idx = (i * 13) % self.embedding_dim
                    vec[idx] += val

                # High-level domain hazard projection
                filename = os.path.basename(image_path).lower()
                name_emb = self.compute_text_embedding(filename)
                vec = 0.6 * vec + 0.4 * name_emb

                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec /= norm
        except Exception as e:
            print(f"Warning: Failed to extract image embedding for {image_path}: {e}")

        return vec

    def index_reports(self, reports: List[Dict[str, Any]]) -> None:
        """Synchronize the complete report snapshot into Qdrant."""
        records = []
        seen = set()
        for report in reports:
            report_id = report.get("report_id")
            if not report_id or report_id in seen:
                raise ValueError("Each report requires a unique, non-empty report_id")
            seen.add(report_id)
            text = f"{report.get('incident_type', '')} {report.get('text', '')} {report.get('action_required', '')} {' '.join(report.get('affected_edges', []))}"
            records.append({
                "id": report_id,
                "embedding": self.compute_text_embedding(text),
                "payload": {"report_id": report_id, "report": report, "text_corpus": text},
            })
        self.store.replace("reports", records)

    def index_images(self, image_metadata: List[Dict[str, Any]]) -> None:
        """Synchronize the complete image snapshot into Qdrant."""
        records = []
        seen = set()
        for image in image_metadata:
            path = image.get("image_path")
            if not path or not os.path.isfile(path):
                raise ValueError(f"Image file does not exist: {path!r}")
            image_id = image.get("image_id") or os.path.basename(path)
            if image_id in seen:
                raise ValueError("Each image requires a unique image_id (or filename)")
            seen.add(image_id)
            records.append({
                "id": image_id,
                "embedding": self.compute_image_embedding(path),
                "payload": {"image_id": image_id, "path": path,
                            "associated_report": image.get("associated_report"),
                            "hazard_type": image.get("hazard_type")},
            })
        self.store.replace("images", records)

    def retrieve_reports(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search Qdrant, including reports persisted by previous app runs."""
        if top_k <= 0:
            return []
        matches = self.store.search("reports", self.compute_text_embedding(query), top_k)
        return [{"score": round(hit.score, 4), "report": hit.payload["report"]}
                for hit in matches]

    def retrieve_images(self, query_text: Optional[str] = None, query_image_path: Optional[str] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search persisted image vectors using the existing multimodal features."""
        if top_k <= 0:
            return []
        if query_image_path and os.path.exists(query_image_path):
            embedding = self.compute_image_embedding(query_image_path)
        elif query_text:
            embedding = self.compute_text_embedding(query_text)
        else:
            return []
        matches = self.store.search("images", embedding, top_k)
        return [{"score": round(hit.score, 4), "image_id": hit.payload["image_id"],
                 "path": hit.payload["path"], "hazard_type": hit.payload.get("hazard_type")}
                for hit in matches]

    def retrieve_hybrid_context(
        self,
        query: str,
        graph_store: Any,
        top_k: int = 4
    ) -> Dict[str, Any]:
        """
        Combines semantic incident retrieval with spatial graph topology.
        Discovers related edges, nodes, and capacity bottlenecks.
        """
        text_matches = self.retrieve_reports(query, top_k=top_k)
        
        affected_edges = set()
        affected_nodes = set()
        retrieved_reports = []

        for m in text_matches:
            rep = m["report"]
            retrieved_reports.append(rep)
            for eid in rep.get("affected_edges", []):
                affected_edges.add(eid)
            for nid in rep.get("affected_nodes", []):
                affected_nodes.add(nid)

        # Graph Context Enrichment
        subgraph_nodes = []
        for nid in affected_nodes:
            if nid in graph_store.graph:
                ndata = graph_store.graph.nodes[nid]
                subgraph_nodes.append({
                    "id": nid,
                    "name": ndata.get("name"),
                    "type": ndata.get("type"),
                    "status": ndata.get("operational_status"),
                    "available_capacity": ndata.get("available_capacity")
                })

        return {
            "query": query,
            "matched_reports": text_matches,
            "impacted_edge_ids": list(affected_edges),
            "impacted_node_ids": list(affected_nodes),
            "graph_nodes_context": subgraph_nodes
        }

