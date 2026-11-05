"""Persistent Qdrant storage shared across retrievers and Streamlit reloads."""
import os
import uuid
import hashlib
import json
import numpy as np
from functools import lru_cache, wraps
from threading import RLock
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# Qdrant Local is shared across Streamlit script threads. Serialize collection
# creation, whole snapshot writes, and searches against its SQLite/in-memory state.
_CLIENT_LOCK = RLock()


def _synchronized(operation):
    @wraps(operation)
    def locked(*args, **kwargs):
        with _CLIENT_LOCK:
            return operation(*args, **kwargs)
    return locked


@lru_cache(maxsize=None)
def _shared_client(url, api_key, path):
    # A local database permits one client per directory. Share it when Streamlit
    # invalidates its resource cache or reloads the retriever module.
    if url:
        return QdrantClient(url=url, api_key=api_key or None, timeout=30)
    # SQLite's thread affinity can be disabled because all application store
    # operations are protected by _CLIENT_LOCK, including complete snapshots.
    return QdrantClient(path=path, force_disable_check_same_thread=True)


@_synchronized
def configured_client():
    url = os.getenv("QDRANT_URL", "").strip()
    path = Path(os.getenv("QDRANT_PATH", "data/qdrant_storage")).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return _shared_client(url, os.getenv("QDRANT_API_KEY", ""), str(path.resolve()))


class QdrantVectorStore:
    """Separate cosine collections for reports and image feature vectors."""

    @_synchronized
    def __init__(self, embedding_dim: int, client=None, collection_prefix=None, embedding_profiles=None):
        self.client = client if client is not None else configured_client()
        prefix = collection_prefix or os.getenv("QDRANT_COLLECTION_PREFIX", "crisisnav")
        if not embedding_profiles or set(embedding_profiles) != {"reports", "images"}:
            raise ValueError("Versioned embedding profiles are required for both modalities")
        self.embedding_dim = embedding_dim
        self.embedding_profiles = embedding_profiles
        self.collections = {}
        for kind, profile in embedding_profiles.items():
            signature = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()[:16]
            self.collections[kind] = f"{prefix}_{kind}_{embedding_dim}_{signature}"

        for name in self.collections.values():
            if not self.client.collection_exists(name):
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=models.VectorParams(
                        size=embedding_dim, distance=models.Distance.COSINE
                    ),
                )
            params = self.client.get_collection(name).config.params.vectors
            if not isinstance(params, models.VectorParams) or (
                params.size != embedding_dim or params.distance != models.Distance.COSINE
            ):
                raise ValueError(f"Qdrant collection {name!r} must use {embedding_dim}-dimensional cosine vectors")

    @_synchronized
    def replace(self, kind: str, records: List[Dict[str, Any]]) -> None:
        """Sync a complete snapshot, upserting stable IDs before removing stale IDs.

        Only this application's dedicated collection is synchronized. Validation
        and embedding generation finish before this method writes anything.
        """
        # Validate the complete batch before the first write, so a bad vector
        # cannot partially replace a healthy snapshot.
        for record in records:
            self._validate_vector(record["embedding"])
        points = [models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"crisisnav:{kind}:{r['id']}")),
            vector=r["embedding"].tolist(), payload={
                **r["payload"], "embedding_profile": self.embedding_profiles[kind]
            },
        ) for r in records]
        name = self.collections[kind]
        for offset in range(0, len(points), 64):
            self.client.upsert(collection_name=name, points=points[offset:offset + 64], wait=True)
        stale = models.Filter(must_not=[models.HasIdCondition(has_id=[p.id for p in points])]) if points else models.Filter()
        self.client.delete(collection_name=name, points_selector=models.FilterSelector(filter=stale), wait=True)

    @_synchronized
    def search(self, kind: str, vector, top_k: int):
        if top_k <= 0:
            return []
        self._validate_vector(vector)
        if not np.any(vector):
            return []
        points = self.client.query_points(
            collection_name=self.collections[kind], query=vector.tolist(),
            query_filter=models.Filter(must_not=[models.FieldCondition(
                key="report.incident_status", match=models.MatchValue(value="resolved")
            )]) if kind == "reports" else None,
            limit=top_k, with_payload=True, with_vectors=False,
        ).points
        for point in points:
            self._validate_profile(kind, point.payload)
        return points

    def _validate_vector(self, vector):
        vector = np.asarray(vector)
        if vector.shape != (self.embedding_dim,) or not np.isfinite(vector).all():
            raise ValueError(f"Expected a finite {self.embedding_dim}-dimensional embedding")

    def _validate_profile(self, kind, payload):
        if not payload or payload.get("embedding_profile") != self.embedding_profiles[kind]:
            raise ValueError("Stored embedding profile is incompatible. Reindex this collection before searching.")

    @_synchronized
    def reports_snapshot(self):
        reports = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collections["reports"], limit=128,
                offset=offset, with_payload=True, with_vectors=False,
            )
            for point in points:
                self._validate_profile("reports", point.payload)
                reports.append(point.payload["report"])
            if offset is None:
                return reports
