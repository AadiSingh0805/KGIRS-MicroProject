"""Persistent Qdrant storage shared across retrievers and Streamlit reloads."""
import os
import uuid
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
    def __init__(self, embedding_dim: int, client=None, collection_prefix=None):
        self.client = client if client is not None else configured_client()
        prefix = collection_prefix or os.getenv("QDRANT_COLLECTION_PREFIX", "crisisnav")
        self.collections = {
            kind: f"{prefix}_{kind}_{embedding_dim}" for kind in ("reports", "images")
        }
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
        points = [models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"crisisnav:{kind}:{r['id']}")),
            vector=r["embedding"].tolist(), payload=r["payload"],
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
        return self.client.query_points(
            collection_name=self.collections[kind], query=vector.tolist(),
            limit=top_k, with_payload=True, with_vectors=False,
        ).points
