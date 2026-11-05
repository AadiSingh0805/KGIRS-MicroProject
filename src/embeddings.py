"""Explicit, versioned text embeddings. A failed provider never changes spaces."""
import hashlib
import json
import os
import urllib.error
import urllib.request
import numpy as np


class EmbeddingError(RuntimeError):
    pass


class TextEmbedder:
    def __init__(self, dimension=None, provider=None, api_key=None):
        self.provider = (provider or os.getenv("EMBEDDING_PROVIDER", "hash")).strip().lower()
        if self.provider not in ("hash", "gemini"):
            raise ValueError("EMBEDDING_PROVIDER must be hash or gemini")
        default_dim = 128 if self.provider == "hash" else 768
        try:
            self.dimension = int(dimension if dimension is not None else os.getenv("EMBEDDING_DIM") or default_dim)
        except (TypeError, ValueError) as error:
            raise ValueError("EMBEDDING_DIM must be a positive integer") from error
        if self.dimension <= 0 or (dimension is not None and self.dimension != dimension):
            raise ValueError("Embedding dimension must be a positive integer")
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = "semantic-hash-v1" if self.provider == "hash" else "gemini-embedding-001"
        if self.provider == "gemini":
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY is required when EMBEDDING_PROVIDER=gemini")
            if self.dimension > 3072:
                raise ValueError("gemini-embedding-001 supports at most 3072 dimensions")
        self.profile = {
            "provider": self.provider, "model": self.model,
            "dimension": self.dimension, "version": 1,
            "normalization": "l2", "document_task": "RETRIEVAL_DOCUMENT",
            "query_task": "RETRIEVAL_QUERY",
        }

    def embed(self, text, task_type="RETRIEVAL_DOCUMENT"):
        if task_type not in ("RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"):
            raise ValueError("Unsupported embedding task")
        if self.provider == "hash":
            return self._hash(text)
        request = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:embedContent",
            data=json.dumps({
                "model": f"models/{self.model}", "content": {"parts": [{"text": text}]},
                "taskType": task_type, "outputDimensionality": self.dimension,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                values = json.loads(response.read())["embedding"]["values"]
            vector = np.asarray(values, dtype=np.float32)
            if vector.shape != (self.dimension,) or not np.isfinite(vector).all():
                raise ValueError("Unexpected vector dimensions or non-finite values")
            norm = np.linalg.norm(vector)
            if not np.isfinite(norm) or norm == 0:
                raise ValueError("Invalid zero or non-finite embedding norm")
            return vector / norm
        except Exception as error:
            # Do not print response bodies, URLs, or credentials. Never fall back
            # to a different model for an individual document or query.
            raise EmbeddingError(
                f"Gemini embedding failed for {self.model} ({self.dimension} dimensions). "
                "No alternate embeddings were generated. Check credentials, connectivity, "
                "quota, and the configured dimension, then retry."
            ) from error

    def _hash(self, text):
        vector = np.zeros(self.dimension, dtype=np.float32)
        tokens = text.lower().replace(",", " ").replace(".", " ").replace(":", " ").split()
        for index, token in enumerate(tokens):
            hashed = int(hashlib.md5(token.encode()).hexdigest(), 16)
            position = hashed % self.dimension
            sign = 1.0 if (hashed >> 8) % 2 == 0 else -1.0
            weight = 1.0 + 1.0 / (index + 1)
            if any(word in token for word in ("flood", "water", "crack", "bridge", "collapse", "hospital", "landslide", "blocked")):
                weight *= 3.0
            vector[position] += sign * weight
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector
