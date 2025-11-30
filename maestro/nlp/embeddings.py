"""Embedding utilities and FAISS-backed index."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, List, Tuple

import faiss  # type: ignore
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from maestro.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingModel(ABC):
    """Abstract embedding model."""

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Return embeddings for the provided texts."""


class HFEmbeddingModel(EmbeddingModel):
    """SentenceTransformers-based embedding model."""

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        name = model_name or settings.embedding_model_name
        self.device = device or settings.device
        self.model = SentenceTransformer(name, device=self.device)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, convert_to_numpy=True, device=self.device, batch_size=8)
        return embeddings.astype("float32")


class EmbeddingIndex(ABC):
    """Abstract vector index."""

    @abstractmethod
    def add_items(self, ids: List[int], vectors: np.ndarray) -> None:
        """Add vectors to the index."""

    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
        """Return (id, score) pairs for nearest vectors."""

    @abstractmethod
    def persist(self) -> None:
        """Persist index to disk."""


class FaissEmbeddingIndex(EmbeddingIndex):
    """FAISS-backed index with optional GPU acceleration."""

    def __init__(self, dim: int, index_path: Path | None = None, use_gpu: bool = torch.cuda.is_available()) -> None:
        self.dim = dim
        self.index_path = Path(index_path or settings.faiss_index_path)
        self.use_gpu = use_gpu
        self.index = faiss.IndexFlatL2(dim)
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        if self.index_path.exists():
            self._load()

    def add_items(self, ids: List[int], vectors: np.ndarray) -> None:
        if len(ids) != vectors.shape[0]:
            raise ValueError("ids and vectors length mismatch")
        self.index.add_with_ids(vectors, np.array(ids, dtype="int64"))
        logger.info("Added %s vectors to index", len(ids))
        self.persist()

    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
        distances, indices = self.index.search(query_vector.astype("float32"), k)
        results: List[Tuple[int, float]] = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            results.append((int(idx), float(dist)))
        return results

    def persist(self) -> None:
        cpu_index = self.index
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(cpu_index, str(self.index_path))

    def _load(self) -> None:
        logger.info("Loading FAISS index from %s", self.index_path)
        cpu_index = faiss.read_index(str(self.index_path))
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        else:
            self.index = cpu_index

