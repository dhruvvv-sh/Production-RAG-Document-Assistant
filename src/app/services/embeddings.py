import math
from functools import lru_cache
from typing import Protocol

from sentence_transformers import SentenceTransformer

from app.config import get_settings


class Embedder(Protocol):
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """Lazy sentence-transformer adapter that returns unit-normalized vectors."""

    def __init__(self, model_name: str, dimension: int) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
            actual_dimension = self._model.get_embedding_dimension()
            if actual_dimension != self.dimension:
                raise ValueError(
                    f"Embedding model produces {actual_dimension} dimensions; "
                    f"the database expects {self.dimension}"
                )
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        tokenizer = self.model.tokenizer
        special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
        window_size = max(1, self.model.max_seq_length - special_tokens)
        overlap = min(32, window_size // 4)
        step = max(1, window_size - overlap)
        windows: list[str] = []
        owners: list[int] = []
        for owner, text in enumerate(texts):
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if not token_ids:
                windows.append(text)
                owners.append(owner)
                continue
            for start in range(0, len(token_ids), step):
                window = token_ids[start : start + window_size]
                windows.append(tokenizer.decode(window, skip_special_tokens=True))
                owners.append(owner)
                if start + window_size >= len(token_ids):
                    break

        window_vectors = self.model.encode(
            windows,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        sums = [[0.0] * self.dimension for _ in texts]
        counts = [0] * len(texts)
        for owner, vector in zip(owners, window_vectors, strict=True):
            sums[owner] = [total + value for total, value in zip(sums[owner], vector, strict=True)]
            counts[owner] += 1

        results: list[list[float]] = []
        for vector, count in zip(sums, counts, strict=True):
            averaged = [value / count for value in vector]
            norm = math.sqrt(sum(value * value for value in averaged)) or 1.0
            results.append([value / norm for value in averaged])
        return results

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


@lru_cache
def get_embedder() -> SentenceTransformerEmbedder:
    settings = get_settings()
    return SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_dimension)
