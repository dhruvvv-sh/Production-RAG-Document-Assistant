from functools import lru_cache
from typing import Protocol

from sentence_transformers import CrossEncoder

from app.config import get_settings


class SupportValidator(Protocol):
    def score(self, claim: str, passages: list[str]) -> list[float]: ...


class NLISupportValidator:
    """Scores whether each evidence passage entails a generated claim."""

    def __init__(self, model_name: str, batch_size: int) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model

    def _entailment_index(self) -> int:
        labels = self.model.model.config.id2label
        for index, label in labels.items():
            if str(label).casefold() == "entailment":
                return int(index)
        raise ValueError("Faithfulness model does not define an entailment label")

    def score(self, claim: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        tokenizer = self.model.tokenizer
        claim_tokens = tokenizer.encode(claim, add_special_tokens=False)
        special_tokens = tokenizer.num_special_tokens_to_add(pair=True)
        window_size = max(1, self.model.max_seq_length - len(claim_tokens) - special_tokens)
        overlap = min(64, window_size // 4)
        step = max(1, window_size - overlap)
        pairs: list[tuple[str, str]] = []
        owners: list[int] = []
        for owner, passage in enumerate(passages):
            passage_tokens = tokenizer.encode(passage, add_special_tokens=False)
            if not passage_tokens:
                pairs.append((passage, claim))
                owners.append(owner)
                continue
            for start in range(0, len(passage_tokens), step):
                window = passage_tokens[start : start + window_size]
                premise = tokenizer.decode(window, skip_special_tokens=True)
                pairs.append((premise, claim))
                owners.append(owner)
                if start + window_size >= len(passage_tokens):
                    break

        probabilities = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            apply_softmax=True,
        )
        entailment_index = self._entailment_index()
        scores = [0.0] * len(passages)
        for owner, prediction in zip(owners, probabilities, strict=True):
            scores[owner] = max(scores[owner], float(prediction[entailment_index]))
        return scores


@lru_cache
def get_support_validator() -> NLISupportValidator:
    settings = get_settings()
    return NLISupportValidator(settings.faithfulness_model, settings.faithfulness_batch_size)
