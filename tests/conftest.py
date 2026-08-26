from unittest.mock import MagicMock
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def mock_sentence_transformer(monkeypatch):
    """Bypasses heavy model loading and returns mock embeddings instantly."""

    class FakeEmbedder:

        def get_embedding_dimension(self):
            return 384

        def get_sentence_embedding_dimension(self):
            """Alias for SentenceTransformer compatibility."""
            return self.get_embedding_dimension()

        def encode(self, sentences, **kwargs):
            if isinstance(sentences, str):
                sentences = [sentences]
            return np.zeros((len(sentences), 384), dtype=np.float32)

    monkeypatch.setattr(
        "core.memory.semantic_memory.SentenceTransformer",
        lambda model_name: FakeEmbedder(),
    )
