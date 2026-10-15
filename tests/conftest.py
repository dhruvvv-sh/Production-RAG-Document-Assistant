import os
import re
from math import sqrt

os.environ["DATABASE_URL"] = "sqlite+pysqlite:///:memory:"

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import Settings, get_settings
from app.database import Base, get_db
from app.main import app
from app.services.embeddings import get_embedder
from app.services.reranking import get_reranker


class FakeEmbedder:
    dimension = 384

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in re.findall(r"\w+", text.casefold()):
            index = sum((position + 1) * ord(character) for position, character in enumerate(token))
            vector[index % self.dimension] += 1.0
        norm = sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class FakeReranker:
    def score(self, query: str, passages: list[str]) -> list[float]:
        query_terms = set(re.findall(r"\w+", query.casefold()))
        return [
            float(len(query_terms & set(re.findall(r"\w+", passage.casefold()))))
            for passage in passages
        ]


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, expire_on_commit=False)()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)


@pytest.fixture
def client(db_session):
    def override_db():
        yield db_session

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_embedder] = lambda: FakeEmbedder()
    app.dependency_overrides[get_reranker] = lambda: FakeReranker()
    app.dependency_overrides[get_settings] = lambda: Settings(
        database_url="sqlite+pysqlite:///:memory:",
        chunk_min_tokens=20,
        chunk_max_tokens=30,
        chunk_overlap_tokens=5,
    )
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
