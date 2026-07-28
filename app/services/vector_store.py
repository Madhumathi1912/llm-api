import uuid
import logging

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tenacity import retry, stop_after_attempt, wait_fixed, before_sleep_log

from app.config import settings
from app.services.embedding_client import EMBEDDING_DIMENSIONS

logger = logging.getLogger(__name__)

COLLECTION_NAME = "documents"


class VectorStore:
    """
    Owns ONE responsibility: storing chunk vectors and searching them by
    similarity. Knows nothing about embeddings generation or RAG prompt
    assembly — RagService orchestrates those on top of this.
    """

    def __init__(self):
        self._client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        self._ensure_collection_exists()

    @retry(
        wait=wait_fixed(3),  # wait 3s between attempts
        stop=stop_after_attempt(20),  # up to ~60s total — enough even for slow first-time startup
        before_sleep=before_sleep_log(logger, logging.INFO),
        reraise=True,
    )
    def _ensure_collection_exists(self) -> None:
        """
        Creates the collection on first run. Safe to call every startup —
        does nothing if it already exists.

        Retried on ANY exception (connection refused, timeouts, etc.) —
        at app startup, Qdrant's container may still be initializing even
        though it's already "started" per Docker Compose's basic
        depends_on. This retry loop gives it time to become reachable
        instead of crashing the app on the very first attempt.
        """
        print(f"[VectorStore] Attempting to connect to Qdrant at {settings.qdrant_host}:{settings.qdrant_port}...")
        existing = [c.name for c in self._client.get_collections().collections]
        if COLLECTION_NAME not in existing:
            self._client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIMENSIONS,
                    distance=Distance.COSINE,  # standard choice for text embeddings
                ),
            )
            logger.info(f"Created Qdrant collection '{COLLECTION_NAME}'")


    def add_chunk(self, text: str, vector: list[float], source: str) -> None:
        """
        Stores one chunk. The payload (text + source) is what you get
        back on search — the vector itself is just numbers, useless
        without the original text attached.
        """
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={"text": text, "source": source},
        )
        self._client.upsert(collection_name=COLLECTION_NAME, points=[point])


    def add_chunks_batch(self, texts: list[str], vectors: list[list[float]], source: str) -> None:
        """
        Stores multiple chunks in a single call — more efficient than
        calling add_chunk() in a loop when indexing a whole document.
        """
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": text, "source": source},
            )
            for text, vector in zip(texts, vectors)
        ]
        self._client.upsert(collection_name=COLLECTION_NAME, points=points)


    def search(self, query_vector: list[float], top_k: int = 3) -> list[dict]:
        """
        Returns the top_k most similar chunks to the given query vector.
        Each result includes the original text, source, and similarity score.
        """
        results = self._client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
        ).points

        return [
            {
                "text": r.payload["text"],
                "source": r.payload["source"],
                "score": r.score,  # cosine similarity, higher = more similar
            }
            for r in results
        ]


# Single shared instance, same pattern as cache_client.py / cost_logger.py.
vector_store = VectorStore()