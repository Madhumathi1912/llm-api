from openai import OpenAIError

from app.config import settings
from app.services.openai_client import OpenAIClient, openai_client
from app.services.embedding_client import EmbeddingClient, embedding_client
from app.services.vector_store import VectorStore, vector_store
from app.services.chunking import Chunker, chunker
from app.services.cost_logger import CostLogger, cost_logger
from app.services.text_extractor import TextExtractor, text_extractor

RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY "
    "the context provided below. If the answer is not contained in the "
    "context, say you don't know rather than guessing."
)


class RagService:
    """
    Orchestrates the full RAG flow: chunking + embedding for ingestion,
    and embed-query -> retrieve -> augment-prompt -> generate for
    answering questions. Depends on EmbeddingClient, VectorStore,
    OpenAIClient, and CostLogger through their public interfaces only —
    none of their internals are assumed here.
    """
    def __init__(
        self, 
        text_extractor_: TextExtractor,
        chunker_: Chunker,
        embedding_client_: EmbeddingClient,
        vector_store_: VectorStore,
        openai_client_: OpenAIClient,
        cost_logger_: CostLogger,
    ):
        self._text_extractor = text_extractor_
        self._chunker = chunker_
        self._embedding_client = embedding_client_
        self._vector_store = vector_store_
        self._openai_client = openai_client_
        self._cost_logger = cost_logger_


    def ingest_document(self, text: str, source: str) -> int:
        """
        Chunks a document, embeds each chunk, and stores them in the
        vector store. Returns the number of chunks created.
        """
        chunks = self._chunker.chunk(text)
        vectors = self._embedding_client.embed_batch(chunks)
        self._vector_store.add_chunks_batch(chunks, vectors, source=source)
        return len(chunks)


    def ingest_uploaded_file(self, filename: str, file_bytes: bytes, source: str=None) -> tuple[str, int]:
        """
        Extracts text from an uploaded file, then runs it through the SAME ingestion pipeline/same logic, as ingest_document(). 
        Returns the resolved source label and the number of chunks created.
        """
        text = self._text_extractor.extract(filename, file_bytes)
        if not text.strip():
            raise ValueError("No extractable text found in the uploaded file (possibly a scanned/image-only PDF).")
        resolved_source = source or filename
        chunks_created = self.ingest_document(text, resolved_source)
        return resolved_source, chunks_created
    

    def _build_augmented_prompt(self, question: str, retrieved_chunks: list[dict]) -> str:
        context = "\n\n".join(
            f"[Source: {chunk['source']}]\n{chunk['text']}" for chunk in retrieved_chunks
        )
        return f"Context:\n{context}\n\nQuestion: {question}"
    

    def ask(self, question: str, top_k: int=3) -> dict:
        """
        Full RAG flow: embed the question, retrieve the most similar
        chunks, build an augmented prompt, and call the LLM to generate
        an answer grounded in that retrieved context.
        """
        query_vector = self._embedding_client.embed(question)
        retrieved_chunks = self._vector_store.search(query_vector, top_k)
        augmented_prompt = self._build_augmented_prompt(question, retrieved_chunks)

        try:
            completion = self._openai_client.create_completion(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": RAG_SYSTEM_PROMPT},
                    {"role": "user", "content": augmented_prompt},
                ],
                temperature=0.3,  #lower temperature — faithful, not creative, answers
                max_tokens=400,
            )
        except OpenAIError as e:
            raise e
        
        usage = completion.usage
        self._cost_logger.log_usage(
            endpoint="/rag/ask",
            model=completion.model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cached=False,
        )

        return {
            'answer': completion.choices[0].message.content,
            'retrieved_chunks': retrieved_chunks,
            'model': completion.model,
            'prompt_tokens': usage.prompt_tokens,
            'completion_tokens': usage.completion_tokens
        }
        

# Single shared instance, built from the already-existing singletons
rag_service = RagService(
    text_extractor_=text_extractor,
    chunker_=chunker,
    embedding_client_=embedding_client,
    vector_store_=vector_store,
    openai_client_=openai_client,
    cost_logger_=cost_logger,
)