"""ChromaDB-backed local vector index over synced communications."""
import logging

import chromadb

from config.settings import CHROMA_PATH
from src.rag.embedder import chunk_text, get_embedding_function

log = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, path: str = CHROMA_PATH, collection: str = "communications"):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name=collection,
            embedding_function=get_embedding_function(),
            metadata={"hnsw:space": "cosine"},
        )

    def index_records(self, records: list[dict]) -> int:
        """Chunk and upsert communication records. Returns chunks indexed."""
        ids, docs, metas = [], [], []
        for rec in records:
            full_text = f"From: {rec['sender']}\nSubject: {rec['subject']}\n\n{rec['content']}"
            for i, chunk in enumerate(chunk_text(full_text)):
                ids.append(f"{rec['id']}::chunk{i}")
                docs.append(chunk)
                metas.append(
                    {
                        "record_id": rec["id"],
                        "source": rec["source"],
                        "sender": rec["sender"],
                        "subject": rec["subject"],
                        "timestamp": rec["timestamp"],
                    }
                )
        if ids:
            self.collection.upsert(ids=ids, documents=docs, metadatas=metas)
        return len(ids)

    def search(self, query: str, k: int = 4) -> list[dict]:
        results = self.collection.query(query_texts=[query], n_results=k)
        hits = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            hits.append({"content": doc, "metadata": meta, "distance": dist})
        return hits

    def count(self) -> int:
        return self.collection.count()
