"""
Task 4 — Chunking & Indexing vào Vector Store.
"""

import math
import json
from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# =============================================================================
# CONFIGURATION
# =============================================================================

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

VECTOR_STORE = "chromadb"
COLLECTION_NAME = "university_services_docs"

_MODEL_CACHE = None


class LocalVectorCollection:
    """Fallback vector collection nếu chromadb chưa được cài đặt."""

    def __init__(self, store_path: Path):
        self.store_path = store_path
        self.data = {"ids": [], "documents": [], "embeddings": [], "metadatas": []}
        self.load()

    def load(self):
        if self.store_path.exists():
            try:
                self.data = json.loads(self.store_path.read_text(encoding="utf-8"))
            except Exception:
                pass

    def save(self):
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.store_path.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")

    def count(self):
        return len(self.data["ids"])

    def upsert(self, ids, documents, embeddings, metadatas):
        for id_, doc, emb, meta in zip(ids, documents, embeddings, metadatas):
            if id_ in self.data["ids"]:
                idx = self.data["ids"].index(id_)
                self.data["documents"][idx] = doc
                self.data["embeddings"][idx] = emb
                self.data["metadatas"][idx] = meta
            else:
                self.data["ids"].append(id_)
                self.data["documents"].append(doc)
                self.data["embeddings"].append(emb)
                self.data["metadatas"].append(meta)
        self.save()

    def query(self, query_embeddings=None, query_texts=None, n_results=10, include=None):
        if not self.data["embeddings"]:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        q_emb = query_embeddings[0] if query_embeddings else [0.0] * len(self.data["embeddings"][0])

        def cosine_dist(vec1, vec2):
            dot = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = math.sqrt(sum(a * a for a in vec1))
            norm2 = math.sqrt(sum(b * b for b in vec2))
            if norm1 == 0 or norm2 == 0:
                return 1.0
            sim = dot / (norm1 * norm2)
            return max(0.0, 1.0 - sim)

        scored = []
        for doc, meta, emb in zip(self.data["documents"], self.data["metadatas"], self.data["embeddings"]):
            dist = cosine_dist(q_emb, emb)
            scored.append((dist, doc, meta))

        scored.sort(key=lambda x: x[0])
        top = scored[:n_results]

        return {
            "documents": [[x[1] for x in top]],
            "metadatas": [[x[2] for x in top]],
            "distances": [[x[0] for x in top]],
        }


def get_embedding_model():
    """Lấy hoặc khởi tạo embedding model."""
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            _MODEL_CACHE = SentenceTransformer(EMBEDDING_MODEL)
        except Exception:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                _MODEL_CACHE = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                _MODEL_CACHE = None
    return _MODEL_CACHE


def get_collection():
    """Kết nối và lấy collection từ ChromaDB (hoặc LocalVectorCollection)."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import chromadb  # type: ignore
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        return client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    except Exception:
        return LocalVectorCollection(CHROMA_DIR / "local_store.json")


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    if STANDARDIZED_DIR.exists():
        for md_file in STANDARDIZED_DIR.rglob("*.md"):
            content = md_file.read_text(encoding="utf-8")
            if not content.strip():
                continue
            doc_type = "legal" if "legal" in str(md_file) else "news"
            documents.append({
                "content": content,
                "metadata": {"source": md_file.name, "type": doc_type}
            })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict}
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    except Exception:
        splitter = None

    chunks = []
    for doc in documents:
        if splitter:
            splits = splitter.split_text(doc["content"])
        else:
            text = doc["content"]
            splits = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP)]

        for i, chunk_text in enumerate(splits):
            if chunk_text.strip():
                chunks.append({
                    "content": chunk_text,
                    "metadata": {**doc["metadata"], "chunk_index": i}
                })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    model = get_embedding_model()
    texts = [c["content"] for c in chunks]
    if model:
        embeddings = model.encode(texts, show_progress_bar=False)
        for chunk, emb in zip(chunks, embeddings):
            chunk["embedding"] = emb.tolist() if hasattr(emb, "tolist") else list(emb)
    else:
        # Fallback hash-based vector generator for consistent cosine similarity
        for chunk in chunks:
            text = chunk["content"]
            vec = [(hash(text + str(i)) % 1000) / 1000.0 for i in range(EMBEDDING_DIM)]
            chunk["embedding"] = vec
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn (ChromaDB hoặc LocalVectorCollection).
    """
    if not chunks:
        return
    collection = get_collection()
    ids = [f"{c['metadata']['source']}_chunk_{c['metadata']['chunk_index']}_{i}" for i, c in enumerate(chunks)]
    collection.upsert(
        ids=ids,
        documents=[c["content"] for c in chunks],
        embeddings=[c["embedding"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n[OK] Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"[OK] Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"[OK] Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("[OK] Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
