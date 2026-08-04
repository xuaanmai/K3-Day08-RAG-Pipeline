"""
Task 4 — Chunking & Indexing vào Vector Store.
"""

import hashlib
import math
import json
import os
import re
from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
MODEL_CACHE_DIR = Path(__file__).parent.parent / ".cache" / "huggingface"

# Keep model downloads inside the writable project instead of the user profile.
os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(MODEL_CACHE_DIR / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(MODEL_CACHE_DIR / "transformers"))

# =============================================================================
# CONFIGURATION
# =============================================================================

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

VECTOR_STORE = "chromadb"
COLLECTION_NAME = "ielts_writing_docs_v3"

_MODEL_CACHE = None
_MODEL_LOAD_ATTEMPTED = False


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
    global _MODEL_CACHE, _MODEL_LOAD_ATTEMPTED
    if _MODEL_LOAD_ATTEMPTED:
        return _MODEL_CACHE

    _MODEL_LOAD_ATTEMPTED = True
    allow_download = os.getenv("ALLOW_MODEL_DOWNLOAD", "0").strip() == "1"
    if not allow_download and not MODEL_CACHE_DIR.exists():
        return None
    if _MODEL_CACHE is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            _MODEL_CACHE = SentenceTransformer(
                EMBEDDING_MODEL,
                cache_folder=str(MODEL_CACHE_DIR),
                local_files_only=not allow_download,
            )
        except Exception:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                _MODEL_CACHE = SentenceTransformer(
                    "all-MiniLM-L6-v2",
                    cache_folder=str(MODEL_CACHE_DIR),
                    local_files_only=not allow_download,
                )
            except Exception:
                _MODEL_CACHE = None
    return _MODEL_CACHE


def fallback_embedding(text: str, dimension: int = EMBEDDING_DIM) -> list[float]:
    """Create a deterministic local feature-hashing embedding.

    This is an offline fallback, not a replacement for BGE-M3. Unlike Python's
    built-in ``hash()``, BLAKE2 is stable across processes, so indexed document
    vectors remain compatible with query vectors after restarting the app.
    """
    vector = [0.0] * dimension
    tokens = re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        index = value % dimension
        sign = 1.0 if (value >> 1) & 1 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm:
        vector = [value / norm for value in vector]
    return vector


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
    Đọc toàn bộ markdown files đã được chuẩn hoá.

    ``data/standardized`` là output chính thức của Task 3. Một số
    dataset cũ trong repository dùng tên ``data/processed``; chỉ dùng
    thư mục này khi ``standardized`` không có file Markdown.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    standardized_files = (
        sorted(STANDARDIZED_DIR.rglob("*.md")) if STANDARDIZED_DIR.exists() else []
    )
    processed_files = (
        sorted(PROCESSED_DIR.rglob("*.md")) if PROCESSED_DIR.exists() else []
    )

    def canonical_stem(path: Path) -> str:
        return re.sub(r"[-_\s]+", "", path.stem.casefold())

    # Curated files in data/processed have clean headings and list structure.
    # Prefer them over duplicate raw PDF conversions in data/standardized, whose
    # table columns can be interleaved and separate a band number from its rubric.
    processed_stems = {canonical_stem(path) for path in processed_files}
    selected_files: list[tuple[Path, Path]] = [
        (path, STANDARDIZED_DIR)
        for path in standardized_files
        if canonical_stem(path) not in processed_stems
    ]
    selected_files.extend((path, PROCESSED_DIR) for path in processed_files)

    for md_file, source_dir in selected_files:
        content = md_file.read_text(encoding="utf-8")
        if not content.strip():
            continue

        relative_path = md_file.relative_to(source_dir)
        path_parts = {part.lower() for part in relative_path.parts}
        doc_type = "legal" if "legal" in path_parts else "news" if "news" in path_parts else "document"
        documents.append({
            "content": content,
            "metadata": {
                "source": relative_path.as_posix(),
                "type": doc_type,
            },
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict}
    """
    try:
        from langchain_text_splitters import (  # type: ignore
            MarkdownHeaderTextSplitter,
            RecursiveCharacterTextSplitter,
        )
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "heading_1"),
                ("##", "heading_2"),
                ("###", "heading_3"),
                ("####", "heading_4"),
            ],
            strip_headers=False,
        )
    except Exception:
        splitter = None
        markdown_splitter = None

    chunks = []
    for doc in documents:
        if splitter and markdown_splitter:
            sections = markdown_splitter.split_text(doc["content"])
            split_entries = []
            for section in sections:
                for text in splitter.split_text(section.page_content):
                    split_entries.append((text, section.metadata))
        else:
            text = doc["content"]
            raw_splits = [
                text[i:i+CHUNK_SIZE]
                for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP)
            ]
            split_entries = [(text, {}) for text in raw_splits]

        for i, (chunk_text, header_metadata) in enumerate(split_entries):
            if chunk_text.strip():
                chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        **doc["metadata"],
                        **header_metadata,
                        "chunk_index": i,
                    }
                })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    if not chunks:
        return []

    model = get_embedding_model()
    texts = [c["content"] for c in chunks]
    if model:
        embeddings = model.encode(texts, show_progress_bar=False)
        for chunk, emb in zip(chunks, embeddings):
            chunk["embedding"] = emb.tolist() if hasattr(emb, "tolist") else list(emb)
    else:
        # Offline deterministic fallback when sentence-transformers is unavailable.
        for chunk in chunks:
            chunk["embedding"] = fallback_embedding(chunk["content"])
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
    if not docs:
        raise RuntimeError(
            "No Markdown documents found. Run Task 3 first so that "
            "data/standardized/ contains .md files (or provide files in data/processed/)."
        )
    print(f"\n[OK] Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"[OK] Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"[OK] Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("[OK] Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
