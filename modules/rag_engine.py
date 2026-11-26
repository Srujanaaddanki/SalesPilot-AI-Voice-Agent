"""Local RAG engine: chunking, FAISS/TF-IDF vector indexing, and grounded retrieval."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import faiss  # type: ignore

    HAS_FAISS = True
except Exception:
    HAS_FAISS = False


DEFAULT_KNOWLEDGE_PATH = Path("data/company_knowledge.txt")


class RAGEngine:
    """Chunks company knowledge documents and retrieves relevant context via vector similarity."""

    def __init__(
        self,
        knowledge_path: Optional[str | Path] = None,
        min_score: float = 0.12,
    ) -> None:
        self.knowledge_path = Path(knowledge_path) if knowledge_path else DEFAULT_KNOWLEDGE_PATH
        self.min_score = min_score
        self.chunks: List[Dict[str, str]] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix: Optional[np.ndarray] = None
        self.faiss_index = None
        self._build_index()

    def _load_documents(self) -> str:
        if self.knowledge_path.exists():
            return self.knowledge_path.read_text(encoding="utf-8")
        return ""

    @staticmethod
    def chunk_text(text: str, max_chars: int = 750) -> List[Dict[str, str]]:
        """Split markdown/text into section-aware chunks while keeping headings with content."""
        if not text.strip():
            return []

        raw_blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
        chunks: List[Dict[str, str]] = []
        current_heading = "General"
        buffer = ""

        for block in raw_blocks:
            first_line = block.splitlines()[0].strip()
            if first_line.startswith("#"):
                if buffer:
                    chunks.append({"section": current_heading, "content": buffer})
                    buffer = ""
                current_heading = first_line.lstrip("# ").strip()

            if len(block) <= max_chars:
                if buffer and len(buffer) + len(block) + 2 <= max_chars:
                    buffer = f"{buffer}\n\n{block}"
                else:
                    if buffer:
                        chunks.append({"section": current_heading, "content": buffer})
                    buffer = block
            else:
                if buffer:
                    chunks.append({"section": current_heading, "content": buffer})
                    buffer = ""
                # Split oversized block by sentences/lines
                lines = block.splitlines()
                sub = ""
                for line in lines:
                    if len(sub) + len(line) + 1 <= max_chars:
                        sub = f"{sub}\n{line}".strip()
                    else:
                        if sub:
                            chunks.append({"section": current_heading, "content": sub})
                        sub = line
                if sub:
                    chunks.append({"section": current_heading, "content": sub})

        if buffer:
            chunks.append({"section": current_heading, "content": buffer})

        return chunks

    def _build_index(self) -> None:
        text = self._load_documents()
        self.chunks = self.chunk_text(text)
        if not self.chunks:
            return

        corpus = [f"{c['section']}\n{c['content']}" for c in self.chunks]
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        sparse = self.vectorizer.fit_transform(corpus)
        dense = sparse.astype(np.float32).toarray()

        # L2-normalize so inner product == cosine similarity
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.matrix = dense / norms

        if HAS_FAISS and self.matrix.shape[0] > 0:
            dim = self.matrix.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(self.matrix)
            self.faiss_index = index

    def reload(self) -> int:
        """Rebuild the index from disk and return number of chunks."""
        self.faiss_index = None
        self.matrix = None
        self.vectorizer = None
        self._build_index()
        return len(self.chunks)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, object]]:
        """Return top-k relevant chunks above min_score threshold."""
        if not query or not query.strip() or not self.chunks or self.vectorizer is None or self.matrix is None:
            return []

        q_vec = self.vectorizer.transform([query]).astype(np.float32).toarray()
        q_norm = np.linalg.norm(q_vec, axis=1, keepdims=True)
        if q_norm[0][0] == 0:
            # Fallback to simple keyword overlap if TF-IDF OOV
            return self._keyword_fallback(query, top_k=top_k)

        q_vec = q_vec / q_norm
        k = min(top_k, len(self.chunks))

        if self.faiss_index is not None:
            scores, indices = self.faiss_index.search(q_vec, k)
            raw_pairs: List[Tuple[int, float]] = [
                (int(idx), float(sc)) for idx, sc in zip(indices[0], scores[0]) if idx >= 0
            ]
        else:
            sims = (self.matrix @ q_vec.T).ravel()
            top_idx = np.argsort(-sims)[:k]
            raw_pairs = [(int(idx), float(sims[idx])) for idx in top_idx]

        # Keyword overlap boost for short domain queries, ignoring generic filler words
        generic_words = {"what", "which", "where", "when", "does", "offer", "provide", "have", "your", "with", "about", "tell"}
        q_terms = {
            t for t in re.findall(r"[a-zA-Z0-9]{3,}", query.lower())
            if t not in generic_words
        }
        results: List[Dict[str, object]] = []
        for idx, sim in raw_pairs:
            if sim < 0.09:
                continue
            chunk = self.chunks[idx]
            c_lower = f"{chunk['section']} {chunk['content']}".lower()
            overlap = sum(1 for t in q_terms if t in c_lower)
            boosted = sim + (0.03 * overlap)
            if boosted >= max(self.min_score, 0.16):
                results.append(
                    {
                        "chunk_id": idx,
                        "section": chunk["section"],
                        "content": chunk["content"],
                        "score": round(boosted, 3),
                    }
                )

        results.sort(key=lambda r: float(r["score"]), reverse=True)
        return results[:top_k]

    def _keyword_fallback(self, query: str, top_k: int = 3) -> List[Dict[str, object]]:
        q_terms = set(re.findall(r"[a-zA-Z0-9]{3,}", query.lower()))
        if not q_terms:
            return []
        scored: List[Dict[str, object]] = []
        for idx, chunk in enumerate(self.chunks):
            c_lower = f"{chunk['section']} {chunk['content']}".lower()
            hits = sum(1 for t in q_terms if t in c_lower)
            if hits > 0:
                scored.append(
                    {
                        "chunk_id": idx,
                        "section": chunk["section"],
                        "content": chunk["content"],
                        "score": round(hits / max(len(q_terms), 1), 3),
                    }
                )
        scored.sort(key=lambda r: float(r["score"]), reverse=True)
        return scored[:top_k]

    def get_context(self, query: str, top_k: int = 3) -> Tuple[str, List[Dict[str, object]]]:
        """Retrieve formatted knowledge context string and metadata list."""
        hits = self.retrieve(query, top_k=top_k)
        if not hits:
            return "", []
        formatted = "\n\n---\n\n".join(
            f"[{h['section']} | relevance={h['score']}]\n{h['content']}" for h in hits
        )
        return formatted, hits
