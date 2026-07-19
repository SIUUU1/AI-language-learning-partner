"""
vectorstore.py — ChromaDB 로 "이미 배운 표현" 의미 기반 관리
"""
from __future__ import annotations

import hashlib
from typing import List

from .config import CHROMA_DIR, USE_REAL_LLM, OPENAI_API_KEY

_DIM = 256


def _hash_embed(text: str) -> List[float]:
    """네트워크 불필요한 결정적 임베딩 (bag-of-ngram 해싱)."""
    vec = [0.0] * _DIM
    toks = text.lower().split()
    grams = toks + [a + " " + b for a, b in zip(toks, toks[1:])]
    for g in grams:
        h = int(hashlib.md5(g.encode()).hexdigest(), 16)
        vec[h % _DIM] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


try:
    from chromadb import EmbeddingFunction  # type: ignore

    class _LocalEmbeddingFunction(EmbeddingFunction):
        def __call__(self, input):  # noqa: A002  (chroma 인터페이스 시그니처)
            return [_hash_embed(t) for t in input]

        def name(self):  # chroma 1.x 가 요구
            return "lingualoop-local-hash"
except Exception:  # pragma: no cover
    _LocalEmbeddingFunction = None  # type: ignore


class ExpressionMemory:
    """사용자별 학습 표현 벡터 저장소."""

    def __init__(self):
        self.enabled = False
        self._col = None
        try:
            import chromadb
            self._client = chromadb.PersistentClient(path=CHROMA_DIR)
            ef = None
            if USE_REAL_LLM and OPENAI_API_KEY:
                try:
                    from chromadb.utils import embedding_functions
                    ef = embedding_functions.OpenAIEmbeddingFunction(
                        api_key=OPENAI_API_KEY, model_name="text-embedding-3-small")
                except Exception:
                    ef = _LocalEmbeddingFunction() if _LocalEmbeddingFunction else None
            else:
                ef = _LocalEmbeddingFunction() if _LocalEmbeddingFunction else None
            self._col = self._client.get_or_create_collection(
                name="learned_expressions", embedding_function=ef)
            self.enabled = True
        except Exception as e:  # pragma: no cover
            print(f"[chroma disabled] {e}")

    def novelty(self, user_id: str, expression: str, threshold: float = 0.25) -> bool:
        """이 표현이 사용자에게 '새로운' 표현인지 판정 (가장 가까운 이웃 거리 기준)."""
        if not self.enabled:
            return True
        try:
            res = self._col.query(query_texts=[expression], n_results=1,
                                  where={"user_id": user_id})
            dists = (res.get("distances") or [[]])[0]
            if not dists:
                return True
            return dists[0] > threshold  # 충분히 멀면 새 표현
        except Exception:  # pragma: no cover
            return True

    def add(self, user_id: str, expression: str, meaning: str = "") -> None:
        if not self.enabled:
            return
        try:
            uid = hashlib.md5(f"{user_id}:{expression}".encode()).hexdigest()
            self._col.upsert(
                ids=[uid], documents=[expression],
                metadatas=[{"user_id": user_id, "meaning": meaning}])
        except Exception:  # pragma: no cover
            pass

    def count(self, user_id: str) -> int:
        if not self.enabled:
            return 0
        try:
            return len(self._col.get(where={"user_id": user_id}).get("ids", []))
        except Exception:  # pragma: no cover
            return 0


# 프로세스 전역 싱글턴
_memory: ExpressionMemory | None = None


def get_memory() -> ExpressionMemory:
    global _memory
    if _memory is None:
        _memory = ExpressionMemory()
    return _memory
