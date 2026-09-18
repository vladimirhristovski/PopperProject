import uuid
from dataclasses import asdict

import chromadb
from sentence_transformers import SentenceTransformer

from popper_upgrade.skills.schema import Skill


class SkillStore:
    def __init__(self, persist_directory, embedding_model="all-MiniLM-L6-v2", dedup_threshold=0.95):
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(
            "popper_skills", metadata={"hnsw:space": "cosine"}
        )
        self._embedder = SentenceTransformer(embedding_model)
        self._dedup_threshold = dedup_threshold

    def _embed(self, text):
        return self._embedder.encode(text).tolist()

    def add(self, skill: Skill) -> bool:
        embedding = self._embed(skill.applicability)
        if self._collection.count() > 0:
            existing = self._collection.query(query_embeddings=[embedding], n_results=1)
            if existing["ids"][0]:
                similarity = 1 - existing["distances"][0][0]
                if similarity >= self._dedup_threshold:
                    return False
        if not skill.id:
            skill.id = str(uuid.uuid4())
        self._collection.add(
            ids=[skill.id],
            embeddings=[embedding],
            metadatas=[asdict(skill)],
            documents=[skill.applicability],
        )
        return True

    def retrieve(self, query: str, k: int = 3):
        count = self._collection.count()
        if count == 0:
            return []
        embedding = self._embed(query)
        results = self._collection.query(query_embeddings=[embedding], n_results=min(k, count))
        return [Skill(**metadata) for metadata in results["metadatas"][0]]

    def record_outcome(self, skill_id: str, passed: bool):
        existing = self._collection.get(ids=[skill_id])
        if not existing["ids"]:
            return
        metadata = existing["metadatas"][0]
        metadata["times_used"] += 1
        if passed:
            metadata["times_passed"] += 1
        self._collection.update(ids=[skill_id], metadatas=[metadata])
