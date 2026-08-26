from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import faiss
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class Knowledge:
    """A single piece of semantic knowledge in JARVIS's long-term memory."""

    knowledge_id: str
    subject: str
    predicate: str
    value: Any

    confidence: float = 0.5
    importance: float = 0.5

    source: Optional[str] = None

    created_at: float = 0.0
    updated_at: float = 0.0

    evidence_count: int = 1

    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

        now = time.time()
        if not self.created_at:
            self.created_at = now

        if not self.updated_at:
            self.updated_at = self.created_at

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Knowledge:
        tags = data.get("tags", [])
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except json.JSONDecodeError:
                tags = []

        value = data.get("value")
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                pass

        return cls(
            knowledge_id=data["knowledge_id"],
            subject=data["subject"],
            predicate=data["predicate"],
            value=value,
            confidence=float(data.get("confidence", 0.5)),
            importance=float(data.get("importance", 0.5)),
            source=data.get("source"),
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
            evidence_count=int(data.get("evidence_count", 1)),
            tags=tags,
        )


class SemanticMemory:
    """Long-term knowledge store of JARVIS.

    Combines:
    - Persistent SQLite Database for durable storage.
    - NetworkX Graph Engine for relationship tracking.
    - FAISS Vector Store using IndexIDMap2 for semantic similarity search.

    ----------------------------------------------------------------
    FIX LOG (this version)
    ----------------------------------------------------------------
    1. CRASH BUG: `self.embedder.get_embedding_dimension()` does not
       exist on SentenceTransformer — the real method is
       `get_sentence_embedding_dimension()`. This was almost
       certainly throwing an AttributeError on every construction,
       silently killing the whole semantic-memory organ depending on
       how the caller handled the exception.

    2. FAISS IDS NOW PERSISTED: previously, faiss_id was recomputed
       every process restart as a sequential counter over whatever
       order `SELECT * FROM knowledge` happened to return (SQLite
       does not guarantee row order without ORDER BY). If that order
       ever differed from the order the on-disk FAISS index was
       originally built in, vectors would silently map to the WRONG
       knowledge_id after a restart — silent data corruption with no
       error. faiss_id is now a real persisted column, assigned once
       per knowledge item and never recomputed.

    3. HYBRID SEARCH: added `hybrid_search()` which combines FAISS
       semantic similarity with the lexical LIKE search and merges/
       dedupes results (semantic first, lexical fills gaps). This is
       now what MemoryManager should call — semantic_search() alone
       stays available for callers that specifically want pure
       vector search.
    ----------------------------------------------------------------
    """

    VERSION = "0.3.0"

    def __init__(
        self,
        db_path: str = "jarvis_semantic_memory.db",
        faiss_index_path: str = "jarvis_faiss.index",
        max_knowledge: int = 10000,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.db_path = db_path
        self.faiss_index_path = faiss_index_path
        self.max_knowledge = max(1, int(max_knowledge))
        self._lock = threading.RLock()

        # 1. SQLite Database Setup
        self._init_sqlite_db()

        # 2. Embedding Model & FAISS Setup
        print("[SemanticMemory] Loading Embedding Model & FAISS Vector Store...")
        self.embedder = SentenceTransformer(model_name)
        
        if hasattr(self.embedder, "get_sentence_embedding_dimension"):
            self.vector_dim = self.embedder.get_sentence_embedding_dimension()
        elif hasattr(self.embedder, "get_embedding_dimension"):
            self.vector_dim = self.embedder.get_embedding_dimension()
        else:
            self.vector_dim = getattr(self.embedder, "vector_dim", 384)
        
        # FIX: correct SentenceTransformer API name.
        self.vector_dim = self.embedder.get_sentence_embedding_dimension()
        self.faiss_index = faiss.IndexIDMap2(faiss.IndexFlatL2(self.vector_dim))
        self.id_to_faiss_idx: Dict[str, int] = {}
        self.faiss_idx_to_id: Dict[int, str] = {}
        self._next_faiss_id = 1

        # 3. NetworkX Graph Setup
        self.graph = nx.DiGraph()

        # 4. Hydrate Graph & FAISS from DB
        self._hydrate_stores()

        self.created_at = time.time()
        self.updated_at = self.created_at

    # =============================================================
    # INITIALIZATION & HYDRATION HELPERS
    # =============================================================

    def _get_db_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_sqlite_db(self) -> None:
        with self._lock, self._get_db_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge (
                    knowledge_id TEXT PRIMARY KEY,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    value TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    importance REAL NOT NULL,
                    source TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    evidence_count INTEGER NOT NULL,
                    tags TEXT NOT NULL,
                    faiss_id INTEGER
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_subject ON knowledge(subject);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_predicate ON knowledge(predicate);")

            # Migration path for databases created before faiss_id existed.
            try:
                conn.execute("SELECT faiss_id FROM knowledge LIMIT 1;")
            except sqlite3.OperationalError:
                conn.execute("ALTER TABLE knowledge ADD COLUMN faiss_id INTEGER;")

            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_faiss_id ON knowledge(faiss_id) "
                "WHERE faiss_id IS NOT NULL;"
            )
            conn.commit()

    def _hydrate_stores(self) -> None:
        with self._lock, self._get_db_connection() as conn:
            cursor = conn.execute("SELECT * FROM knowledge")
            rows = [dict(row) for row in cursor.fetchall()]

            if not rows:
                return

            # Assign faiss_id to any legacy rows that don't have one yet
            # (pre-migration data), persisting it so it never shifts again.
            max_existing = 0
            for row in rows:
                if row.get("faiss_id") is not None:
                    max_existing = max(max_existing, int(row["faiss_id"]))

            next_id = max_existing + 1
            rows_needing_id = [r for r in rows if r.get("faiss_id") is None]
            if rows_needing_id:
                for row in rows_needing_id:
                    row["faiss_id"] = next_id
                    next_id += 1
                    conn.execute(
                        "UPDATE knowledge SET faiss_id = ? WHERE knowledge_id = ?",
                        (row["faiss_id"], row["knowledge_id"]),
                    )
                conn.commit()

            self._next_faiss_id = next_id

            texts_to_embed = []
            faiss_ids = []

            for row in rows:
                item = Knowledge.from_dict(row)
                self._add_to_graph(item)

                faiss_id = int(row["faiss_id"])
                self.id_to_faiss_idx[item.knowledge_id] = faiss_id
                self.faiss_idx_to_id[faiss_id] = item.knowledge_id

                text_to_embed = f"{item.subject} {item.predicate} {str(item.value)}"
                texts_to_embed.append(text_to_embed)
                faiss_ids.append(faiss_id)

            # Try to reuse the on-disk index (ids are now stable/persisted
            # so this is safe across restarts). If it's missing, unreadable,
            # or its id set doesn't match what's in the DB, rebuild from
            # scratch using the persisted faiss_ids.
            loaded_ok = False
            if os.path.exists(self.faiss_index_path):
                try:
                    candidate = faiss.read_index(self.faiss_index_path)
                    if candidate.ntotal == len(faiss_ids):
                        self.faiss_index = candidate
                        loaded_ok = True
                except Exception:
                    loaded_ok = False

            if not loaded_ok and texts_to_embed:
                self._rebuild_faiss_batch(texts_to_embed, faiss_ids)

    def _rebuild_faiss_batch(self, texts: List[str], ids: List[int]) -> None:
        vectors = self.embedder.encode(texts, show_progress_bar=False).astype(np.float32)
        self.faiss_index = faiss.IndexIDMap2(faiss.IndexFlatL2(self.vector_dim))
        self.faiss_index.add_with_ids(vectors, np.array(ids, dtype=np.int64))
        self._save_faiss_to_disk()

    # =============================================================
    # STORE / LEARN
    # =============================================================

    def remember(
        self,
        subject: str,
        predicate: str,
        value: Any,
        confidence: float = 0.5,
        importance: float = 0.5,
        source: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Knowledge:
        """Add new knowledge or reinforce existing knowledge."""
        subject = self._normalize(subject)
        predicate = self._normalize(predicate)

        if not subject:
            raise ValueError("subject cannot be empty.")
        if not predicate:
            raise ValueError("predicate cannot be empty.")

        with self._lock:
            existing = self._find_exact(subject, predicate, value)

            # -----------------------------------------------------
            # Reinforce existing knowledge
            # -----------------------------------------------------
            if existing is not None:
                existing.evidence_count += 1
                existing.confidence = self._merge_confidence(existing.confidence, confidence)
                existing.importance = max(existing.importance, self._clamp(importance))

                if source:
                    existing.source = source
                if tags:
                    existing.tags = list(set(existing.tags + tags))

                existing.updated_at = time.time()
                existing_faiss_id = self.id_to_faiss_idx.get(existing.knowledge_id)
                self._save_knowledge_to_db(existing, faiss_id=existing_faiss_id)
                self.updated_at = existing.updated_at
                return existing

            # -----------------------------------------------------
            # Create new knowledge
            # -----------------------------------------------------
            now = time.time()
            knowledge = Knowledge(
                knowledge_id=str(uuid.uuid4()),
                subject=subject,
                predicate=predicate,
                value=value,
                confidence=self._clamp(confidence),
                importance=self._clamp(importance),
                source=source,
                created_at=now,
                updated_at=now,
                evidence_count=1,
                tags=list(tags or []),
            )

            faiss_id = self._next_faiss_id
            self._next_faiss_id += 1

            self._save_knowledge_to_db(knowledge, faiss_id=faiss_id)
            self._add_to_graph(knowledge)

            text_to_embed = f"{knowledge.subject} {knowledge.predicate} {str(knowledge.value)}"
            vector = self.embedder.encode([text_to_embed])[0].astype(np.float32)

            self.faiss_index.add_with_ids(
                np.array([vector]), np.array([faiss_id], dtype=np.int64)
            )
            self.id_to_faiss_idx[knowledge.knowledge_id] = faiss_id
            self.faiss_idx_to_id[faiss_id] = knowledge.knowledge_id

            self.updated_at = now
            self._prune()
            self._save_faiss_to_disk()

            return knowledge

    def forget(self, knowledge_id: str) -> bool:
        """Remove knowledge item from DB, Graph, and FAISS."""
        with self._lock:
            item = self.get(knowledge_id)
            if not item:
                return False

            with self._get_db_connection() as conn:
                conn.execute("DELETE FROM knowledge WHERE knowledge_id = ?", (knowledge_id,))
                conn.commit()

            val_norm = self._normalize(item.value)
            if self.graph.has_edge(item.subject, val_norm):
                self.graph.remove_edge(item.subject, val_norm)

            if knowledge_id in self.id_to_faiss_idx:
                faiss_id = self.id_to_faiss_idx.pop(knowledge_id)
                self.faiss_idx_to_id.pop(faiss_id, None)
                self.faiss_index.remove_ids(np.array([faiss_id], dtype=np.int64))

            self.updated_at = time.time()
            self._save_faiss_to_disk()
            return True

    # =============================================================
    # SEARCH & RETRIEVAL METHODS
    # =============================================================

    def semantic_search(self, query: str, top_k: int = 5) -> List[Knowledge]:
        """Meaning-based search using FAISS vector embeddings."""
        with self._lock:
            if self.faiss_index.ntotal == 0:
                return []

            query_vector = self.embedder.encode([query])[0].astype(np.float32)
            k_search = min(top_k, self.faiss_index.ntotal)
            distances, indices = self.faiss_index.search(np.array([query_vector]), k_search)

            results = []
            for idx in indices[0]:
                idx = int(idx)
                if idx != -1 and idx in self.faiss_idx_to_id:
                    k_id = self.faiss_idx_to_id[idx]
                    item = self.get(k_id)
                    if item:
                        results.append(item)
            return results

    def hybrid_search(self, query: str, limit: int = 20) -> List[Knowledge]:
        """
        Robust retrieval: FAISS semantic similarity first (catches
        meaning/paraphrase matches the lexical search would miss),
        then lexical LIKE search fills in any exact keyword matches
        FAISS might rank low. Deduped by knowledge_id, semantic
        results kept first.

        This is what Brain/MemoryManager should call for
        `build_context()` / `search_knowledge()` — it's the method
        that was previously never being invoked.
        """
        semantic_results = self.semantic_search(query, top_k=limit)

        seen_ids = {item.knowledge_id for item in semantic_results}
        remaining = max(0, limit - len(semantic_results))

        lexical_results: List[Knowledge] = []
        if remaining > 0:
            for item in self.search(query, limit=limit):
                if item.knowledge_id not in seen_ids:
                    lexical_results.append(item)
                    seen_ids.add(item.knowledge_id)
                if len(lexical_results) >= remaining:
                    break

        return semantic_results + lexical_results

    def get_graph_relations(self, subject: str) -> List[Dict[str, Any]]:
        """Retrieve connection paths from NetworkX Graph."""
        subject = self._normalize(subject)
        with self._lock:
            if subject not in self.graph:
                return []

            relations = []
            for neighbor in self.graph.successors(subject):
                edge_data = self.graph.get_edge_data(subject, neighbor)
                relations.append({
                    "target": neighbor,
                    "predicate": edge_data.get("predicate", "related_to"),
                })
            return relations

    def find(
        self,
        subject: str,
        predicate: Optional[str] = None,
        value: Any = None,
    ) -> List[Knowledge]:
        """Find knowledge using subject/predicate/value wildcard filters."""
        subject = self._normalize(subject)
        query = "SELECT * FROM knowledge WHERE subject = ?"
        params: List[Any] = [subject]

        if predicate is not None:
            query += " AND predicate = ?"
            params.append(self._normalize(predicate))

        if value is not None:
            query += " AND value = ?"
            params.append(json.dumps(value) if not isinstance(value, str) else value)

        with self._lock, self._get_db_connection() as conn:
            cursor = conn.execute(query, params)
            return [Knowledge.from_dict(dict(row)) for row in cursor.fetchall()]

    def find_by_subject(self, subject: str, limit: int = 50) -> List[Knowledge]:
        results = self.find(subject)
        results.sort(
            key=lambda item: (item.importance, item.confidence, item.updated_at),
            reverse=True,
        )
        return results[:limit]

    def find_by_predicate(self, predicate: str, limit: int = 50) -> List[Knowledge]:
        predicate = self._normalize(predicate)
        with self._lock, self._get_db_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM knowledge WHERE predicate = ? ORDER BY updated_at DESC LIMIT ?",
                (predicate, limit),
            )
            return [Knowledge.from_dict(dict(row)) for row in cursor.fetchall()]

    def find_by_tag(self, tag: str, limit: int = 50) -> List[Knowledge]:
        tag_norm = f"%{self._normalize(tag)}%"
        with self._lock, self._get_db_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM knowledge WHERE tags LIKE ? ORDER BY updated_at DESC LIMIT ?",
                (tag_norm, limit),
            )
            return [Knowledge.from_dict(dict(row)) for row in cursor.fetchall()]

    def search(self, query: str, limit: int = 20) -> List[Knowledge]:
        """Lightweight lexical text search across all attributes."""
        query_norm = f"%{self._normalize(query)}%"
        sql = """
            SELECT * FROM knowledge
            WHERE subject LIKE ? OR predicate LIKE ? OR value LIKE ? OR tags LIKE ?
            ORDER BY importance DESC, confidence DESC, updated_at DESC
            LIMIT ?
        """
        with self._lock, self._get_db_connection() as conn:
            cursor = conn.execute(
                sql, (query_norm, query_norm, query_norm, query_norm, limit)
            )
            return [Knowledge.from_dict(dict(row)) for row in cursor.fetchall()]

    def get(self, knowledge_id: str) -> Optional[Knowledge]:
        with self._lock, self._get_db_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM knowledge WHERE knowledge_id = ?", (knowledge_id,)
            )
            row = cursor.fetchone()
            return Knowledge.from_dict(dict(row)) if row else None

    # =============================================================
    # CONFIDENCE & STATE MODIFIERS
    # =============================================================

    def update_confidence(self, knowledge_id: str, confidence: float) -> bool:
        with self._lock:
            item = self.get(knowledge_id)
            if item is None:
                return False

            item.confidence = self._clamp(confidence)
            item.updated_at = time.time()
            self._save_knowledge_to_db(item, faiss_id=self.id_to_faiss_idx.get(knowledge_id))
            self.updated_at = item.updated_at
            return True

    def reinforce(self, knowledge_id: str, confidence_delta: float = 0.05) -> bool:
        """Increase confidence when new evidence supports knowledge."""
        with self._lock:
            item = self.get(knowledge_id)
            if item is None:
                return False

            item.evidence_count += 1
            item.confidence = self._clamp(item.confidence + confidence_delta)
            item.updated_at = time.time()
            self._save_knowledge_to_db(item, faiss_id=self.id_to_faiss_idx.get(knowledge_id))
            self.updated_at = item.updated_at
            return True

    def weaken(self, knowledge_id: str, confidence_delta: float = 0.10) -> bool:
        """Reduce confidence when evidence contradicts knowledge."""
        with self._lock:
            item = self.get(knowledge_id)
            if item is None:
                return False

            item.confidence = self._clamp(item.confidence - abs(confidence_delta))
            item.updated_at = time.time()
            self._save_knowledge_to_db(item, faiss_id=self.id_to_faiss_idx.get(knowledge_id))
            self.updated_at = item.updated_at
            return True

    # =============================================================
    # COUNT, CLEAR & PRUNE
    # =============================================================

    @property
    def count(self) -> int:
        with self._lock, self._get_db_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM knowledge")
            return cursor.fetchone()[0]

    def clear(self) -> None:
        """Purge all database records, graphs, and vector indexes."""
        with self._lock:
            with self._get_db_connection() as conn:
                conn.execute("DELETE FROM knowledge")
                conn.commit()

            self.graph.clear()
            self.faiss_index = faiss.IndexIDMap2(faiss.IndexFlatL2(self.vector_dim))
            self.id_to_faiss_idx.clear()
            self.faiss_idx_to_id.clear()
            self._next_faiss_id = 1

            if os.path.exists(self.faiss_index_path):
                try:
                    os.remove(self.faiss_index_path)
                except OSError:
                    pass

            self.updated_at = time.time()

    def _prune(self) -> None:
        """Prune low-importance knowledge records when capacity limit is exceeded."""
        current_count = self.count
        if current_count <= self.max_knowledge:
            return

        excess = current_count - self.max_knowledge
        with self._lock, self._get_db_connection() as conn:
            cursor = conn.execute(
                """
                SELECT knowledge_id FROM knowledge
                ORDER BY importance ASC, confidence ASC, updated_at ASC
                LIMIT ?
                """,
                (excess,),
            )
            to_remove = [row["knowledge_id"] for row in cursor.fetchall()]

        for k_id in to_remove:
            self.forget(k_id)

    # =============================================================
    # SNAPSHOT & RESTORE
    # =============================================================

    def snapshot(self, limit: Optional[int] = None) -> Dict[str, Any]:
        with self._lock, self._get_db_connection() as conn:
            query = "SELECT * FROM knowledge ORDER BY updated_at DESC"
            if limit is not None:
                query += f" LIMIT {max(0, limit)}"

            cursor = conn.execute(query)
            items = [Knowledge.from_dict(dict(row)) for row in cursor.fetchall()]

            return {
                "version": self.VERSION,
                "count": len(items),
                "max_knowledge": self.max_knowledge,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "knowledge": [item.to_dict() for item in items],
            }

    def restore(self, snapshot: Dict[str, Any]) -> None:
        """Restore dataset state from a snapshot dictionary."""
        if not isinstance(snapshot, dict):
            return

        raw_items = snapshot.get("knowledge", [])
        if not isinstance(raw_items, list):
            return

        with self._lock:
            self.clear()
            for data in raw_items:
                if isinstance(data, dict):
                    try:
                        item = Knowledge.from_dict(data)
                        self.remember(
                            subject=item.subject,
                            predicate=item.predicate,
                            value=item.value,
                            confidence=item.confidence,
                            importance=item.importance,
                            source=item.source,
                            tags=item.tags,
                        )
                    except Exception:
                        continue

            self.updated_at = time.time()

    # =============================================================
    # INTERNAL UTILITIES
    # =============================================================

    def _save_knowledge_to_db(self, knowledge: Knowledge, faiss_id: Optional[int] = None) -> None:
        val_str = (
            json.dumps(knowledge.value)
            if not isinstance(knowledge.value, str)
            else knowledge.value
        )
        tags_str = json.dumps(knowledge.tags)

        with self._get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO knowledge (
                    knowledge_id, subject, predicate, value, confidence,
                    importance, source, created_at, updated_at, evidence_count, tags, faiss_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(knowledge_id) DO UPDATE SET
                    confidence=excluded.confidence,
                    importance=excluded.importance,
                    source=excluded.source,
                    updated_at=excluded.updated_at,
                    evidence_count=excluded.evidence_count,
                    tags=excluded.tags;
                """,
                (
                    knowledge.knowledge_id,
                    knowledge.subject,
                    knowledge.predicate,
                    val_str,
                    knowledge.confidence,
                    knowledge.importance,
                    knowledge.source,
                    knowledge.created_at,
                    knowledge.updated_at,
                    knowledge.evidence_count,
                    tags_str,
                    faiss_id,
                ),
            )
            conn.commit()

    def _add_to_graph(self, item: Knowledge) -> None:
        self.graph.add_node(item.subject, type="subject")
        if isinstance(item.value, str):
            val_str = self._normalize(item.value)
            self.graph.add_node(val_str, type="value")
            self.graph.add_edge(item.subject, val_str, predicate=item.predicate)

    def _save_faiss_to_disk(self) -> None:
        try:
            faiss.write_index(self.faiss_index, self.faiss_index_path)
        except Exception as e:
            print(f"[SemanticMemory] Error saving FAISS index to disk: {e}")

    def _find_exact(self, subject: str, predicate: str, value: Any) -> Optional[Knowledge]:
        items = self.find(subject, predicate)
        for item in items:
            if item.value == value:
                return item
        return None

    @staticmethod
    def _normalize(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip().lower()

    @staticmethod
    def _clamp(value: float) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _merge_confidence(old: float, new: float) -> float:
        old = max(0.0, min(1.0, old))
        new = max(0.0, min(1.0, new))
        return old + ((new - old) * 0.25)
