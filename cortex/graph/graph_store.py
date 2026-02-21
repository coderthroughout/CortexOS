"""Neo4j graph store: User, Memory, Entity; relationships MENTIONS, CAUSES, etc."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID


class GraphStore:
    def __init__(self, uri: str = "", user: str = "", password: str = ""):
        self._uri = uri
        self._user = user
        self._password = password
        self._driver = None

    def _get_driver(self):
        if self._driver is None:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))
        return self._driver

    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None

    def ensure_user(self, user_id: UUID) -> str:
        """Create or return User node id."""
        with self._get_driver().session() as session:
            result = session.run(
                "MERGE (u:User {id: $id}) RETURN u.id AS id",
                id=str(user_id),
            )
            rec = result.single()
            return rec["id"] if rec else str(user_id)

    def ensure_entity(self, name: str) -> str:
        """Create or return Entity node (by name)."""
        with self._get_driver().session() as session:
            result = session.run(
                "MERGE (e:Entity {name: $name}) ON CREATE SET e.name = $name RETURN e.name AS name",
                name=name.strip(),
            )
            rec = result.single()
            return rec["name"] if rec else name.strip()

    def add_memory_node(self, memory_id: UUID, user_id: UUID, summary: str, memory_type: str) -> None:
        """Create Memory node and link to User."""
        with self._get_driver().session() as session:
            session.run(
                """
                MERGE (m:Memory {id: $mem_id})
                SET m.summary = $summary, m.type = $type
                WITH m
                MERGE (u:User {id: $user_id})
                MERGE (u)-[:EXPERIENCED]->(m)
                """,
                mem_id=str(memory_id),
                user_id=str(user_id),
                summary=summary[:1000],
                type=memory_type,
            )

    def link_memory_entities(self, memory_id: UUID, entities: List[str]) -> None:
        """Create MENTIONS edges from Memory to each Entity."""
        if not entities:
            return
        with self._get_driver().session() as session:
            for name in entities:
                if not name or not str(name).strip():
                    continue
                session.run(
                    """
                    MERGE (m:Memory {id: $mem_id})
                    MERGE (e:Entity {name: $name})
                    MERGE (m)-[:MENTIONS]->(e)
                    """,
                    mem_id=str(memory_id),
                    name=str(name).strip()[:500],
                )

    def link_relationship(self, from_entity: str, relation: str, to_entity: str) -> None:
        """Create RELATES_TO or CAUSES etc. between entities. relation: RELATES_TO, CAUSES, etc."""
        rel = relation.upper().replace(" ", "_") if relation else "RELATES_TO"
        if rel not in ("MENTIONS", "RELATES_TO", "CAUSES", "PART_OF", "SIMILAR_TO", "DERIVED_FROM", "EXPERIENCED"):
            rel = "RELATES_TO"
        with self._get_driver().session() as session:
            session.run(
                f"""
                MERGE (a:Entity {{name: $from_name}})
                MERGE (b:Entity {{name: $to_name}})
                MERGE (a)-[:{rel}]->(b)
                """,
                from_name=from_entity.strip()[:500],
                to_name=to_entity.strip()[:500],
            )

    def traverse(self, entity_names: List[str], depth: int = 2) -> List[str]:
        """
        BFS from given entity names; return memory ids reachable within depth.
        BFS from entity names, return memory ids within depth.
        """
        return self.get_memory_ids_near_entities(entity_names, depth)

    def get_memory_ids_near_entities(self, entity_names: List[str], depth: int = 2) -> List[str]:
        """Return memory ids connected to these entities within depth (BFS)."""
        if not entity_names:
            return []
        seen: set = set()
        with self._get_driver().session() as session:
            # Start from entities, follow MENTIONS to Memory, then MENTIONS to other entities, etc.
            result = session.run(
                """
                MATCH (e:Entity) WHERE e.name IN $names
                MATCH (m:Memory)-[:MENTIONS*1..2]-(e)
                RETURN DISTINCT m.id AS mem_id
                """,
                names=[n.strip() for n in entity_names if n],
            )
            for r in result:
                mid = r.get("mem_id")
                if mid and mid not in seen:
                    seen.add(mid)
            return list(seen)
