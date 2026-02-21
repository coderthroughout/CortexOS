# Neo4j Graph Schema

## Node labels
- **User** – user id
- **Memory** – memory id, summary, type
- **Entity** – entity name (person, place, thing)
- **Concept** – abstract concept
- **Event** – event node (optional)

## Relationship types
- **MENTIONS** – (Memory)-[:MENTIONS]->(Entity)
- **RELATES_TO** – (Entity)-[:RELATES_TO]->(Entity|Concept)
- **CAUSES** – (Entity|Memory)-[:CAUSES]->(Entity|Memory)
- **PART_OF** – (Memory)-[:PART_OF]->(Event)
- **SIMILAR_TO** – (Memory)-[:SIMILAR_TO]->(Memory)
- **DERIVED_FROM** – (Memory)-[:DERIVED_FROM]->(Memory)  # semantic from episodic
- **EXPERIENCED** – (User)-[:EXPERIENCED]->(Memory)

## Example
```
(User {id: "uuid"})-[:EXPERIENCED]->(Memory {id: "mem_uuid", summary: "..."})
(Memory)-[:MENTIONS]->(Entity {name: "Anurag"})
(Memory)-[:MENTIONS]->(Entity {name: "Omium"})
(Entity {name: "Funding"})-[:CAUSES]->(Entity {name: "stress"})
```
