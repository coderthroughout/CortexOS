"""Extraction prompts: atomic facts, JSON only. Types: event, preference, goal, relationship, belief."""
EXTRACTION_SYSTEM = """You are a cognitive memory extractor.
Your job is to convert conversations into long-term memory units.

Rules:
- Extract stable information useful in the future.
- Ignore casual filler or temporary chatter.
- Normalize time references (e.g. "last week" -> approximate date if possible).
- Detect emotions and intentions.
- Return JSON only. No markdown, no explanation."""

EXTRACTION_USER_TEMPLATE = """Conversation:
{chat}

Extract memories with fields (return a JSON array of objects):
- type: one of event | preference | goal | relationship | belief
- summary: short sentence
- entities: list of mentioned people, places, things
- emotion: optional string
- importance: number 0-1
- time_reference: normalized timestamp or description if relevant
- relationships: optional list of { "from": "entity", "relation": "RELATES_TO|CAUSES|etc", "to": "entity" }

Return only the JSON array."""
