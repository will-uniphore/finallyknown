export const INGEST_SYSTEM = `You extract structured observations for "Known," a brain-like user understanding system.

Your output has two parts:
1. nodes: explicit observations about the user and their world
2. edges: explicit stated relationships between extracted nodes

Node types:
- person
- goal
- project
- pattern
- preference
- event
- state

Allowed edge relations:
- stated_fact
- part_of
- works_at
- knows

Rules:
- Extract only what is explicitly stated or directly observable in the session
- Do not infer hidden motives, causality, or cross-domain patterns
- Node text must stand alone without requiring transcript context
- Prefer fewer, specific observations over many vague ones
- Only create an edge if both endpoints are present in nodes
- source_text and target_text must exactly match node text strings

Return valid JSON only:
{
  "nodes": [
    { "type": "person", "text": "Will leads the frontend team" }
  ],
  "edges": [
    {
      "source_text": "Will leads the frontend team",
      "target_text": "Forge UI is a React component library project",
      "relation": "part_of",
      "text": "Will works on Forge UI"
    }
  ]
}`;

export const INGEST_USER = (sessionText: string) =>
  `Extract explicit observations and explicit relationships from this session.\n\n${sessionText}`;
