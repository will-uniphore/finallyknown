export const INGEST_SYSTEM = `You extract TRAIT CODES for "Known," a brain-like user understanding system.

Do not extract raw facts, tasks, or requests. Extract what the conversation REVEALS
about the user at a personality or pattern level.

Good targets:
- decision style
- communication style
- stress responses
- values revealed by tradeoffs
- recurring avoidance or overcompensation patterns
- aesthetic tendencies
- blind spots or self-undermining habits

Bad targets:
- one-off factual details
- project names
- transient asks
- generic summaries of the session

Return a compact JSON object:
{
  "nodes": [
    {
      "text": "avoids confrontation when stressed",
      "type": "stress_response"
    }
  ],
  "edges": []
}

Rules:
- Each node text must be a standalone trait code or behavioral pattern
- Prefer compressed, durable observations over literal paraphrases
- Provide a short free-form domain tag for each node (for example stress_response or aesthetic_style)
- Do not use a fixed category taxonomy
- Only include edges when the conversation itself clearly ties two extracted trait codes together
- Return valid JSON only`;

export const INGEST_USER = (sessionText: string) =>
  `Extract trait codes from this session.\n\n${sessionText}`;

export const CONTRADICTION_SYSTEM = `You compare two trait codes about the same user.

Return one label:
- "same": they express the same underlying pattern in different words
- "contradict": they point in meaningfully opposite directions
- "different": they are related or nearby, but not the same pattern and not a contradiction

Return valid JSON only:
{
  "relation": "same"
}`;

export const CONTRADICTION_USER = (existingTrait: string, newTrait: string) =>
  `Existing trait code: ${existingTrait}
New trait code: ${newTrait}

Do these describe the same trait, a contradiction, or different nearby patterns?`;
