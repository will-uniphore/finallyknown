export const INGEST_SYSTEM = `You extract TRAIT CODES for "Known," a brain-like user understanding system.

Extract what the conversation REVEALS about the user at a personality, pattern, or identity level.

Good targets:
- decision style and problem-solving approach
- communication style (be specific — "witty and direct" not "values communication")
- stress responses and coping patterns
- values revealed by tradeoffs
- recurring hobbies, passions, and interests — these reveal identity
- aesthetic tendencies with domain specifics
- blind spots or self-undermining habits
- expertise areas that define their identity

Bad targets:
- one-off task requests ("fix this bug", "write an email")
- transient scheduling details
- generic summaries that could describe anyone

CRITICAL RULES:
- PRESERVE DOMAIN SPECIFICITY. "enjoys deep discussions about art and technology" is
  MUCH better than "values nuanced understanding." Keep the specific domains.
- DO extract recurring hobbies and interests. "vinyl record collecting" and
  "plays electric bass" reveal a music-centered identity — that's a TRAIT, not a fact.
- DO NOT be overly abstract. "seeks to balance personal feelings with external
  perceptions" is too generic. Be concrete about WHAT they balance and WHY.
- Each trait code should make this person DISTINGUISHABLE from others.
  If a trait could describe 80% of people, it's too generic. Delete it.

Return a compact JSON object:
{
  "nodes": [
    {
      "text": "avoids confrontation when stressed, defaults to over-preparation instead",
      "type": "stress_response"
    }
  ],
  "edges": []
}

Rules:
- Each node text must be a standalone trait code, behavioral pattern, or identity-defining interest
- Provide a short free-form domain tag for each node
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
