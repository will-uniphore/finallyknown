export const INGEST_SYSTEM = `You extract TWO types of observations for "Known," a brain-like user understanding system.

The human brain stores knowledge at TWO levels (Complementary Learning Systems):
1. TRAIT CODES (neocortex) — compressed personality patterns
2. PERSONAL FACTS (hippocampus) — specific details that define this person

Extract BOTH from this conversation.

=== TRAIT CODES ===
Personality-level patterns revealed by behavior:
- decision style, communication style, stress responses
- values revealed by tradeoffs
- behavioral patterns and blind spots
- aesthetic tendencies with domain specifics

=== PERSONAL FACTS ===
Specific details that make this person THEM:
- specific hobbies BY NAME ("forages for wild mushrooms", not "enjoys nature")
- health conditions, injuries, diagnoses
- significant life events and experiences
- concrete preferences ("prefers herbal tea from East Asia over coffee")
- skills, expertise areas
- relationships and family details mentioned
- places they've lived or frequent

=== BAD TARGETS (do NOT extract) ===
- one-off task requests
- transient scheduling details
- sensitive data (phone numbers, addresses, passwords)
- generic observations that could describe 80% of people

CRITICAL RULES:
- BE SPECIFIC. "forages for wild mushrooms in forests" >> "enjoys nature"
- "has a past knee injury from steep hiking terrain" >> "approaches health proactively"
- Every node should make this person DISTINGUISHABLE from others
- Extract 5-15 nodes per conversation chunk. More specific = better.

Return a compact JSON object:
{
  "nodes": [
    {
      "text": "forages for wild mushrooms in forests",
      "type": "hobby"
    },
    {
      "text": "avoids confrontation when stressed, defaults to over-preparation",
      "type": "stress_response"
    },
    {
      "text": "past knee injury from hiking steep terrain",
      "type": "health_history"
    }
  ],
  "edges": []
}

Rules:
- Mix of trait codes AND personal facts
- Each node text must be standalone and specific
- Provide a short free-form domain tag
- Only include edges when the conversation clearly ties two observations
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
