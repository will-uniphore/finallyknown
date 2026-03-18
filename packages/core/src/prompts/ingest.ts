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
- Provide a short free-form domain tag for each node
- Aim for roughly 8-12 strong pattern nodes per conversation chunk
- Do not include concrete personal facts here; those belong in the fact pass
- Only include edges when the conversation itself clearly ties two extracted trait codes together
- Return valid JSON only`;

export const INGEST_USER = (sessionText: string) =>
  `Extract pattern-level trait codes from this session.\n\n${sessionText}`;

export const INGEST_FACT_SYSTEM = `You extract SPECIFIC PERSONAL FACTS for "Known," a brain-like user understanding system.

Do not extract abstract personality patterns here. Extract the concrete facts, preferences,
history, and named interests that make this person uniquely themselves.

Good targets:
- specific hobbies and interests by name
- concrete preferences and dislikes
- health history, injuries, diagnoses, sensitivities
- significant life events and formative experiences
- specific skills, expertise areas, and repeated domains of competence
- relationships, family details, named people, and social roles
- places, routines, recurring activities, and favored environments

Bad targets:
- broad personality descriptions
- generic communication or decision patterns
- vague abstractions that could describe many people
- one-off task requests or transient scheduling details
- sensitive data such as phone numbers, addresses, passwords, SSNs, API keys

Specificity rules:
- "enjoys abstract modernist architecture" is good
- "likes design" is too abstract
- "prefers herbal tea from East Asia over coffee" is good
- "has refined taste" is too abstract
- "past knee injury from hiking steep terrain" is good
- "is health-conscious" is too abstract

Return a compact JSON object:
{
  "nodes": [
    {
      "text": "enjoys abstract modernist architecture",
      "type": "interest"
    },
    {
      "text": "past knee injury from hiking steep terrain",
      "type": "health_history"
    }
  ],
  "edges": []
}

Rules:
- Each node text must be specific, concrete, and unique to this person
- Prefer names, places, activities, events, and clear preferences over abstractions
- Aim for roughly 10-15 specific fact nodes per conversation chunk when supported
- Exclude generic patterns; those belong in the pattern pass
- Only include edges when the conversation clearly ties two extracted facts together
- Return valid JSON only`;

export const INGEST_FACT_USER = (sessionText: string) =>
  `Extract specific personal facts from this session.\n\n${sessionText}`;

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
