export const DISCOVER_SYSTEM = `You are the subconscious reasoning process for "Known," a brain-like user understanding system.

You are given two semantically distant clusters of observations about the same person.
Look for a genuine cross-domain resonance:
- a deep structural similarity
- a causal pattern spanning domains
- the same behavior manifesting in two different contexts
- a non-obvious link that would feel like an aha moment

Rules:
- Only respond with an insight if it is genuinely non-obvious and defensible
- If there is no real connection, return {"found": false}
- Keep the insight concise and useful

Return valid JSON:
{
  "found": true,
  "insight": "The cross-domain connection you found",
  "supporting_node_ids": ["node-id-1", "node-id-2"]
}

or

{
  "found": false
}`;

export const DISCOVER_USER = (
  clusterA: { id: string; type: string; text: string }[],
  clusterB: { id: string; type: string; text: string }[]
) => {
  let prompt = `## Cluster A\n`;
  for (const node of clusterA) {
    prompt += `- [${node.id}] (${node.type}) ${node.text}\n`;
  }

  prompt += `\n## Cluster B\n`;
  for (const node of clusterB) {
    prompt += `- [${node.id}] (${node.type}) ${node.text}\n`;
  }

  return prompt;
};
