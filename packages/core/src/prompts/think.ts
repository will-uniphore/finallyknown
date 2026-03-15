export const THINK_SYSTEM = `You are the conscious reasoning engine for "Known," a brain-like user understanding system.

You are given:
1. A question about the user
2. Retrieved observation nodes
3. Explicit edges that connect those observations
4. Previously discovered insights
5. Optional agent context

Your job is to think, not just retrieve. Reason over the observations and explicit links to form useful, defensible understanding.

Look for:
- Connections between observations that were not explicitly linked before
- Patterns that explain the user's current behavior
- Implications the user may not see themselves
- Structural similarities across different life domains
- How previously discovered insights change the answer right now

Rules:
- Ground every conclusion in the provided nodes, edges, and insights
- Do not restate the full context unless it matters to the answer
- Only emit genuinely new connections in \`new_connections\`
- If there are no good new connections, return an empty array

Return valid JSON:
{
  "response": "Specific, actionable synthesis for the agent.",
  "new_connections": [
    {
      "text": "A newly discovered connection.",
      "supporting_node_ids": ["node-id-1", "node-id-2"]
    }
  ]
}`;

export const THINK_USER = (
  question: string,
  nodes: { id: string; type: string; text: string; confidence: number; similarity: number }[],
  edges: { source_id: string; target_id: string; relation: string; text: string | null; confidence: number }[],
  insights: { id: string; text: string; confidence: number; times_rediscovered: number; times_used: number }[],
  agentContext?: string
) => {
  let prompt = `## Question\n${question}\n\n`;

  if (agentContext) {
    prompt += `## Agent Context\n${agentContext}\n\n`;
  }

  prompt += `## Relevant Observations (${nodes.length})\n`;
  for (const node of nodes) {
    prompt += `- [${node.id}] (${node.type}, confidence ${node.confidence.toFixed(2)}, similarity ${node.similarity.toFixed(3)}) ${node.text}\n`;
  }

  if (edges.length > 0) {
    prompt += `\n## Explicit Relationships (${edges.length})\n`;
    for (const edge of edges) {
      prompt += `- ${edge.source_id} -> ${edge.target_id} [${edge.relation}, confidence ${edge.confidence.toFixed(2)}]`;
      if (edge.text) {
        prompt += ` ${edge.text}`;
      }
      prompt += "\n";
    }
  }

  if (insights.length > 0) {
    prompt += `\n## Stored Insights (${insights.length})\n`;
    for (const insight of insights) {
      prompt += `- [${insight.id}] (confidence ${insight.confidence.toFixed(2)}, rediscovered ${insight.times_rediscovered}x, used ${insight.times_used}x) ${insight.text}\n`;
    }
  }

  return prompt;
};
