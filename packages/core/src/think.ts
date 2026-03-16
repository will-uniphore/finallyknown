import type { KnownConfig } from "./config.js";
import type { EdgeRow, KnownDB, NodeRow } from "./db.js";
import { generateEmbedding, semanticSearch } from "./embeddings.js";
import { shouldSurfaceInsight, storeOrStrengthenInsight } from "./insights.js";
import { getOpenAIClient } from "./openai.js";
import { THINK_SYSTEM, THINK_USER } from "./prompts/think.js";

const ACTIVATION_DECAY_PER_HOP = 0.5;

interface ThinkLLMResult {
  response: string;
  new_connections?: Array<{
    text: string;
    supporting_node_ids?: string[];
  }>;
}

interface ThinkResult {
  response: string;
  insightsUsed: number;
  newInsights: number;
}

interface ActivatedNode {
  node: NodeRow;
  similarity: number;
  activation: number;
}

function formatEdges(edges: EdgeRow[]) {
  return edges.map((edge) => ({
    source_id: edge.source_id,
    target_id: edge.target_id,
    relation: edge.relation,
    text: edge.text,
    confidence: edge.confidence,
  }));
}

function parseThinkResult(content: string): ThinkLLMResult {
  try {
    const parsed = JSON.parse(content) as ThinkLLMResult;
    return {
      response: parsed.response ?? "",
      new_connections: Array.isArray(parsed.new_connections) ? parsed.new_connections : [],
    };
  } catch {
    return { response: content, new_connections: [] };
  }
}

function activateNodes(db: KnownDB, seeds: Array<NodeRow & { similarity: number }>): ActivatedNode[] {
  const activationMap = new Map<string, ActivatedNode>();

  for (const seed of seeds) {
    const seedActivation = seed.similarity * seed.confidence;
    activationMap.set(seed.id, {
      node: seed,
      similarity: seed.similarity,
      activation: seedActivation,
    });

    for (const edge of db.getEdgesForNode(seed.id)) {
      const neighborId = edge.source_id === seed.id ? edge.target_id : edge.source_id;
      const neighbor = db.getNode(neighborId);
      if (!neighbor || neighbor.confidence <= 0.1) {
        continue;
      }

      const spreadActivation = seedActivation * ACTIVATION_DECAY_PER_HOP * edge.confidence;
      const existing = activationMap.get(neighbor.id);
      activationMap.set(neighbor.id, {
        node: neighbor,
        similarity: existing?.similarity ?? 0,
        activation: (existing?.activation ?? 0) + spreadActivation,
      });
    }
  }

  return [...activationMap.values()].sort((left, right) => right.activation - left.activation);
}

export async function think(
  db: KnownDB,
  question: string,
  config: KnownConfig,
  agentContext?: string,
): Promise<ThinkResult> {
  const openai = getOpenAIClient(config);
  const queryEmbedding = await generateEmbedding(question, config);

  const allNodesWithEmb = db.getNodesWithEmbeddings();
  // If graph is small, use ALL nodes (no semantic filtering needed)
  const relevantNodes = allNodesWithEmb.length <= 30
    ? allNodesWithEmb.map(n => ({ ...n, similarity: 1.0 }))
    : semanticSearch(queryEmbedding, allNodesWithEmb, 20);
  const activatedNodes = activateNodes(db, relevantNodes).slice(0, 25);
  const surfacedInsights = semanticSearch(queryEmbedding, db.getInsightsWithEmbeddings(), 10)
    .filter((insight) => shouldSurfaceInsight(insight))
    .slice(0, 5);

  const nodeMap = new Map<string, ActivatedNode>();
  for (const entry of activatedNodes) {
    nodeMap.set(entry.node.id, entry);
  }

  const nodesForPrompt = [...nodeMap.values()].map(({ node, similarity, activation }) => ({
    id: node.id,
    type: node.type,
    text: node.text,
    confidence: node.confidence,
    similarity,
    activation,
    times_observed: node.times_observed,
  }));

  const edgesForPrompt = formatEdges(db.getEdgesBetweenNodes(nodesForPrompt.map((node) => node.id)));
  const insightsForPrompt = surfacedInsights.map((insight) => ({
    id: insight.id,
    text: insight.text,
    confidence: insight.confidence,
    times_rediscovered: insight.times_rediscovered,
    times_used: insight.times_used,
  }));

  const response = await openai.chat.completions.create({
    model: config.synthesisModel,
    messages: [
      { role: "system", content: THINK_SYSTEM },
      {
        role: "user",
        content: THINK_USER(question, nodesForPrompt, edgesForPrompt, insightsForPrompt, agentContext),
      },
    ],
    response_format: { type: "json_object" },
    temperature: 0.4,
  });

  const content = response.choices[0]?.message?.content;
  if (!content) {
    return { response: "Unable to reason about this question.", insightsUsed: 0, newInsights: 0 };
  }

  const result = parseThinkResult(content);

  for (const insight of surfacedInsights) {
    db.markInsightUsed(insight.id);
  }

  let newInsights = 0;
  for (const connection of result.new_connections ?? []) {
    const text = connection.text?.trim();
    if (!text) {
      continue;
    }

    const supportingNodeIds = connection.supporting_node_ids?.filter((nodeId) => nodeMap.has(nodeId)) ?? [];
    const stored = await storeOrStrengthenInsight(db, config, text, supportingNodeIds, 0.85);
    if (stored.created) {
      newInsights += 1;
    }
  }

  return {
    response: result.response || "Unable to reason about this question.",
    insightsUsed: surfacedInsights.length,
    newInsights,
  };
}
