import type { KnownConfig } from "./config.js";
import type { EdgeRow, KnownDB } from "./db.js";
import { generateEmbedding, semanticSearch } from "./embeddings.js";
import { storeOrStrengthenInsight } from "./insights.js";
import { getOpenAIClient } from "./openai.js";
import { THINK_SYSTEM, THINK_USER } from "./prompts/think.js";

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

export async function think(
  db: KnownDB,
  question: string,
  config: KnownConfig,
  agentContext?: string
): Promise<ThinkResult> {
  const openai = getOpenAIClient(config);
  const queryEmbedding = await generateEmbedding(question, config);

  const relevantNodes = semanticSearch(queryEmbedding, db.getNodesWithEmbeddings(), 20);
  const relevantInsights = semanticSearch(queryEmbedding, db.getInsightsWithEmbeddings(), 5);
  const expandedNodes = db.expandViaEdges(
    relevantNodes.map((node) => node.id),
    2
  );

  const nodeMap = new Map<string, (typeof relevantNodes)[number]>();
  for (const node of relevantNodes) {
    nodeMap.set(node.id, node);
  }
  for (const node of expandedNodes) {
    if (!nodeMap.has(node.id)) {
      nodeMap.set(node.id, { ...node, similarity: 0 });
    }
  }

  const nodesForPrompt = [...nodeMap.values()].map((node) => ({
    id: node.id,
    type: node.type,
    text: node.text,
    confidence: node.confidence,
    similarity: node.similarity,
  }));

  const edgesForPrompt = formatEdges(db.getEdgesBetweenNodes(nodesForPrompt.map((node) => node.id)));
  const insightsForPrompt = relevantInsights.map((insight) => ({
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
    temperature: 0.7,
  });

  const content = response.choices[0]?.message?.content;
  if (!content) {
    return { response: "Unable to reason about this question.", insightsUsed: 0, newInsights: 0 };
  }

  const result = parseThinkResult(content);

  for (const insight of relevantInsights) {
    db.markInsightUsed(insight.id);
  }

  let newInsights = 0;
  for (const connection of result.new_connections ?? []) {
    const text = connection.text?.trim();
    if (!text) {
      continue;
    }

    const supportingNodeIds = connection.supporting_node_ids?.filter((nodeId) => nodeMap.has(nodeId)) ?? [];
    const stored = await storeOrStrengthenInsight(db, config, text, supportingNodeIds, 0.9);
    if (stored.created) {
      newInsights += 1;
    }
  }

  return {
    response: result.response || "Unable to reason about this question.",
    insightsUsed: relevantInsights.length,
    newInsights,
  };
}
