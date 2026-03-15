import type { KnownConfig } from "./config.js";
import type { KnownDB } from "./db.js";
import { maximallyDistantClusters } from "./graph.js";
import { storeOrStrengthenInsight } from "./insights.js";
import { maintain } from "./maintain.js";
import { getOpenAIClient } from "./openai.js";
import { DISCOVER_SYSTEM, DISCOVER_USER } from "./prompts/discover.js";

interface DiscoverLLMResult {
  found?: boolean;
  insight?: string;
  supporting_node_ids?: string[];
}

interface DiscoverResult {
  found: boolean;
  insight?: string;
  strengthened?: boolean;
  maintenance: {
    nodesDecayed: number;
    nodesPruned: number;
    insightsPruned: number;
    nodesMerged: number;
  };
}

function parseDiscoverResult(content: string): DiscoverLLMResult {
  try {
    return JSON.parse(content) as DiscoverLLMResult;
  } catch {
    return { found: false };
  }
}

export async function discover(db: KnownDB, config: KnownConfig): Promise<DiscoverResult> {
  const runMaintenance = () => maintain(db, config);

  if (db.getStats().nodeCount < 50) {
    return {
      found: false,
      maintenance: runMaintenance(),
    };
  }

  const pair = maximallyDistantClusters(db, 5);
  if (!pair || pair.clusterA.length < 3 || pair.clusterB.length < 3) {
    return {
      found: false,
      maintenance: runMaintenance(),
    };
  }

  const openai = getOpenAIClient(config);
  const response = await openai.chat.completions.create({
    model: config.synthesisModel,
    messages: [
      { role: "system", content: DISCOVER_SYSTEM },
      {
        role: "user",
        content: DISCOVER_USER(
          pair.clusterA.map((node) => ({ id: node.id, type: node.type, text: node.text })),
          pair.clusterB.map((node) => ({ id: node.id, type: node.type, text: node.text })),
          pair.categoryA,
          pair.categoryB,
        ),
      },
    ],
    response_format: { type: "json_object" },
    temperature: 0.5,
  });

  const content = response.choices[0]?.message?.content;
  if (!content) {
    return {
      found: false,
      maintenance: runMaintenance(),
    };
  }

  const result = parseDiscoverResult(content);
  const insight = result.insight?.trim();
  if (!result.found || !insight) {
    return {
      found: false,
      maintenance: runMaintenance(),
    };
  }

  const supportingNodeIds =
    result.supporting_node_ids?.filter(
      (nodeId) => pair.clusterA.some((node) => node.id === nodeId) || pair.clusterB.some((node) => node.id === nodeId),
    ) ?? [...pair.clusterA.map((node) => node.id), ...pair.clusterB.map((node) => node.id)];

  const stored = await storeOrStrengthenInsight(db, config, insight, supportingNodeIds, 0.85);
  return {
    found: true,
    insight,
    strengthened: stored.strengthened,
    maintenance: runMaintenance(),
  };
}
