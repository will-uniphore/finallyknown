import type { KnownConfig } from "./config.js";
import type { KnownDB } from "./db.js";
import { randomCluster, distantCluster } from "./graph.js";
import { storeOrStrengthenInsight } from "./insights.js";
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
}

function parseDiscoverResult(content: string): DiscoverLLMResult {
  try {
    return JSON.parse(content) as DiscoverLLMResult;
  } catch {
    return { found: false };
  }
}

export async function discover(db: KnownDB, config: KnownConfig): Promise<DiscoverResult> {
  const clusterA = randomCluster(db, 5);
  if (clusterA.length === 0) {
    return { found: false };
  }

  const clusterB = distantCluster(db, clusterA, 5);
  if (clusterB.length === 0) {
    return { found: false };
  }

  const openai = getOpenAIClient(config);
  const response = await openai.chat.completions.create({
    model: config.synthesisModel,
    messages: [
      { role: "system", content: DISCOVER_SYSTEM },
      {
        role: "user",
        content: DISCOVER_USER(
          clusterA.map((node) => ({ id: node.id, type: node.type, text: node.text })),
          clusterB.map((node) => ({ id: node.id, type: node.type, text: node.text }))
        ),
      },
    ],
    response_format: { type: "json_object" },
    temperature: 0.9,
  });

  const content = response.choices[0]?.message?.content;
  if (!content) {
    return { found: false };
  }

  const result = parseDiscoverResult(content);
  const insight = result.insight?.trim();
  if (!result.found || !insight) {
    return { found: false };
  }

  const supportingNodeIds =
    result.supporting_node_ids?.filter((nodeId) => clusterA.some((node) => node.id === nodeId) || clusterB.some((node) => node.id === nodeId)) ??
    [...clusterA.map((node) => node.id), ...clusterB.map((node) => node.id)];

  const stored = await storeOrStrengthenInsight(db, config, insight, supportingNodeIds, 0.85);
  return {
    found: true,
    insight,
    strengthened: stored.strengthened,
  };
}
