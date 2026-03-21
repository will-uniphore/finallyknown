import type { KnownConfig } from "./config.js";
import type { InsightRow, KnownDB } from "./db.js";
import { generateEmbedding, semanticSearch } from "./embeddings.js";

export type Insight = InsightRow;

export interface StoreOrStrengthenInsightResult {
  created: boolean;
  strengthened: boolean;
  insightId: string;
}

export function shouldSurfaceInsight(insight: Pick<InsightRow, "times_rediscovered" | "confidence">): boolean {
  return insight.times_rediscovered >= 2 && insight.confidence >= 0.6;
}

export function getInitiationCandidate(db: KnownDB): Insight | null {
  return (
    db
      .getAllInsights()
      .filter(
        (insight) =>
          insight.confidence >= 0.7 &&
          insight.times_rediscovered >= 2 &&
          insight.times_used === 0 &&
          !insight.initiated_at,
      )
      .sort((left, right) => {
        if (right.confidence !== left.confidence) {
          return right.confidence - left.confidence;
        }
        if (right.times_rediscovered !== left.times_rediscovered) {
          return right.times_rediscovered - left.times_rediscovered;
        }
        return left.discovered_at.localeCompare(right.discovered_at);
      })[0] ?? null
  );
}

export async function storeOrStrengthenInsight(
  db: KnownDB,
  config: KnownConfig,
  text: string,
  supportingNodeIds: string[],
  similarityThreshold: number = 0.9
): Promise<StoreOrStrengthenInsightResult> {
  const embedding = await generateEmbedding(text, config);
  const similar = semanticSearch(embedding, db.getInsightsWithEmbeddings(), 1);

  if (similar.length > 0 && similar[0].similarity >= similarityThreshold) {
    db.strengthenInsight(similar[0].id);
    db.addInsightSupport(similar[0].id, supportingNodeIds);
    return { created: false, strengthened: true, insightId: similar[0].id };
  }

  const row = db.insertInsight({
    text,
    supporting_nodes: JSON.stringify([...new Set(supportingNodeIds)]),
    confidence: 0.4,
    embedding,
  });

  return { created: true, strengthened: false, insightId: row.id };
}
