import type { KnownConfig } from "./config.js";
import type { KnownDB } from "./db.js";
import { cosineSimilarity } from "./embeddings.js";

interface MaintainResult {
  nodesDecayed: number;
  nodesPruned: number;
  insightsPruned: number;
  nodesMerged: number;
}

export function maintain(db: KnownDB, _config: KnownConfig): MaintainResult {
  const now = new Date();
  let nodesDecayed = 0;

  for (const node of db.getAllNodes()) {
    const updatedAt = new Date(node.updated_at);
    const daysSinceUpdate = (now.getTime() - updatedAt.getTime()) / (1000 * 60 * 60 * 24);
    if (daysSinceUpdate <= 0) {
      continue;
    }

    const nextConfidence = node.confidence * Math.pow(1 - node.decay_rate, daysSinceUpdate);
    if (Math.abs(nextConfidence - node.confidence) > 0.001) {
      db.updateNodeConfidence(node.id, nextConfidence);
      nodesDecayed += 1;
    }
  }

  const nodesPruned = db.deleteNodesBelow(0.1);
  const insightsPruned = db.deleteDeadInsights();

  const nodesWithEmbeddings = db.getNodesWithEmbeddings();
  const removedIds = new Set<string>();
  let nodesMerged = 0;

  for (let i = 0; i < nodesWithEmbeddings.length; i += 1) {
    const left = nodesWithEmbeddings[i];
    if (removedIds.has(left.id)) {
      continue;
    }

    for (let j = i + 1; j < nodesWithEmbeddings.length; j += 1) {
      const right = nodesWithEmbeddings[j];
      if (removedIds.has(right.id)) {
        continue;
      }

      const similarity = cosineSimilarity(left.embedding!, right.embedding!);
      if (similarity <= 0.95) {
        continue;
      }

      const keep = left.confidence >= right.confidence ? left : right;
      const remove = keep.id === left.id ? right : left;

      db.updateNodeConfidence(keep.id, Math.min(1.0, keep.confidence + 0.05));
      db.touchNode(keep.id);
      db.mergeNodeInto(keep.id, remove.id);

      removedIds.add(remove.id);
      nodesMerged += 1;
    }
  }

  return { nodesDecayed, nodesPruned, insightsPruned, nodesMerged };
}
