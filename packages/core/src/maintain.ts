import type { KnownConfig } from "./config.js";
import { computeObservationConfidence } from "./db.js";
import type { KnownDB } from "./db.js";

interface MaintainResult {
  nodesDecayed: number;
  nodesPruned: number;
  insightsPruned: number;
  nodesMerged: number;
}

export function maintain(db: KnownDB, _config: KnownConfig): MaintainResult {
  let nodesDecayed = 0;

  for (const node of db.getAllNodes()) {
    const nextConfidence = computeObservationConfidence(node.times_observed, node.updated_at);
    if (Math.abs(nextConfidence - node.confidence) > 0.001) {
      db.updateNodeConfidence(node.id, nextConfidence);
      nodesDecayed += 1;
    }
  }

  const nodesPruned = db.deleteNodesBelow(0.1);
  const insightsPruned = db.deleteDeadInsights();

  return {
    nodesDecayed,
    nodesPruned,
    insightsPruned,
    nodesMerged: 0,
  };
}
