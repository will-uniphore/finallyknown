import type { KnownDB, NodeRow } from "./db.js";
import { bufferToVector, cosineSimilarity } from "./embeddings.js";

export interface ClusterPair {
  categoryA: string;
  categoryB: string;
  clusterA: NodeRow[];
  clusterB: NodeRow[];
  distance: number;
}

function averageEmbedding(nodes: NodeRow[]): Buffer | null {
  const vectors = nodes.filter((node) => node.embedding).map((node) => bufferToVector(node.embedding!));
  if (vectors.length === 0) {
    return null;
  }

  const dim = vectors[0].length;
  const avg = new Float32Array(dim);
  for (const vector of vectors) {
    for (let i = 0; i < dim; i += 1) {
      avg[i] += vector[i];
    }
  }

  for (let i = 0; i < dim; i += 1) {
    avg[i] /= vectors.length;
  }

  return Buffer.from(avg.buffer);
}

function getCategoryNodes(db: KnownDB, category: string, clusterSize: number): NodeRow[] {
  return db.getNodesByType(category).slice(0, clusterSize);
}

export function maximallyDistantClusters(db: KnownDB, clusterSize: number = 5): ClusterPair | null {
  const categories = db
    .getDistinctNodeTypes()
    .map((category) => ({
      category,
      nodes: getCategoryNodes(db, category, clusterSize),
    }))
    .filter((entry) => entry.nodes.length > 0)
    .map((entry) => ({
      ...entry,
      centroid: averageEmbedding(entry.nodes),
    }))
    .filter((entry): entry is { category: string; nodes: NodeRow[]; centroid: Buffer } => entry.centroid !== null);

  if (categories.length < 2) {
    return null;
  }

  let bestPair: ClusterPair | null = null;

  for (let i = 0; i < categories.length; i += 1) {
    for (let j = i + 1; j < categories.length; j += 1) {
      const left = categories[i];
      const right = categories[j];
      const similarity = cosineSimilarity(left.centroid, right.centroid);
      const distance = 1 - similarity;

      if (!bestPair || distance > bestPair.distance) {
        bestPair = {
          categoryA: left.category,
          categoryB: right.category,
          clusterA: left.nodes,
          clusterB: right.nodes,
          distance,
        };
      }
    }
  }

  return bestPair;
}

export function randomCluster(db: KnownDB, clusterSize: number = 5): NodeRow[] {
  const pair = maximallyDistantClusters(db, clusterSize);
  return pair?.clusterA ?? [];
}

export function distantCluster(db: KnownDB, _fromCluster: NodeRow[], clusterSize: number = 5): NodeRow[] {
  const pair = maximallyDistantClusters(db, clusterSize);
  return pair?.clusterB ?? [];
}
