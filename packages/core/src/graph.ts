import type { KnownDB, NodeRow } from "./db.js";
import { cosineSimilarity, bufferToVector } from "./embeddings.js";

/**
 * Pick a random cluster of nodes. Returns a set of related nodes
 * by picking a random node and expanding 1 hop.
 */
export function randomCluster(db: KnownDB, clusterSize: number = 5): NodeRow[] {
  const nodes = db.getNodesWithEmbeddings();
  if (nodes.length === 0) return [];

  const seed = nodes[Math.floor(Math.random() * nodes.length)];
  const expanded = db.expandViaEdges([seed.id], 1);

  if (expanded.length >= clusterSize) {
    return expanded.slice(0, clusterSize);
  }

  // If not enough via edges, fill with semantically similar nodes
  if (seed.embedding) {
    const similar = nodes
      .filter((n) => n.id !== seed.id && n.embedding)
      .map((n) => ({ node: n, sim: cosineSimilarity(seed.embedding!, n.embedding!) }))
      .sort((a, b) => b.sim - a.sim)
      .slice(0, clusterSize - expanded.length);
    const expandedIds = new Set(expanded.map((n) => n.id));
    for (const { node } of similar) {
      if (!expandedIds.has(node.id)) {
        expanded.push(node);
      }
    }
  }

  return expanded.slice(0, clusterSize);
}

/**
 * Pick a cluster that is semantically DISTANT from the given cluster.
 */
export function distantCluster(
  db: KnownDB,
  fromCluster: NodeRow[],
  clusterSize: number = 5
): NodeRow[] {
  const nodes = db.getNodesWithEmbeddings();
  if (nodes.length === 0) return [];

  const fromIds = new Set(fromCluster.map((n) => n.id));
  const candidates = nodes.filter((n) => !fromIds.has(n.id) && n.embedding);

  if (candidates.length === 0) return [];

  // Compute average embedding of fromCluster
  const fromEmbeddings = fromCluster.filter((n) => n.embedding).map((n) => bufferToVector(n.embedding!));
  if (fromEmbeddings.length === 0) return candidates.slice(0, clusterSize);

  const dim = fromEmbeddings[0].length;
  const avg = new Float32Array(dim);
  for (const emb of fromEmbeddings) {
    for (let i = 0; i < dim; i++) avg[i] += emb[i];
  }
  for (let i = 0; i < dim; i++) avg[i] /= fromEmbeddings.length;
  const avgBuf = Buffer.from(avg.buffer);

  // Sort by LOWEST similarity (most distant)
  const ranked = candidates
    .map((n) => ({ node: n, sim: cosineSimilarity(avgBuf, n.embedding!) }))
    .sort((a, b) => a.sim - b.sim);

  // Pick the most distant node, then expand
  const seed = ranked[0].node;
  const expanded = db.expandViaEdges([seed.id], 1);
  const result = expanded.filter((n) => !fromIds.has(n.id));

  if (result.length >= clusterSize) return result.slice(0, clusterSize);

  // Fill with other distant nodes
  const resultIds = new Set(result.map((n) => n.id));
  for (const { node } of ranked) {
    if (!resultIds.has(node.id) && !fromIds.has(node.id)) {
      result.push(node);
      if (result.length >= clusterSize) break;
    }
  }

  return result.slice(0, clusterSize);
}
