import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { join } from "path";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { getConfig } from "../config.js";
import { KnownDB } from "../db.js";
import { cosineSimilarity, semanticSearch } from "../embeddings.js";
import { maintain } from "../maintain.js";

describe("KnownDB", () => {
  let db: KnownDB;
  let tmpDir: string;

  beforeEach(() => {
    tmpDir = mkdtempSync(join(tmpdir(), "known-test-"));
    db = new KnownDB(join(tmpDir, "test.db"));
  });

  afterEach(() => {
    db.close();
    rmSync(tmpDir, { recursive: true, force: true });
  });

  it("creates tables", () => {
    const stats = db.getStats();
    expect(stats.nodeCount).toBe(0);
    expect(stats.edgeCount).toBe(0);
    expect(stats.insightCount).toBe(0);
  });

  it("re-confirms an observation instead of duplicating it", () => {
    const first = db.upsertNodeObservation({
      type: "person",
      text: "Will leads the frontend team",
      confidence: 1,
      source: "session-1",
      decay_rate: 0.01,
      embedding: null,
    });

    db.updateNodeConfidence(first.row.id, 0.4);

    const second = db.upsertNodeObservation({
      type: "person",
      text: "Will leads the frontend team",
      confidence: 1,
      source: "session-2",
      decay_rate: 0.01,
      embedding: null,
    });

    expect(first.created).toBe(true);
    expect(second.created).toBe(false);
    expect(db.getStats().nodeCount).toBe(1);
    expect(db.getNode(first.row.id)?.confidence).toBe(1);
    expect(db.getNode(first.row.id)?.source).toBe("session-2");
  });

  it("deduplicates explicit edges", () => {
    const will = db.insertNode({ type: "person", text: "Will", confidence: 1, source: null, decay_rate: 0.01, embedding: null });
    const forge = db.insertNode({ type: "project", text: "Forge UI", confidence: 1, source: null, decay_rate: 0.01, embedding: null });

    const first = db.upsertEdge({
      source_id: will.id,
      target_id: forge.id,
      relation: "part_of",
      text: "Will works on Forge UI",
      confidence: 1,
      source: "session-1",
    });

    const second = db.upsertEdge({
      source_id: will.id,
      target_id: forge.id,
      relation: "part_of",
      text: "Will works on Forge UI",
      confidence: 1,
      source: "session-2",
    });

    expect(first.created).toBe(true);
    expect(second.created).toBe(false);
    expect(db.getStats().edgeCount).toBe(1);
  });

  it("supports graph traversal across explicit edges", () => {
    const n1 = db.insertNode({ type: "person", text: "Will", confidence: 1, source: null, decay_rate: 0.01, embedding: null });
    const n2 = db.insertNode({ type: "project", text: "Forge UI", confidence: 1, source: null, decay_rate: 0.01, embedding: null });
    const n3 = db.insertNode({ type: "person", text: "Umesh", confidence: 1, source: null, decay_rate: 0.01, embedding: null });

    db.upsertEdge({ source_id: n1.id, target_id: n2.id, relation: "part_of", text: "Will works on Forge UI", confidence: 1, source: null });
    db.upsertEdge({ source_id: n3.id, target_id: n1.id, relation: "knows", text: "Umesh knows Will", confidence: 1, source: null });

    const hop1 = db.expandViaEdges([n2.id], 1);
    const hop2 = db.expandViaEdges([n2.id], 2);

    expect(hop1.map((node) => node.id)).toContain(n1.id);
    expect(hop2.map((node) => node.id)).toContain(n3.id);
  });

  it("strengthens and tracks insights", () => {
    const insight = db.insertInsight({
      text: "Will over-prepares when stressed",
      supporting_nodes: JSON.stringify(["node-a", "node-b"]),
      confidence: 0.7,
      embedding: null,
    });

    db.strengthenInsight(insight.id);
    db.addInsightSupport(insight.id, ["node-c", "node-b"]);
    db.markInsightUsed(insight.id);

    const updated = db.getInsight(insight.id)!;
    expect(updated.times_rediscovered).toBe(1);
    expect(updated.times_used).toBe(1);
    expect(updated.last_used).toBeTruthy();
    expect(JSON.parse(updated.supporting_nodes)).toEqual(["node-a", "node-b", "node-c"]);
  });

  it("removes dead nodes from insight evidence when pruning", () => {
    const alive = db.insertNode({ type: "state", text: "alive", confidence: 0.5, source: null, decay_rate: 0.01, embedding: null });
    const dead = db.insertNode({ type: "state", text: "dead", confidence: 0.05, source: null, decay_rate: 0.01, embedding: null });
    const insight = db.insertInsight({
      text: "Insight with mixed evidence",
      supporting_nodes: JSON.stringify([alive.id, dead.id]),
      confidence: 0.7,
      embedding: null,
    });

    const pruned = db.deleteNodesBelow(0.1);
    expect(pruned).toBe(1);

    const updated = db.getInsight(insight.id)!;
    expect(JSON.parse(updated.supporting_nodes)).toEqual([alive.id]);
  });
});

describe("embeddings", () => {
  it("computes cosine similarity", () => {
    const a = Buffer.from(new Float32Array([1, 0, 0]).buffer);
    const b = Buffer.from(new Float32Array([1, 0, 0]).buffer);
    const c = Buffer.from(new Float32Array([0, 1, 0]).buffer);

    expect(cosineSimilarity(a, b)).toBeCloseTo(1);
    expect(cosineSimilarity(a, c)).toBeCloseTo(0);
  });

  it("performs semantic search", () => {
    const query = Buffer.from(new Float32Array([1, 0, 0]).buffer);
    const items = [
      { id: "1", embedding: Buffer.from(new Float32Array([0.9, 0.1, 0]).buffer) },
      { id: "2", embedding: Buffer.from(new Float32Array([0, 1, 0]).buffer) },
      { id: "3", embedding: Buffer.from(new Float32Array([0.5, 0.5, 0]).buffer) },
    ];

    const results = semanticSearch(query, items, 2);
    expect(results).toHaveLength(2);
    expect(results[0].id).toBe("1");
  });
});

describe("maintain", () => {
  let db: KnownDB;
  let tmpDir: string;

  beforeEach(() => {
    tmpDir = mkdtempSync(join(tmpdir(), "known-test-"));
    db = new KnownDB(join(tmpDir, "test.db"));
  });

  afterEach(() => {
    db.close();
    rmSync(tmpDir, { recursive: true, force: true });
  });

  it("decays old nodes", () => {
    const node = db.insertNode({
      type: "state",
      text: "stressed about deadline",
      confidence: 1,
      source: null,
      decay_rate: 0.1,
      embedding: null,
    });

    db.db.prepare("UPDATE nodes SET updated_at = ? WHERE id = ?").run(
      new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString(),
      node.id
    );

    const result = maintain(db, getConfig({ openaiApiKey: "test" }));
    const updated = db.getNode(node.id)!;

    expect(result.nodesDecayed).toBe(1);
    expect(updated.confidence).toBeLessThan(0.7);
    expect(updated.confidence).toBeGreaterThan(0.5);
  });

  it("merges near-duplicate nodes and preserves insight evidence", () => {
    const emb1 = new Float32Array(8);
    const emb2 = new Float32Array(8);
    for (let i = 0; i < emb1.length; i += 1) {
      emb1[i] = 1;
      emb2[i] = 0.999;
    }

    const keep = db.insertNode({
      type: "preference",
      text: "prefers TypeScript",
      confidence: 0.8,
      source: null,
      decay_rate: 0.01,
      embedding: Buffer.from(emb1.buffer),
    });

    const remove = db.insertNode({
      type: "preference",
      text: "likes TypeScript",
      confidence: 0.6,
      source: null,
      decay_rate: 0.01,
      embedding: Buffer.from(emb2.buffer),
    });

    const project = db.insertNode({
      type: "project",
      text: "Known",
      confidence: 1,
      source: null,
      decay_rate: 0.01,
      embedding: Buffer.from(new Float32Array([0, 1, 0, 0, 0, 0, 0, 0]).buffer),
    });

    db.upsertEdge({
      source_id: remove.id,
      target_id: project.id,
      relation: "part_of",
      text: "likes TypeScript is part of Known",
      confidence: 1,
      source: null,
    });

    const insight = db.insertInsight({
      text: "TypeScript preference shapes project choices",
      supporting_nodes: JSON.stringify([remove.id, project.id]),
      confidence: 0.7,
      embedding: null,
    });

    const result = maintain(db, getConfig({ openaiApiKey: "test" }));
    const updatedInsight = db.getInsight(insight.id)!;

    expect(result.nodesMerged).toBe(1);
    expect(db.getStats().nodeCount).toBe(2);
    expect(db.getEdgesForNode(keep.id)).toHaveLength(1);
    expect(JSON.parse(updatedInsight.supporting_nodes)).toEqual([keep.id, project.id]);
  });
});
