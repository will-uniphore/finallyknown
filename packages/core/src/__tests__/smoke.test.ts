import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { rmSync, mkdtempSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import { getConfig } from "../config.js";
import { computeObservationConfidence, KnownDB } from "../db.js";
import { cosineSimilarity, semanticSearch } from "../embeddings.js";
import { ingest } from "../ingest.js";
import { shouldSurfaceInsight } from "../insights.js";
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

  it("re-confirms a trait code instead of duplicating it", () => {
    const first = db.upsertNodeObservation({
      type: "stress_response",
      text: "avoids confrontation when stressed",
      confidence: 1,
      source: "session-1",
      decay_rate: 0.01,
      times_observed: 1,
      embedding: null,
    });

    db.updateNodeConfidence(first.row.id, 0.4);

    const second = db.upsertNodeObservation({
      type: "stress_response",
      text: "avoids confrontation when stressed",
      confidence: 1,
      source: "session-2",
      decay_rate: 0.01,
      times_observed: 1,
      embedding: null,
    });

    const updated = db.getNode(first.row.id)!;
    expect(first.created).toBe(true);
    expect(second.created).toBe(false);
    expect(db.getStats().nodeCount).toBe(1);
    expect(updated.times_observed).toBe(2);
    expect(updated.confidence).toBe(1);
    expect(updated.source).toBe("session-2");
  });

  it("applies contradiction weakening to a stored trait code", () => {
    const oldTimestamp = new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString();
    const node = db.insertNode({
      type: "collaboration_style",
      text: "defaults to harmony over direct confrontation",
      confidence: 1,
      source: null,
      decay_rate: 0.01,
      times_observed: 4,
      embedding: null,
    });

    db.db.prepare("UPDATE nodes SET updated_at = ?, confidence = ? WHERE id = ?").run(
      oldTimestamp,
      computeObservationConfidence(4, oldTimestamp),
      node.id,
    );

    const before = db.getNode(node.id)!.confidence;
    db.applyContradictionPenalty(node.id, 0.7);
    const after = db.getNode(node.id)!;

    expect(after.confidence).toBeCloseTo(before * 0.7, 2);
    expect(after.times_observed).toBeLessThan(4);
  });

  it("supports one-hop graph traversal by default", () => {
    const n1 = db.insertNode({
      type: "work_style",
      text: "prefers sparse interfaces",
      confidence: 1,
      source: null,
      decay_rate: 0.01,
      times_observed: 1,
      embedding: null,
    });
    const n2 = db.insertNode({
      type: "aesthetic_style",
      text: "values negative space",
      confidence: 1,
      source: null,
      decay_rate: 0.01,
      times_observed: 1,
      embedding: null,
    });
    const n3 = db.insertNode({
      type: "communication_style",
      text: "keeps feedback minimal",
      confidence: 1,
      source: null,
      decay_rate: 0.01,
      times_observed: 1,
      embedding: null,
    });

    db.upsertEdge({ source_id: n1.id, target_id: n2.id, relation: "related", text: null, confidence: 1, source: null });
    db.upsertEdge({ source_id: n2.id, target_id: n3.id, relation: "related", text: null, confidence: 1, source: null });

    const hop1 = db.expandViaEdges([n1.id]);

    expect(hop1.map((node) => node.id)).toContain(n2.id);
    expect(hop1.map((node) => node.id)).not.toContain(n3.id);
  });

  it("strengthens and gates insights with the three-strike rule", () => {
    const insight = db.insertInsight({
      text: "minimalism shows up across work and aesthetics",
      supporting_nodes: JSON.stringify(["node-a", "node-b"]),
      confidence: 0.4,
      embedding: null,
    });

    expect(shouldSurfaceInsight(insight)).toBe(false);

    db.strengthenInsight(insight.id);
    expect(shouldSurfaceInsight(db.getInsight(insight.id)!)).toBe(false);

    db.strengthenInsight(insight.id);
    db.markInsightUsed(insight.id);

    const updated = db.getInsight(insight.id)!;
    expect(updated.times_rediscovered).toBe(2);
    expect(updated.times_used).toBe(1);
    expect(updated.last_used).toBeTruthy();
    expect(shouldSurfaceInsight(updated)).toBe(true);
  });

  it("removes dead nodes from insight evidence when pruning", () => {
    const alive = db.insertNode({
      type: "state",
      text: "stable",
      confidence: 0.5,
      source: null,
      decay_rate: 0.01,
      times_observed: 1,
      embedding: null,
    });
    const dead = db.insertNode({
      type: "state",
      text: "faded",
      confidence: 0.05,
      source: null,
      decay_rate: 0.01,
      times_observed: 1,
      embedding: null,
    });
    const insight = db.insertInsight({
      text: "Insight with mixed evidence",
      supporting_nodes: JSON.stringify([alive.id, dead.id]),
      confidence: 0.4,
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

  it("recomputes confidence from observation count and recency", () => {
    const updatedAt = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString();
    const node = db.insertNode({
      type: "stress_response",
      text: "over-prepares when uncertain",
      confidence: 1,
      source: null,
      decay_rate: 0.1,
      times_observed: 3,
      embedding: null,
    });

    db.db.prepare("UPDATE nodes SET updated_at = ?, confidence = ? WHERE id = ?").run(updatedAt, 1, node.id);

    const result = maintain(db, getConfig({ openaiApiKey: "test" }));
    const updated = db.getNode(node.id)!;
    const expected = computeObservationConfidence(3, updatedAt);

    expect(result.nodesDecayed).toBe(1);
    expect(updated.confidence).toBeCloseTo(expected, 5);
  });

  it("prunes weak nodes and does not merge them in dream maintenance", () => {
    const node = db.insertNode({
      type: "trait",
      text: "nearly forgotten pattern",
      confidence: 0.05,
      source: null,
      decay_rate: 0.01,
      times_observed: 0.1,
      embedding: null,
    });

    db.db
      .prepare("UPDATE nodes SET updated_at = ?, confidence = ? WHERE id = ?")
      .run(new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString(), 0.05, node.id);

    const result = maintain(db, getConfig({ openaiApiKey: "test" }));
    expect(result.nodesPruned).toBe(1);
    expect(result.nodesMerged).toBe(0);
  });
});

describe("ingest skips low-signal sessions", () => {
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

  it("skips short sessions before calling the model", async () => {
    const result = await ingest(db, "USER:\nNeed help with this query.", getConfig({ openaiApiKey: "" }), "session-1");
    expect(result).toEqual({ nodesCreated: 0, edgesCreated: 0 });
  });

  it("skips sessions with no user messages", async () => {
    const assistantOnly = `ASSISTANT:\n${"Long answer. ".repeat(80)}`;
    const result = await ingest(db, assistantOnly, getConfig({ openaiApiKey: "" }), "session-2");
    expect(result).toEqual({ nodesCreated: 0, edgesCreated: 0 });
  });
});
