import Database from "better-sqlite3";
import { mkdirSync } from "fs";
import { nanoid } from "nanoid";
import { dirname } from "path";

const DAY_MS = 1000 * 60 * 60 * 24;

export interface NodeRow {
  id: string;
  type: string;
  text: string;
  confidence: number;
  source: string | null;
  created_at: string;
  updated_at: string;
  decay_rate: number;
  times_observed: number;
  embedding: Buffer | null;
}

export interface EdgeRow {
  id: string;
  source_id: string;
  target_id: string;
  relation: string;
  text: string | null;
  confidence: number;
  source: string | null;
  created_at: string;
}

export interface InsightRow {
  id: string;
  text: string;
  supporting_nodes: string;
  confidence: number;
  discovered_at: string;
  times_rediscovered: number;
  times_used: number;
  last_used: string | null;
  initiated_at: string | null;
  embedding: Buffer | null;
}

const SCHEMA = `
CREATE TABLE IF NOT EXISTS nodes (
    id              TEXT PRIMARY KEY,
    type            TEXT NOT NULL,
    text            TEXT NOT NULL,
    confidence      REAL DEFAULT 1.0,
    source          TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    decay_rate      REAL DEFAULT 0.01,
    times_observed  REAL DEFAULT 1,
    embedding       BLOB
);

CREATE TABLE IF NOT EXISTS edges (
    id          TEXT PRIMARY KEY,
    source_id   TEXT NOT NULL REFERENCES nodes(id),
    target_id   TEXT NOT NULL REFERENCES nodes(id),
    relation    TEXT NOT NULL,
    text        TEXT,
    confidence  REAL DEFAULT 1.0,
    source      TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS insights (
    id                  TEXT PRIMARY KEY,
    text                TEXT NOT NULL,
    supporting_nodes    TEXT NOT NULL,
    confidence          REAL DEFAULT 0.4,
    discovered_at       TEXT NOT NULL,
    times_rediscovered  INTEGER DEFAULT 0,
    times_used          INTEGER DEFAULT 0,
    last_used           TEXT,
    initiated_at        TEXT,
    embedding           BLOB
);

CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type);
CREATE INDEX IF NOT EXISTS idx_nodes_confidence ON nodes(confidence DESC);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
CREATE INDEX IF NOT EXISTS idx_insights_confidence ON insights(confidence DESC);
`;

function nowIso() {
  return new Date().toISOString();
}

export function daysSince(timestamp: string, now: Date = new Date()): number {
  const parsed = new Date(timestamp);
  if (Number.isNaN(parsed.getTime())) {
    return 0;
  }

  return Math.max(0, (now.getTime() - parsed.getTime()) / DAY_MS);
}

export function computeObservationConfidence(timesObserved: number, lastSeenAt: string, now: Date = new Date()): number {
  const observed = Math.max(timesObserved, 0.01);
  const elapsedDays = daysSince(lastSeenAt, now);
  return Math.max(0, Math.min(1, observed / (observed + elapsedDays)));
}

function observationCountForConfidence(confidence: number, elapsedDays: number): number {
  const clampedConfidence = Math.max(0.01, Math.min(0.99, confidence));
  const safeDays = Math.max(elapsedDays, 1);
  return (clampedConfidence * safeDays) / (1 - clampedConfidence);
}

function parseSupportingNodes(raw: string): string[] {
  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed.filter((value): value is string => typeof value === "string");
  } catch {
    return [];
  }
}

function stringifySupportingNodes(nodeIds: string[]) {
  return JSON.stringify([...new Set(nodeIds)]);
}

function hasColumn(db: Database.Database, table: string, column: string): boolean {
  const columns = db.prepare(`PRAGMA table_info(${table})`).all() as Array<{ name: string }>;
  return columns.some((entry) => entry.name === column);
}

export class KnownDB {
  db: Database.Database;

  constructor(dbPath: string) {
    mkdirSync(dirname(dbPath), { recursive: true });
    this.db = new Database(dbPath);
    this.db.pragma("journal_mode = WAL");
    this.db.pragma("foreign_keys = ON");
    this.db.exec(SCHEMA);
    this.migrate();
  }

  private migrate() {
    if (!hasColumn(this.db, "nodes", "times_observed")) {
      this.db.prepare("ALTER TABLE nodes ADD COLUMN times_observed REAL DEFAULT 1").run();
    }
    if (!hasColumn(this.db, "insights", "initiated_at")) {
      this.db.prepare("ALTER TABLE insights ADD COLUMN initiated_at TEXT").run();
    }

    this.db.prepare("UPDATE nodes SET times_observed = COALESCE(times_observed, 1)").run();
  }

  close() {
    this.db.close();
  }

  // --- Nodes ---

  insertNode(node: Omit<NodeRow, "id" | "created_at" | "updated_at"> & { id?: string }): NodeRow {
    const now = nowIso();
    const timesObserved = Math.max(node.times_observed ?? 1, 0.01);
    const row: NodeRow = {
      id: node.id ?? nanoid(),
      type: node.type,
      text: node.text,
      confidence: node.confidence ?? computeObservationConfidence(timesObserved, now),
      source: node.source ?? null,
      created_at: now,
      updated_at: now,
      decay_rate: node.decay_rate ?? 0.01,
      times_observed: timesObserved,
      embedding: node.embedding ?? null,
    };

    this.db
      .prepare(
        `INSERT INTO nodes (id, type, text, confidence, source, created_at, updated_at, decay_rate, times_observed, embedding)
         VALUES (@id, @type, @text, @confidence, @source, @created_at, @updated_at, @decay_rate, @times_observed, @embedding)`,
      )
      .run(row);

    return row;
  }

  findNodeByTypeAndText(_type: string, text: string): NodeRow | undefined {
    return this.db
      .prepare(
        `SELECT * FROM nodes
         WHERE LOWER(TRIM(text)) = LOWER(TRIM(?))
         ORDER BY confidence DESC, updated_at DESC
         LIMIT 1`,
      )
      .get(text) as NodeRow | undefined;
  }

  reconfirmNodeObservation(
    id: string,
    updates?: {
      source?: string | null;
      embedding?: Buffer | null;
      type?: string;
      text?: string;
    },
  ): NodeRow | undefined {
    const existing = this.getNode(id);
    if (!existing) {
      return undefined;
    }

    const updatedAt = nowIso();
    const timesObserved = Math.max(existing.times_observed + 1, 0.01);
    const confidence = computeObservationConfidence(timesObserved, updatedAt);

    this.db
      .prepare(
        `UPDATE nodes
         SET confidence = ?,
             source = ?,
             updated_at = ?,
             embedding = ?,
             type = ?,
             text = ?,
             times_observed = ?
         WHERE id = ?`,
      )
      .run(
        confidence,
        updates?.source ?? existing.source,
        updatedAt,
        updates?.embedding ?? existing.embedding,
        updates?.type ?? existing.type,
        updates?.text ?? existing.text,
        timesObserved,
        existing.id,
      );

    return this.getNode(existing.id);
  }

  applyContradictionPenalty(id: string, factor: number = 0.7): NodeRow | undefined {
    const existing = this.getNode(id);
    if (!existing) {
      return undefined;
    }

    const currentConfidence = computeObservationConfidence(existing.times_observed, existing.updated_at);
    const targetConfidence = Math.max(0.01, currentConfidence * factor);
    const elapsedDays = Math.max(daysSince(existing.updated_at), 1);
    const timesObserved = observationCountForConfidence(targetConfidence, elapsedDays);

    this.db
      .prepare(
        `UPDATE nodes
         SET confidence = ?,
             times_observed = ?
         WHERE id = ?`,
      )
      .run(targetConfidence, timesObserved, existing.id);

    return this.getNode(existing.id);
  }

  upsertNodeObservation(node: Omit<NodeRow, "id" | "created_at" | "updated_at"> & { id?: string }): {
    created: boolean;
    row: NodeRow;
  } {
    const existing = this.findNodeByTypeAndText(node.type, node.text);

    if (!existing) {
      return { created: true, row: this.insertNode(node) };
    }

    const row =
      this.reconfirmNodeObservation(existing.id, {
        source: node.source ?? existing.source,
        embedding: node.embedding ?? existing.embedding,
        type: node.type || existing.type,
        text: existing.text,
      }) ?? existing;

    return { created: false, row };
  }

  updateNodeEmbedding(id: string, embedding: Buffer) {
    this.db.prepare("UPDATE nodes SET embedding = ? WHERE id = ?").run(embedding, id);
  }

  updateNodeConfidence(id: string, confidence: number) {
    this.db.prepare("UPDATE nodes SET confidence = ? WHERE id = ?").run(confidence, id);
  }

  touchNode(id: string) {
    this.db.prepare("UPDATE nodes SET updated_at = ? WHERE id = ?").run(nowIso(), id);
  }

  getNode(id: string): NodeRow | undefined {
    return this.db.prepare("SELECT * FROM nodes WHERE id = ?").get(id) as NodeRow | undefined;
  }

  getAllNodes(): NodeRow[] {
    return this.db.prepare("SELECT * FROM nodes").all() as NodeRow[];
  }

  getNodesByType(type: string): NodeRow[] {
    return this.db
      .prepare(
        `SELECT * FROM nodes
         WHERE type = ?
           AND embedding IS NOT NULL
           AND confidence > 0.1
         ORDER BY confidence DESC, times_observed DESC`,
      )
      .all(type) as NodeRow[];
  }

  getDistinctNodeTypes(): string[] {
    return (
      this.db
        .prepare(
          `SELECT DISTINCT type
           FROM nodes
           WHERE embedding IS NOT NULL
             AND confidence > 0.1
             AND TRIM(type) <> ''`,
        )
        .all() as Array<{ type: string }>
    ).map((row) => row.type);
  }

  getNodesWithEmbeddings(): NodeRow[] {
    return this.db.prepare("SELECT * FROM nodes WHERE embedding IS NOT NULL AND confidence > 0.1").all() as NodeRow[];
  }

  removeNodeFromInsightSupport(nodeId: string) {
    const insights = this.getAllInsights();
    const deleteInsight = this.db.prepare("DELETE FROM insights WHERE id = ?");
    const updateInsight = this.db.prepare("UPDATE insights SET supporting_nodes = ? WHERE id = ?");

    const tx = this.db.transaction(() => {
      for (const insight of insights) {
        const currentNodeIds = parseSupportingNodes(insight.supporting_nodes);
        const filtered = currentNodeIds.filter((candidate) => candidate !== nodeId);
        if (filtered.length === currentNodeIds.length) {
          continue;
        }

        if (filtered.length === 0) {
          deleteInsight.run(insight.id);
          continue;
        }

        updateInsight.run(stringifySupportingNodes(filtered), insight.id);
      }
    });

    tx();
  }

  deleteNode(id: string) {
    this.removeNodeFromInsightSupport(id);
    this.db.prepare("DELETE FROM edges WHERE source_id = ? OR target_id = ?").run(id, id);
    this.db.prepare("DELETE FROM nodes WHERE id = ?").run(id);
  }

  deleteNodesBelow(confidenceThreshold: number) {
    const ids = this.db
      .prepare("SELECT id FROM nodes WHERE confidence < ?")
      .all(confidenceThreshold) as { id: string }[];

    const tx = this.db.transaction(() => {
      for (const { id } of ids) {
        this.deleteNode(id);
      }
    });

    tx();
    return ids.length;
  }

  // --- Edges ---

  insertEdge(edge: Omit<EdgeRow, "id" | "created_at"> & { id?: string }): EdgeRow {
    const row: EdgeRow = {
      id: edge.id ?? nanoid(),
      source_id: edge.source_id,
      target_id: edge.target_id,
      relation: edge.relation,
      text: edge.text ?? null,
      confidence: edge.confidence ?? 1.0,
      source: edge.source ?? null,
      created_at: nowIso(),
    };

    this.db
      .prepare(
        `INSERT INTO edges (id, source_id, target_id, relation, text, confidence, source, created_at)
         VALUES (@id, @source_id, @target_id, @relation, @text, @confidence, @source, @created_at)`,
      )
      .run(row);

    return row;
  }

  findEdge(sourceId: string, targetId: string, relation: string): EdgeRow | undefined {
    return this.db
      .prepare(
        `SELECT * FROM edges
         WHERE source_id = ?
           AND target_id = ?
           AND relation = ?
         LIMIT 1`,
      )
      .get(sourceId, targetId, relation) as EdgeRow | undefined;
  }

  upsertEdge(edge: Omit<EdgeRow, "id" | "created_at"> & { id?: string }): { created: boolean; row: EdgeRow } {
    const existing = this.findEdge(edge.source_id, edge.target_id, edge.relation);

    if (!existing) {
      return { created: true, row: this.insertEdge(edge) };
    }

    this.db
      .prepare(
        `UPDATE edges
         SET confidence = ?,
             source = ?,
             text = COALESCE(?, text)
         WHERE id = ?`,
      )
      .run(Math.max(existing.confidence, edge.confidence ?? 1.0), edge.source ?? existing.source, edge.text ?? existing.text, existing.id);

    return { created: false, row: this.getEdge(existing.id)! };
  }

  getEdge(id: string): EdgeRow | undefined {
    return this.db.prepare("SELECT * FROM edges WHERE id = ?").get(id) as EdgeRow | undefined;
  }

  getEdgesFrom(nodeId: string): EdgeRow[] {
    return this.db.prepare("SELECT * FROM edges WHERE source_id = ?").all(nodeId) as EdgeRow[];
  }

  getEdgesTo(nodeId: string): EdgeRow[] {
    return this.db.prepare("SELECT * FROM edges WHERE target_id = ?").all(nodeId) as EdgeRow[];
  }

  getEdgesForNode(nodeId: string): EdgeRow[] {
    return this.db.prepare("SELECT * FROM edges WHERE source_id = ? OR target_id = ?").all(nodeId, nodeId) as EdgeRow[];
  }

  getEdgesBetweenNodes(nodeIds: string[]): EdgeRow[] {
    if (nodeIds.length === 0) {
      return [];
    }

    const placeholders = nodeIds.map(() => "?").join(",");
    return this.db
      .prepare(
        `SELECT * FROM edges
         WHERE source_id IN (${placeholders})
           AND target_id IN (${placeholders})`,
      )
      .all(...nodeIds, ...nodeIds) as EdgeRow[];
  }

  mergeNodeInto(keepId: string, removeId: string) {
    if (keepId === removeId) {
      return;
    }

    const keep = this.getNode(keepId);
    const remove = this.getNode(removeId);
    if (!keep || !remove) {
      return;
    }

    const tx = this.db.transaction(() => {
      const insightMap = new Map<string, InsightRow>();
      for (const insight of this.getInsightsReferencingNode(keepId)) {
        insightMap.set(insight.id, insight);
      }
      for (const insight of this.getInsightsReferencingNode(removeId)) {
        insightMap.set(insight.id, insight);
      }

      for (const insight of insightMap.values()) {
        const replaced = parseSupportingNodes(insight.supporting_nodes).map((nodeId) => (nodeId === removeId ? keepId : nodeId));
        this.updateInsightSupportingNodes(insight.id, replaced);
      }

      const edges = this.getEdgesForNode(removeId);
      for (const edge of edges) {
        const sourceId = edge.source_id === removeId ? keepId : edge.source_id;
        const targetId = edge.target_id === removeId ? keepId : edge.target_id;

        if (sourceId === targetId) {
          continue;
        }

        this.upsertEdge({
          source_id: sourceId,
          target_id: targetId,
          relation: edge.relation,
          text: edge.text,
          confidence: edge.confidence,
          source: edge.source,
        });
      }

      this.db.prepare("DELETE FROM edges WHERE source_id = ? OR target_id = ?").run(removeId, removeId);
      this.db.prepare("DELETE FROM nodes WHERE id = ?").run(removeId);
    });

    tx();
  }

  // --- Insights ---

  insertInsight(
    insight: Omit<InsightRow, "id" | "discovered_at" | "times_rediscovered" | "times_used" | "last_used" | "initiated_at"> & {
      id?: string;
    },
  ): InsightRow {
    const row: InsightRow = {
      id: insight.id ?? nanoid(),
      text: insight.text,
      supporting_nodes: insight.supporting_nodes,
      confidence: insight.confidence ?? 0.4,
      discovered_at: nowIso(),
      times_rediscovered: 0,
      times_used: 0,
      last_used: null,
      initiated_at: null,
      embedding: insight.embedding ?? null,
    };

    this.db
      .prepare(
        `INSERT INTO insights (id, text, supporting_nodes, confidence, discovered_at, times_rediscovered, times_used, last_used, initiated_at, embedding)
         VALUES (@id, @text, @supporting_nodes, @confidence, @discovered_at, @times_rediscovered, @times_used, @last_used, @initiated_at, @embedding)`,
      )
      .run(row);

    return row;
  }

  updateInsightEmbedding(id: string, embedding: Buffer) {
    this.db.prepare("UPDATE insights SET embedding = ? WHERE id = ?").run(embedding, id);
  }

  updateInsightSupportingNodes(id: string, nodeIds: string[]) {
    const deduped = stringifySupportingNodes(nodeIds);
    this.db.prepare("UPDATE insights SET supporting_nodes = ? WHERE id = ?").run(deduped, id);
  }

  addInsightSupport(id: string, nodeIds: string[]) {
    const existing = this.getInsight(id);
    if (!existing) {
      return;
    }

    const merged = [...parseSupportingNodes(existing.supporting_nodes), ...nodeIds];
    this.updateInsightSupportingNodes(id, merged);
  }

  strengthenInsight(id: string) {
    this.db
      .prepare(
        `UPDATE insights
         SET times_rediscovered = times_rediscovered + 1,
             confidence = MIN(1.0, confidence + 0.1)
         WHERE id = ?`,
      )
      .run(id);
  }

  markInsightUsed(id: string) {
    this.db
      .prepare(
        `UPDATE insights
         SET times_used = times_used + 1,
             last_used = ?,
             confidence = MIN(1.0, confidence + 0.05)
         WHERE id = ?`,
      )
      .run(nowIso(), id);
  }

  markInsightInitiated(id: string) {
    this.db.prepare("UPDATE insights SET initiated_at = ? WHERE id = ?").run(nowIso(), id);
  }

  getInsight(id: string): InsightRow | undefined {
    return this.db.prepare("SELECT * FROM insights WHERE id = ?").get(id) as InsightRow | undefined;
  }

  getAllInsights(): InsightRow[] {
    return this.db.prepare("SELECT * FROM insights").all() as InsightRow[];
  }

  getInsightsWithEmbeddings(): InsightRow[] {
    return this.db.prepare("SELECT * FROM insights WHERE embedding IS NOT NULL").all() as InsightRow[];
  }

  getMostRecentInsightInitiatedAt(): string | null {
    const row = this.db.prepare("SELECT MAX(initiated_at) AS initiated_at FROM insights").get() as {
      initiated_at: string | null;
    };
    return row.initiated_at ?? null;
  }

  getInsightsReferencingNode(nodeId: string): InsightRow[] {
    return this.getAllInsights().filter((insight) => parseSupportingNodes(insight.supporting_nodes).includes(nodeId));
  }

  getTopInsights(limit: number = 10): InsightRow[] {
    return this.db.prepare("SELECT * FROM insights ORDER BY confidence DESC LIMIT ?").all(limit) as InsightRow[];
  }

  deleteDeadInsights() {
    const result = this.db.prepare("DELETE FROM insights WHERE confidence < 0.2 AND times_used = 0").run();
    return result.changes;
  }

  // --- Graph Traversal ---

  expandViaEdges(nodeIds: string[], depth: number = 1): NodeRow[] {
    if (nodeIds.length === 0) {
      return [];
    }

    const placeholders = nodeIds.map(() => "?").join(",");
    return this.db
      .prepare(
        `WITH RECURSIVE reachable(id, hop) AS (
          SELECT id, 0 FROM nodes WHERE id IN (${placeholders})
          UNION
          SELECT CASE WHEN e.source_id = r.id THEN e.target_id ELSE e.source_id END, r.hop + 1
          FROM edges e
          JOIN reachable r ON (e.source_id = r.id OR e.target_id = r.id)
          WHERE r.hop < ?
        )
        SELECT DISTINCT n.* FROM nodes n
        JOIN reachable r ON n.id = r.id
        WHERE n.confidence > 0.1`,
      )
      .all(...nodeIds, depth) as NodeRow[];
  }

  // --- Stats ---

  getStats() {
    const nodeCount = (this.db.prepare("SELECT COUNT(*) AS c FROM nodes").get() as { c: number }).c;
    const edgeCount = (this.db.prepare("SELECT COUNT(*) AS c FROM edges").get() as { c: number }).c;
    const insightCount = (this.db.prepare("SELECT COUNT(*) AS c FROM insights").get() as { c: number }).c;
    const topInsights = this.getTopInsights(5);
    return { nodeCount, edgeCount, insightCount, topInsights };
  }
}
