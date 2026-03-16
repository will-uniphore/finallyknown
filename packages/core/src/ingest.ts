import type { KnownConfig } from "./config.js";
import type { KnownDB, NodeRow } from "./db.js";
import { generateEmbeddings, semanticSearch } from "./embeddings.js";
import { getOpenAIClient } from "./openai.js";
import { CONTRADICTION_SYSTEM, CONTRADICTION_USER, INGEST_SYSTEM, INGEST_USER } from "./prompts/ingest.js";

const MIN_SESSION_CHARS = 500;
const DEDUP_SIMILARITY_THRESHOLD = 0.7;
const MAX_CONTRADICTION_CHECKS = 3;

interface ExtractedNode {
  text: string;
  type?: string;
}

interface ExtractedEdge {
  source_text: string;
  target_text: string;
  relation: string;
  text?: string;
}

interface ExtractionResult {
  nodes?: ExtractedNode[];
  edges?: ExtractedEdge[];
}

interface ContradictionResult {
  relation?: "same" | "contradict" | "different";
}

function normalizeNodeType(type?: string): string {
  const normalized = type
    ?.trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");

  return normalized || "trait";
}

function dedupeNodes(nodes: ExtractedNode[]) {
  const deduped = new Map<string, ExtractedNode>();
  for (const node of nodes) {
    const text = node.text.trim();
    if (!text) {
      continue;
    }

    const key = text.toLowerCase();
    const type = normalizeNodeType(node.type);
    if (!deduped.has(key)) {
      deduped.set(key, { text, type });
    }
  }

  return [...deduped.values()];
}

function parseExtraction(content: string): ExtractionResult {
  try {
    const parsed = JSON.parse(content) as ExtractionResult;
    return {
      nodes: Array.isArray(parsed.nodes) ? parsed.nodes : [],
      edges: Array.isArray(parsed.edges) ? parsed.edges : [],
    };
  } catch {
    return { nodes: [], edges: [] };
  }
}

function hasUserMessages(sessionText: string): boolean {
  if (/\bUSER\s*:/i.test(sessionText)) {
    return true;
  }

  if (/"role"\s*:\s*"user"/i.test(sessionText)) {
    return true;
  }

  if (/\bASSISTANT\s*:/i.test(sessionText) || /"role"\s*:\s*"assistant"/i.test(sessionText)) {
    return false;
  }

  return true;
}

async function judgeObservationRelationship(
  config: KnownConfig,
  existing: string,
  candidate: string,
): Promise<"same" | "contradict" | "different"> {
  const openai = getOpenAIClient(config);
  const response = await openai.chat.completions.create({
    model: config.extractionModel,
    messages: [
      { role: "system", content: CONTRADICTION_SYSTEM },
      { role: "user", content: CONTRADICTION_USER(existing, candidate) },
    ],
    response_format: { type: "json_object" },
    temperature: 0,
  });

  const content = response.choices[0]?.message?.content;
  if (!content) {
    return "different";
  }

  try {
    const parsed = JSON.parse(content) as ContradictionResult;
    return parsed.relation === "same" || parsed.relation === "contradict" ? parsed.relation : "different";
  } catch {
    return "different";
  }
}

function shouldSkipSession(sessionText: string): boolean {
  return sessionText.trim().length < MIN_SESSION_CHARS || !hasUserMessages(sessionText);
}

async function resolveObservationStorage(
  db: KnownDB,
  config: KnownConfig,
  node: ExtractedNode,
  embedding: Buffer,
  source: string | null,
): Promise<{ created: boolean; row: NodeRow }> {
  const exact = db.findNodeByTypeAndText(normalizeNodeType(node.type), node.text);
  if (exact) {
    const row = db.reconfirmNodeObservation(exact.id, {
      source,
      embedding,
      type: exact.type === "trait" ? normalizeNodeType(node.type) : exact.type,
    });

    return { created: false, row: row ?? exact };
  }

  const similar = semanticSearch(embedding, db.getNodesWithEmbeddings(), MAX_CONTRADICTION_CHECKS).filter(
    (candidate) => candidate.similarity >= DEDUP_SIMILARITY_THRESHOLD,
  );

  let confirmedMatch: NodeRow | undefined;

  for (const candidate of similar) {
    const relation = await judgeObservationRelationship(config, candidate.text, node.text);
    if (relation === "same") {
      confirmedMatch = db.reconfirmNodeObservation(candidate.id, {
        source,
        embedding,
        type: candidate.type === "trait" ? normalizeNodeType(node.type) : candidate.type,
      });
      break;
    }

    if (relation === "contradict") {
      db.applyContradictionPenalty(candidate.id, 0.7);
    }
  }

  if (confirmedMatch) {
    return { created: false, row: confirmedMatch };
  }

  const row = db.insertNode({
    type: normalizeNodeType(node.type),
    text: node.text.trim(),
    confidence: 1,
    source,
    decay_rate: 0.01,
    times_observed: 1,
    embedding,
  });

  return { created: true, row };
}

const MAX_CHUNK_CHARS = 30000; // ~22K tokens — safe for mini's 128K window with prompt overhead

function chunkText(text: string, maxChars: number): string[] {
  if (text.length <= maxChars) return [text];
  const chunks: string[] = [];
  for (let i = 0; i < text.length; i += maxChars) {
    chunks.push(text.slice(i, i + maxChars));
  }
  return chunks;
}

export async function ingest(
  db: KnownDB,
  sessionText: string,
  config: KnownConfig,
  sessionId?: string,
): Promise<{ nodesCreated: number; edgesCreated: number }> {
  if (shouldSkipSession(sessionText)) {
    return { nodesCreated: 0, edgesCreated: 0 };
  }

  // Chunk long sessions to stay within context window
  const chunks = chunkText(sessionText, MAX_CHUNK_CHARS);
  let totalNodesCreated = 0;
  let totalEdgesCreated = 0;

  for (const chunk of chunks) {
    const result = await ingestChunk(db, chunk, config, sessionId);
    totalNodesCreated += result.nodesCreated;
    totalEdgesCreated += result.edgesCreated;
  }

  return { nodesCreated: totalNodesCreated, edgesCreated: totalEdgesCreated };
}

async function ingestChunk(
  db: KnownDB,
  sessionText: string,
  config: KnownConfig,
  sessionId?: string,
): Promise<{ nodesCreated: number; edgesCreated: number }> {
  const openai = getOpenAIClient(config);
  const response = await openai.chat.completions.create({
    model: config.extractionModel,
    messages: [
      { role: "system", content: INGEST_SYSTEM },
      { role: "user", content: INGEST_USER(sessionText) },
    ],
    response_format: { type: "json_object" },
    temperature: 0.1,
  });

  const content = response.choices[0]?.message?.content;
  if (!content) {
    return { nodesCreated: 0, edgesCreated: 0 };
  }

  const extracted = parseExtraction(content);
  const nodes = dedupeNodes(extracted.nodes ?? []);
  if (nodes.length === 0) {
    return { nodesCreated: 0, edgesCreated: 0 };
  }

  const embeddings = await generateEmbeddings(nodes.map((node) => node.text), config);
  const textToNodeId = new Map<string, string>();
  let nodesCreated = 0;

  for (const [index, node] of nodes.entries()) {
    const result = await resolveObservationStorage(db, config, node, embeddings[index]!, sessionId ?? null);
    textToNodeId.set(node.text, result.row.id);
    if (result.created) {
      nodesCreated += 1;
    }
  }

  let edgesCreated = 0;
  for (const edge of extracted.edges ?? []) {
    const sourceText = edge.source_text?.trim();
    const targetText = edge.target_text?.trim();
    const relation = edge.relation?.trim();

    if (!sourceText || !targetText || !relation) {
      continue;
    }

    const sourceId = textToNodeId.get(sourceText);
    const targetId = textToNodeId.get(targetText);
    if (!sourceId || !targetId || sourceId === targetId) {
      continue;
    }

    const result = db.upsertEdge({
      source_id: sourceId,
      target_id: targetId,
      relation,
      text: edge.text?.trim() || null,
      confidence: 1.0,
      source: sessionId ?? null,
    });

    if (result.created) {
      edgesCreated += 1;
    }
  }

  return { nodesCreated, edgesCreated };
}
