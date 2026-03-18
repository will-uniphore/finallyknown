import type { KnownConfig } from "./config.js";
import type { KnownDB, NodeRow } from "./db.js";
import { generateEmbeddings, semanticSearch } from "./embeddings.js";
import { getOpenAIClient } from "./openai.js";
import {
  CONTRADICTION_SYSTEM,
  CONTRADICTION_USER,
  INGEST_FACT_SYSTEM,
  INGEST_FACT_USER,
  INGEST_SYSTEM,
  INGEST_USER,
} from "./prompts/ingest.js";

const MIN_SESSION_CHARS = 500;
const DEDUP_SIMILARITY_THRESHOLD = 0.8;
const MAX_CONTRADICTION_CHECKS = 3;
const FACT_TYPE_PREFIX = "fact:";

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
  const trimmed = type?.trim().toLowerCase() ?? "";
  const isFact = trimmed.startsWith(FACT_TYPE_PREFIX);
  const rawType = isFact ? trimmed.slice(FACT_TYPE_PREFIX.length) : trimmed;
  const normalized = rawType
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");

  if (isFact) {
    return `${FACT_TYPE_PREFIX}${normalized || "detail"}`;
  }

  return normalized || "trait";
}

function isFactNodeType(type: string) {
  return type.startsWith(FACT_TYPE_PREFIX);
}

function isGenericNodeType(type: string) {
  return type === "trait" || type === `${FACT_TYPE_PREFIX}detail`;
}

function choosePreferredNodeType(existingType: string, incomingType: string) {
  if (isFactNodeType(incomingType) && !isFactNodeType(existingType)) {
    return incomingType;
  }

  if (!isGenericNodeType(incomingType) && isGenericNodeType(existingType)) {
    return incomingType;
  }

  return existingType;
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
    const existing = deduped.get(key);
    if (!existing) {
      deduped.set(key, { text, type });
      continue;
    }

    deduped.set(key, {
      text,
      type: choosePreferredNodeType(existing.type ?? "trait", type),
    });
  }

  return [...deduped.values()];
}

function dedupeEdges(edges: ExtractedEdge[]) {
  const deduped = new Map<string, ExtractedEdge>();

  for (const edge of edges) {
    const sourceText = edge.source_text?.trim();
    const targetText = edge.target_text?.trim();
    const relation = edge.relation?.trim();
    if (!sourceText || !targetText || !relation) {
      continue;
    }

    const text = edge.text?.trim();
    const key = [sourceText.toLowerCase(), targetText.toLowerCase(), relation.toLowerCase(), text?.toLowerCase() ?? ""].join("\0");
    if (!deduped.has(key)) {
      deduped.set(key, {
        source_text: sourceText,
        target_text: targetText,
        relation,
        text: text || undefined,
      });
    }
  }

  return [...deduped.values()];
}

function tagFactNodes(nodes: ExtractedNode[]) {
  return nodes.map((node) => ({
    ...node,
    type: `${FACT_TYPE_PREFIX}${node.type?.trim() || "detail"}`,
  }));
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

function shouldReplaceStoredType(existingType: string, incomingType: string) {
  if (isFactNodeType(incomingType) && !isFactNodeType(existingType)) {
    return true;
  }

  return isGenericNodeType(existingType);
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
    const nextType = normalizeNodeType(node.type);
    const row = db.reconfirmNodeObservation(exact.id, {
      source,
      embedding,
      type: shouldReplaceStoredType(exact.type, nextType) ? nextType : exact.type,
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
      const nextType = normalizeNodeType(node.type);
      confirmedMatch = db.reconfirmNodeObservation(candidate.id, {
        source,
        embedding,
        type: shouldReplaceStoredType(candidate.type, nextType) ? nextType : candidate.type,
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
  const [patternResponse, factResponse] = await Promise.all([
    openai.chat.completions.create({
      model: config.extractionModel,
      messages: [
        { role: "system", content: INGEST_SYSTEM },
        { role: "user", content: INGEST_USER(sessionText) },
      ],
      response_format: { type: "json_object" },
      temperature: 0.1,
    }),
    openai.chat.completions.create({
      model: config.extractionModel,
      messages: [
        { role: "system", content: INGEST_FACT_SYSTEM },
        { role: "user", content: INGEST_FACT_USER(sessionText) },
      ],
      response_format: { type: "json_object" },
      temperature: 0.1,
    }),
  ]);

  const patternExtraction = parseExtraction(patternResponse.choices[0]?.message?.content ?? "");
  const factExtraction = parseExtraction(factResponse.choices[0]?.message?.content ?? "");
  const nodes = dedupeNodes([...(patternExtraction.nodes ?? []), ...tagFactNodes(factExtraction.nodes ?? [])]);
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
  for (const edge of dedupeEdges([...(patternExtraction.edges ?? []), ...(factExtraction.edges ?? [])])) {
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
