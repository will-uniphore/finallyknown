import type { KnownConfig } from "./config.js";
import type { KnownDB } from "./db.js";
import { generateEmbeddings } from "./embeddings.js";
import { getOpenAIClient } from "./openai.js";
import { INGEST_SYSTEM, INGEST_USER } from "./prompts/ingest.js";

interface ExtractedNode {
  type: string;
  text: string;
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

function dedupeNodes(nodes: ExtractedNode[]) {
  const deduped = new Map<string, ExtractedNode>();
  for (const node of nodes) {
    const type = node.type.trim();
    const text = node.text.trim();
    if (!type || !text) {
      continue;
    }
    deduped.set(`${type.toLowerCase()}::${text.toLowerCase()}`, { type, text });
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

export async function ingest(
  db: KnownDB,
  sessionText: string,
  config: KnownConfig,
  sessionId?: string
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
    const result = db.upsertNodeObservation({
      type: node.type,
      text: node.text,
      confidence: 1.0,
      source: sessionId ?? null,
      decay_rate: 0.01,
      embedding: embeddings[index] ?? null,
    });

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
    if (!sourceId || !targetId) {
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
