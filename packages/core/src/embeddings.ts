import type { KnownConfig } from "./config.js";
import { getOpenAIClient } from "./openai.js";

export async function generateEmbedding(text: string, config: KnownConfig): Promise<Buffer> {
  const openai = getOpenAIClient(config);
  const response = await openai.embeddings.create({
    model: config.embeddingModel,
    input: text,
  });

  return Buffer.from(new Float32Array(response.data[0].embedding).buffer);
}

export async function generateEmbeddings(texts: string[], config: KnownConfig): Promise<Buffer[]> {
  if (texts.length === 0) {
    return [];
  }

  const openai = getOpenAIClient(config);
  const response = await openai.embeddings.create({
    model: config.embeddingModel,
    input: texts,
  });

  return response.data
    .sort((a, b) => a.index - b.index)
    .map((item) => Buffer.from(new Float32Array(item.embedding).buffer));
}

export function bufferToVector(buf: Buffer): Float32Array {
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / Float32Array.BYTES_PER_ELEMENT);
}

export function cosineSimilarity(a: Buffer, b: Buffer): number {
  const va = bufferToVector(a);
  const vb = bufferToVector(b);
  const length = Math.min(va.length, vb.length);

  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < length; i++) {
    dot += va[i] * vb[i];
    normA += va[i] * va[i];
    normB += vb[i] * vb[i];
  }

  const denominator = Math.sqrt(normA) * Math.sqrt(normB);
  return denominator === 0 ? 0 : dot / denominator;
}

export function semanticSearch<T extends { embedding: Buffer | null }>(
  query: Buffer,
  items: T[],
  limit: number = 20
): (T & { similarity: number })[] {
  return items
    .filter((item) => item.embedding !== null)
    .map((item) => ({
      ...item,
      similarity: cosineSimilarity(query, item.embedding!),
    }))
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, limit);
}
