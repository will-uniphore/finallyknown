import OpenAI from "openai";
import type { KnownConfig } from "./config.js";

let client: OpenAI | null = null;
let clientApiKey = "";

export function getOpenAIClient(config: KnownConfig): OpenAI {
  if (!config.openaiApiKey) {
    throw new Error("OPENAI_API_KEY is required for ingest, query, and discover.");
  }

  if (!client || clientApiKey !== config.openaiApiKey) {
    client = new OpenAI({ apiKey: config.openaiApiKey });
    clientApiKey = config.openaiApiKey;
  }

  return client;
}
