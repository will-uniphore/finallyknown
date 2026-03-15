import { join } from "path";
import { homedir } from "os";

export interface KnownConfig {
  openaiApiKey: string;
  dbPath: string;
  embeddingModel: string;
  extractionModel: string;
  synthesisModel: string;
}

export function getConfig(overrides?: Partial<KnownConfig>): KnownConfig {
  const knownDir = join(homedir(), ".known");
  return {
    openaiApiKey: overrides?.openaiApiKey ?? process.env.OPENAI_API_KEY ?? "",
    dbPath: overrides?.dbPath ?? process.env.KNOWN_DB_PATH ?? join(knownDir, "brain.db"),
    embeddingModel: overrides?.embeddingModel ?? "text-embedding-3-small",
    extractionModel: overrides?.extractionModel ?? "gpt-4o-mini",
    synthesisModel: overrides?.synthesisModel ?? "gpt-4o",
  };
}
