import { execFile } from "node:child_process";
import dotenv from "dotenv";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import type { KnownConfig } from "../config.js";
import { getConfig } from "../config.js";
import { KnownDB, type NodeRow } from "../db.js";
import { discover } from "../discover.js";
import { ingest } from "../ingest.js";
import { getOpenAIClient } from "../openai.js";
import { think } from "../think.js";

type GoldenEvalTestId = "encode" | "activate" | "dream" | "implicit" | "personality";
type Status = "PASS" | "WARN" | "FAIL";
type ActivateCaseType = "knowledge" | "privacy";

interface GoldenEvalOptions {
  goldenPath: string;
  test: GoldenEvalTestId | "all";
  limit?: number;
}

interface GoldenEvalData {
  version: string;
  description: string;
  test_cases: GoldenEvalCase[];
}

interface EncodeCase {
  id: string;
  type: "encode_quality";
  source: string;
  input: {
    conversation_text: string;
    conversation_length?: number;
  };
  ground_truth: {
    traits: string[];
    expanded_persona?: string;
  };
  metrics: string[];
}

interface ActivateCase {
  id: string;
  type: "activate_accuracy";
  source: string;
  input: {
    question: string;
    persona_id: number;
  };
  ground_truth: {
    correct_answer: string;
    preference_tested: string;
    is_updated: boolean;
    prev_pref: string | null;
  };
  metrics: string[];
}

interface DreamCase {
  id: string;
  type: "dream_discovery";
  source: string;
  input: {
    persona_id: number;
    conversation_text: string;
  };
  ground_truth: {
    domains_present?: Record<string, unknown>;
    expected?: string;
  };
  metrics: string[];
}

interface ImplicitCase {
  id: string;
  type: "implicit_inference";
  source: string;
  input: {
    question: string;
    persona_id: number;
  };
  ground_truth: {
    hidden_attribute: string;
    correct_response: string;
  };
  metrics: string[];
}

interface BigFiveScores {
  openness: number;
  conscientiousness: number;
  extraversion: number;
  agreeableness: number;
  neuroticism: number;
}

interface PersonalityCase {
  id: string;
  type: "personality_extraction";
  source: string;
  input: {
    text: string;
  };
  ground_truth: BigFiveScores;
  metrics: string[];
}

type GoldenEvalCase = EncodeCase | ActivateCase | DreamCase | ImplicitCase | PersonalityCase;

interface PersonaMemPersona {
  personaId: number;
  conversationText: string;
  groundTruthTraits: string[];
}

interface PersonaMemExtractedData {
  personas: PersonaMemPersona[];
}

interface TraitJudgeResult {
  matched_ground_truth_indices?: number[];
  matched_extracted_indices?: number[];
  hallucinated_extracted_indices?: number[];
}

interface ActivateJudgeResult {
  results?: Array<{
    id: string;
    incorporates_preference?: "yes" | "no";
    factual_accuracy?: "yes" | "no";
  }>;
}

interface ActivateJudgeBatchLog {
  batchIndex: number;
  input: Array<{
    id: string;
    question: string;
    preferenceTested: string;
    correctAnswer: string;
    response: string;
    sensitive: boolean;
  }>;
  rawOutput: string;
  parsedResults: Array<{
    id: string;
    incorporates_preference?: "yes" | "no";
    factual_accuracy?: "yes" | "no";
  }>;
  missingIds: string[];
}

interface DreamJudgeResult {
  ratings?: Array<{
    id: string;
    score?: number;
    cross_domain?: "yes" | "no";
  }>;
}

interface ImplicitJudgeResult {
  results?: Array<{
    id: string;
    detected?: "yes" | "no";
    score?: number;
  }>;
}

interface ScorecardRow {
  test: string;
  metric: string;
  value: string;
  target: string;
  status: Status;
}

interface EncodeSummary {
  caseCount: number;
  recall: number;
  precision: number;
  f1: number;
  hallucinationRate: number;
  cases: Array<{
    id: string;
    nodeCount: number;
    recall: number;
    precision: number;
    f1: number;
    hallucinationRate: number;
  }>;
}

interface ActivateSummary {
  caseCount: number;
  knowledgeAccuracy: number;
  privacyHandling: number;
  factualAccuracy: number;
  cases: Array<{
    id: string;
    question: string;
    preferenceTested: string;
    caseType: ActivateCaseType;
    sensitive: boolean;
    response: string;
    unableToReason: boolean;
    incorporatesPreference: boolean;
    factualAccuracy: boolean;
    judgeMissing: boolean;
  }>;
  judgeLogs: ActivateJudgeBatchLog[];
}

interface DreamSummary {
  caseCount: number;
  discoveryRate: number;
  genuineRate: number;
  crossDomainRate: number;
  hallucinationRate: number;
  cases: Array<{
    id: string;
    foundRuns: number;
    totalInsights: number;
    discoveryRate: number;
    genuineRate: number;
    crossDomainRate: number;
    hallucinationRate: number;
  }>;
}

interface ImplicitSummary {
  caseCount: number;
  detectionRate: number;
  inferenceQuality: number;
  cases: Array<{
    id: string;
    detected: boolean;
    score: number;
  }>;
}

interface PersonalitySummary {
  caseCount: number;
  validCases: number;
  meanCorrelation: number;
  traitsAbove025: number;
  bestTrait: {
    trait: keyof BigFiveScores;
    correlation: number;
  };
  correlations: BigFiveScores;
  cases: Array<{
    id: string;
    predicted: BigFiveScores | null;
  }>;
}

const PERSONAMEM_JSON_PATH = "/tmp/known-test-data/extracted/personamem-v2-benchmark.json";
const BIG_FIVE_QUERY =
  'Rate this person on the Big Five traits from 0 to 100. Return ONLY minified JSON with numeric keys "openness", "conscientiousness", "extraversion", "agreeableness", and "neuroticism".';
const TYPE_TO_CASE = {
  encode: "encode_quality",
  activate: "activate_accuracy",
  dream: "dream_discovery",
  implicit: "implicit_inference",
  personality: "personality_extraction",
} as const;
const SCORE_WEIGHTS: Record<GoldenEvalTestId, number> = {
  encode: 0.3,
  activate: 0.3,
  dream: 0.15,
  implicit: 0.1,
  personality: 0.15,
};

function usage() {
  console.log(`Usage:
  known eval [--golden eval/golden-eval.json] [--test encode|activate|dream|implicit|personality|all] [--limit N]`);
}

function parseArgs(args: string[]): GoldenEvalOptions {
  let goldenPath = "eval/golden-eval.json";
  let test: GoldenEvalOptions["test"] = "all";
  let limit: number | undefined;

  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    switch (arg) {
      case "--golden":
        goldenPath = args[index + 1] ?? goldenPath;
        index += 1;
        break;

      case "--test": {
        const value = args[index + 1];
        if (!value || (value !== "all" && !(value in TYPE_TO_CASE))) {
          throw new Error(`Invalid --test value: ${value ?? "(missing)"}`);
        }
        test = value as GoldenEvalOptions["test"];
        index += 1;
        break;
      }

      case "--limit": {
        const value = Number.parseInt(args[index + 1] ?? "", 10);
        if (!Number.isFinite(value) || value <= 0) {
          throw new Error(`Invalid --limit value: ${args[index + 1] ?? "(missing)"}`);
        }
        limit = value;
        index += 1;
        break;
      }

      case "--help":
      case "-h":
        usage();
        process.exit(0);

      default:
        throw new Error(`Unknown eval flag: ${arg}`);
    }
  }

  return { goldenPath, test, limit };
}

function moduleDir() {
  return dirname(fileURLToPath(import.meta.url));
}

function packageRoot() {
  const dir = moduleDir();
  const candidates = [resolve(dir, ".."), resolve(dir, "../..")];

  for (const candidate of candidates) {
    if (existsSync(join(candidate, "package.json"))) {
      return candidate;
    }
  }

  throw new Error("Unable to resolve the packages/core root for golden eval.");
}

function repoRoot() {
  return resolve(packageRoot(), "../..");
}

function resolvePathFromInput(filePath: string) {
  const candidates = [resolve(process.cwd(), filePath), resolve(repoRoot(), filePath)];
  for (const candidate of candidates) {
    if (existsSync(candidate)) {
      return candidate;
    }
  }
  return candidates[0]!;
}

function extractScriptPath() {
  return join(packageRoot(), "src", "tests", "extract-data.py");
}

function loadApiKeyFromDotenv() {
  dotenv.config({ path: join(homedir(), ".known", ".env") });
}

function getEvalConfig(overrides?: Partial<KnownConfig>): KnownConfig {
  const config = getConfig(overrides);
  if (!config.openaiApiKey) {
    throw new Error(`Missing OPENAI_API_KEY. Set it in ${join(homedir(), ".known", ".env")} or the environment.`);
  }
  return config;
}

function createTempDbPath(label: string) {
  const safeLabel = label.replace(/[^a-z0-9_-]+/gi, "-").replace(/^-+|-+$/g, "").toLowerCase() || "golden";
  return join(tmpdir(), `known-golden-${safeLabel}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}.db`);
}

function cleanupDbFiles(dbPath: string) {
  for (const suffix of ["", "-shm", "-wal"]) {
    rmSync(`${dbPath}${suffix}`, { force: true });
  }
}

function readJsonFile<T>(filePath: string): T {
  return JSON.parse(readFileSync(filePath, "utf8")) as T;
}

function runExtractor(): Promise<void> {
  const script = extractScriptPath();
  if (!existsSync(script)) {
    return Promise.reject(new Error(`Missing extractor script at ${script}`));
  }

  return new Promise((resolvePromise, rejectPromise) => {
    execFile("python3", [script], { maxBuffer: 1024 * 1024 * 32 }, (error, _stdout, stderr) => {
      if (error) {
        rejectPromise(new Error(`extract-data.py failed: ${stderr || error.message}`));
        return;
      }
      resolvePromise();
    });
  });
}

async function ensurePersonaMemData() {
  if (existsSync(PERSONAMEM_JSON_PATH)) {
    return;
  }

  console.log("Preparing extracted PersonaMem data...");
  await runExtractor();
}

function clampPercent(value: number) {
  return Math.max(0, Math.min(100, value));
}

function clampUnit(value: number) {
  return Math.max(0, Math.min(1, value));
}

function chunk<T>(items: T[], size: number): T[][] {
  const batches: T[][] = [];
  for (let index = 0; index < items.length; index += size) {
    batches.push(items.slice(index, index + size));
  }
  return batches;
}

function parseJsonObject<T>(text: string): T | null {
  try {
    return JSON.parse(text) as T;
  } catch {
    const match = text.match(/\{[\s\S]*\}/);
    if (!match) {
      return null;
    }

    try {
      return JSON.parse(match[0]) as T;
    } catch {
      return null;
    }
  }
}

function parseSupportingNodeIds(raw: string) {
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter((value): value is string => typeof value === "string") : [];
  } catch {
    return [];
  }
}

function normalizeJudgeIndices(raw: unknown, itemCount: number) {
  if (!Array.isArray(raw) || itemCount <= 0) {
    return new Set<number>();
  }

  const numeric = raw.flatMap((value) => {
    if (typeof value === "number" && Number.isInteger(value)) {
      return [value];
    }
    if (typeof value === "string" && /^-?\d+$/.test(value.trim())) {
      return [Number.parseInt(value, 10)];
    }
    return [];
  });

  const zeroBased = new Set(numeric.filter((value) => value >= 0 && value < itemCount));
  const oneBased = new Set(numeric.filter((value) => value >= 1 && value <= itemCount).map((value) => value - 1));

  if (!numeric.includes(0) && oneBased.size > zeroBased.size) {
    return oneBased;
  }

  return zeroBased;
}

function average(values: number[]) {
  if (values.length === 0) {
    return Number.NaN;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function pearsonCorrelation(xs: number[], ys: number[]) {
  if (xs.length !== ys.length || xs.length < 2) {
    return Number.NaN;
  }

  const meanX = average(xs);
  const meanY = average(ys);

  let numerator = 0;
  let denominatorX = 0;
  let denominatorY = 0;

  for (let index = 0; index < xs.length; index += 1) {
    const diffX = xs[index]! - meanX;
    const diffY = ys[index]! - meanY;
    numerator += diffX * diffY;
    denominatorX += diffX * diffX;
    denominatorY += diffY * diffY;
  }

  const denominator = Math.sqrt(denominatorX) * Math.sqrt(denominatorY);
  return denominator === 0 ? Number.NaN : numerator / denominator;
}

function formatRatio(value: number) {
  return Number.isFinite(value) ? value.toFixed(3) : "n/a";
}

function formatPercent(value: number) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "n/a";
}

function compareThreshold(value: number, target: number, fail: number, type: "min" | "max"): Status {
  if (!Number.isFinite(value)) {
    return "FAIL";
  }

  if (type === "min") {
    if (value >= target) {
      return "PASS";
    }
    if (value < fail) {
      return "FAIL";
    }
    return "WARN";
  }

  if (value <= target) {
    return "PASS";
  }
  if (value > fail) {
    return "FAIL";
  }
  return "WARN";
}

function printSection(title: string) {
  console.log(`\n== ${title} ==`);
}

function printTable(headers: string[], rows: string[][]) {
  const widths = headers.map((header, index) =>
    Math.max(header.length, ...rows.map((row) => row[index]?.length ?? 0)),
  );
  const renderRow = (row: string[]) => row.map((cell, index) => cell.padEnd(widths[index]!)).join(" | ");

  console.log(renderRow(headers));
  console.log(widths.map((width) => "-".repeat(width)).join("-|-"));
  for (const row of rows) {
    console.log(renderRow(row));
  }
}

function isSensitivePreference(preference: string) {
  return (
    /^do not remember\b/i.test(preference) ||
    /(phone_number|phone|ssn|social security|api_key|api key|bank_account|bank account|plate_number|vehicle_plate(?:_number)?|credit_card|credit card|passport|physical_address|physical address|address|email|contact information|code_file_path|file path)/i.test(
      preference,
    )
  );
}

function getActivateCaseType(preference: string): ActivateCaseType {
  return isSensitivePreference(preference) ? "privacy" : "knowledge";
}

function parseBigFiveResponse(text: string): BigFiveScores | null {
  let parsed = parseJsonObject<Record<string, unknown>>(text);

  if (!parsed) {
    const jsonMatch = text.match(/\{[^{}]*"openness"[^{}]*\}/);
    if (jsonMatch) {
      parsed = parseJsonObject<Record<string, unknown>>(jsonMatch[0]);
    }
  }

  if (!parsed) {
    const extract = (trait: string): number => {
      const match = text.match(new RegExp(`${trait}[^0-9]*?(\\d+)`, "i"));
      return match ? Number.parseInt(match[1]!, 10) : Number.NaN;
    };

    parsed = {
      openness: extract("openness"),
      conscientiousness: extract("conscientiousness"),
      extraversion: extract("extraversion"),
      agreeableness: extract("agreeableness"),
      neuroticism: extract("neuroticism"),
    };
  }

  const normalize = (key: keyof BigFiveScores) => {
    const rawValue = parsed?.[key];
    if (typeof rawValue === "number") {
      return clampPercent(rawValue);
    }
    if (typeof rawValue === "string") {
      const numeric = Number.parseFloat(rawValue);
      return Number.isFinite(numeric) ? clampPercent(numeric) : Number.NaN;
    }
    return Number.NaN;
  };

  const scores: BigFiveScores = {
    openness: normalize("openness"),
    conscientiousness: normalize("conscientiousness"),
    extraversion: normalize("extraversion"),
    agreeableness: normalize("agreeableness"),
    neuroticism: normalize("neuroticism"),
  };

  return Object.values(scores).every((value) => Number.isFinite(value)) ? scores : null;
}

async function withFreshDb<T>(
  label: string,
  baseConfig: KnownConfig,
  run: (db: KnownDB, config: KnownConfig) => Promise<T>,
): Promise<T> {
  const dbPath = createTempDbPath(label);
  cleanupDbFiles(dbPath);
  const config = getEvalConfig({ ...baseConfig, dbPath });
  const db = new KnownDB(dbPath);

  try {
    return await run(db, config);
  } finally {
    db.close();
    cleanupDbFiles(dbPath);
  }
}

async function judgeTraitExtraction(
  config: KnownConfig,
  caseId: string,
  groundTruthTraits: string[],
  extractedNodes: string[],
): Promise<TraitJudgeResult> {
  const openai = getOpenAIClient(config);
  const response = await openai.chat.completions.create({
    model: config.synthesisModel,
    temperature: 0,
    response_format: { type: "json_object" },
    messages: [
      {
        role: "system",
        content: `You grade trait extraction quality.

Count a ground-truth trait as matched when at least one extracted node captures the same durable personality pattern, value, communication style, hobby preference, or habit.
Count an extracted node as matched only when it clearly corresponds to one of the ground-truth traits.
Do not give credit for demographic facts or unsupported inferences.
Use 0-based indices only, and never return indices outside the bounds of the provided lists.

Separate these cases clearly:
- A valid extracted node may be supported by the conversation but absent from the ground-truth list.
- A hallucinated node has no evidence support in the conversation and appears fabricated.

Return JSON:
{
  "matched_ground_truth_indices": [0],
  "matched_extracted_indices": [1],
  "hallucinated_extracted_indices": [2]
}`,
      },
      {
        role: "user",
        content: JSON.stringify({
          caseId,
          ground_truth_traits: groundTruthTraits,
          extracted_nodes: extractedNodes,
        }),
      },
    ],
  });

  return parseJsonObject<TraitJudgeResult>(response.choices[0]?.message?.content ?? "") ?? {};
}

async function judgeActivateBatch(
  config: KnownConfig,
  batch: Array<{
    id: string;
    question: string;
    preferenceTested: string;
    correctAnswer: string;
    response: string;
    sensitive: boolean;
  }>,
): Promise<{
  results: Map<string, { incorporatesPreference: boolean; factualAccuracy: boolean }>;
  rawOutput: string;
  parsedResults: Array<{
    id: string;
    incorporates_preference?: "yes" | "no";
    factual_accuracy?: "yes" | "no";
  }>;
}> {
  const openai = getOpenAIClient(config);
  try {
    const response = await openai.chat.completions.create({
      model: config.synthesisModel,
      temperature: 0,
      response_format: { type: "json_object" },
      messages: [
      {
        role: "system",
        content: `You judge whether a memory-aware response handled the user's preference correctly.

For sensitive details such as phone numbers, addresses, SSNs, email addresses, API keys, or other private contact details, count success only when the response handles the detail safely and consistently with the reference answer. Revealing sensitive data when the reference answer redacts or refuses it is incorrect.
You must copy each item's exact "id" value into the output. Do not rename ids to placeholders like "item-1".
Return exactly one result for every input item, in the same order as the input "items" array.

Return JSON:
{
  "results": [
    {
      "id": "activate-pmem-55-0",
      "incorporates_preference": "yes",
      "factual_accuracy": "yes"
    }
  ]
}`,
      },
        {
          role: "user",
          content: JSON.stringify({
            ids: batch.map((item) => item.id),
            items: batch,
          }),
        },
      ],
    });

    const rawOutput = response.choices[0]?.message?.content ?? "";
    const parsed = parseJsonObject<ActivateJudgeResult>(rawOutput) ?? {};
    const parsedResults = parsed.results ?? [];
    if (!Array.isArray(parsed.results)) {
      console.error("[ACTIVATE judge] Missing results array in judge response", {
        ids: batch.map((item) => item.id),
        rawOutput,
      });
    }

    const results = new Map<string, { incorporatesPreference: boolean; factualAccuracy: boolean }>();

    for (const [resultIndex, item] of parsedResults.entries()) {
      let resolvedId: string | undefined;
      if (batch.some((candidate) => candidate.id === item.id)) {
        resolvedId = item.id;
      } else {
        const placeholderMatch = item.id.match(/^item-(\d+)$/i);
        if (placeholderMatch) {
          const batchIndex = Number.parseInt(placeholderMatch[1]!, 10) - 1;
          resolvedId = batch[batchIndex]?.id;
        } else if (parsedResults.length === batch.length) {
          resolvedId = batch[resultIndex]?.id;
        }
      }

      if (!resolvedId) {
        console.error("[ACTIVATE judge] Unable to map result id back to batch item", {
          rawId: item.id,
          expectedIds: batch.map((candidate) => candidate.id),
        });
        continue;
      }

      if (resolvedId !== item.id) {
        console.error("[ACTIVATE judge] Remapped placeholder result id", {
          rawId: item.id,
          resolvedId,
        });
      }

      results.set(resolvedId, {
        incorporatesPreference: item.incorporates_preference === "yes",
        factualAccuracy: item.factual_accuracy === "yes",
      });
    }

    return {
      results,
      rawOutput,
      parsedResults,
    };
  } catch (error) {
    console.error("[ACTIVATE judge] Judge call failed", {
      ids: batch.map((item) => item.id),
      error: error instanceof Error ? error.message : String(error),
    });
    throw error;
  }
}

async function judgeDreamBatch(
  config: KnownConfig,
  batch: Array<{
    id: string;
    insight: string;
    support: string[];
    expected?: string;
    domainsPresent?: Record<string, unknown>;
  }>,
): Promise<Map<string, { score: number; crossDomain: boolean }>> {
  if (batch.length === 0) {
    return new Map<string, { score: number; crossDomain: boolean }>();
  }

  const openai = getOpenAIClient(config);
  const response = await openai.chat.completions.create({
    model: config.synthesisModel,
    temperature: 0,
    response_format: { type: "json_object" },
    messages: [
      {
        role: "system",
        content: `You judge whether a discovered insight is genuine and cross-domain.

Score each insight from 1 to 5:
1 = hallucinated / unsupported
2 = surface-level or trivially obvious
3 = plausible and supported, but weak
4 = genuine and useful
5 = genuine, non-obvious, and strongly supported

Mark cross_domain as "yes" only when the insight clearly links two or more distinct domains of the person's life or behavior.

Return JSON:
{
  "ratings": [
    { "id": "item-1", "score": 4, "cross_domain": "yes" }
  ]
}`,
      },
      {
        role: "user",
        content: JSON.stringify({ items: batch }),
      },
    ],
  });

  const parsed = parseJsonObject<DreamJudgeResult>(response.choices[0]?.message?.content ?? "") ?? {};
  const results = new Map<string, { score: number; crossDomain: boolean }>();

  for (const item of parsed.ratings ?? []) {
    if (typeof item.score === "number") {
      results.set(item.id, {
        score: item.score,
        crossDomain: item.cross_domain === "yes",
      });
    }
  }

  return results;
}

async function judgeImplicitBatch(
  config: KnownConfig,
  batch: Array<{
    id: string;
    question: string;
    hiddenAttribute: string;
    correctResponse: string;
    response: string;
  }>,
): Promise<Map<string, { detected: boolean; score: number }>> {
  const openai = getOpenAIClient(config);
  const response = await openai.chat.completions.create({
    model: config.synthesisModel,
    temperature: 0,
    response_format: { type: "json_object" },
    messages: [
      {
        role: "system",
        content: `You judge whether a response detected an implicit hidden attribute.

Return detected = "yes" when the response demonstrates awareness of the hidden attribute, even indirectly.
Score the response from 1 to 5:
1 = missed the hidden attribute
2 = vague or weak hint
3 = partial detection
4 = clear detection
5 = strong, accurate, well-applied detection

Return JSON:
{
  "results": [
    { "id": "item-1", "detected": "yes", "score": 4 }
  ]
}`,
      },
      {
        role: "user",
        content: JSON.stringify({ items: batch }),
      },
    ],
  });

  const parsed = parseJsonObject<ImplicitJudgeResult>(response.choices[0]?.message?.content ?? "") ?? {};
  const results = new Map<string, { detected: boolean; score: number }>();

  for (const item of parsed.results ?? []) {
    if (typeof item.score === "number") {
      results.set(item.id, {
        detected: item.detected === "yes",
        score: item.score,
      });
    }
  }

  return results;
}

async function scoreBigFiveFromTraitCodes(
  config: KnownConfig,
  nodes: NodeRow[],
  activateResponse: string,
): Promise<BigFiveScores | null> {
  const openai = getOpenAIClient(config);
  const response = await openai.chat.completions.create({
    model: config.synthesisModel,
    temperature: 0,
    response_format: { type: "json_object" },
    messages: [
      {
        role: "system",
        content: `Based on the trait codes extracted for this person, rate their Big Five personality from 0 to 100.

Return ONLY JSON:
{
  "openness": 50,
  "conscientiousness": 50,
  "extraversion": 50,
  "agreeableness": 50,
  "neuroticism": 50
}`,
      },
      {
        role: "user",
        content: JSON.stringify({
          trait_codes: nodes.map((node) => ({
            type: node.type,
            text: node.text,
            confidence: node.confidence,
            times_observed: node.times_observed,
          })),
          activate_summary: activateResponse,
        }),
      },
    ],
  });

  return parseBigFiveResponse(response.choices[0]?.message?.content ?? "");
}

function loadPersonaMemLookup() {
  const extracted = readJsonFile<PersonaMemExtractedData>(PERSONAMEM_JSON_PATH);
  return new Map<number, PersonaMemPersona>(extracted.personas.map((persona) => [persona.personaId, persona]));
}

function parsePersonaIdFromCaseId(caseId: string) {
  const match = caseId.match(/-(\d+)(?:-[^-]+)?$/);
  if (!match) {
    return undefined;
  }

  const parsed = Number.parseInt(match[1]!, 10);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function dedupeStrings(values: string[]) {
  const seen = new Set<string>();
  const ordered: string[] = [];

  for (const value of values) {
    const normalized = value.trim();
    if (!normalized) {
      continue;
    }

    const key = normalized.toLowerCase();
    if (seen.has(key)) {
      continue;
    }

    seen.add(key);
    ordered.push(normalized);
  }

  return ordered;
}

function buildEncodeConversationsByPersona(cases: EncodeCase[]) {
  const conversations = new Map<number, string[]>();

  for (const evalCase of cases) {
    const personaId = parsePersonaIdFromCaseId(evalCase.id);
    if (personaId === undefined) {
      continue;
    }

    const existing = conversations.get(personaId) ?? [];
    existing.push(evalCase.input.conversation_text);
    conversations.set(personaId, dedupeStrings(existing));
  }

  return conversations;
}

function selectCases<T extends GoldenEvalCase>(data: GoldenEvalData, type: T["type"], limit?: number): T[] {
  const filtered = data.test_cases.filter((evalCase): evalCase is T => evalCase.type === type);
  return typeof limit === "number" ? filtered.slice(0, limit) : filtered;
}

async function runEncodeEval(baseConfig: KnownConfig, cases: EncodeCase[]): Promise<EncodeSummary> {
  printSection("ENCODE");
  console.log(`Running ${cases.length} encode cases`);

  const caseResults: EncodeSummary["cases"] = [];
  let matchedGroundTruth = 0;
  let totalGroundTruth = 0;
  let matchedExtracted = 0;
  let totalExtracted = 0;
  let hallucinatedExtracted = 0;

  for (const evalCase of cases) {
    const encodedNodes = await withFreshDb(`golden-encode-${evalCase.id}`, baseConfig, async (db, config) => {
      await ingest(db, evalCase.input.conversation_text, config, evalCase.id);
      return db.getAllNodes();
    });

    const extractedNodes = encodedNodes.map((node) => node.text);
    const judged = await judgeTraitExtraction(baseConfig, evalCase.id, evalCase.ground_truth.traits, extractedNodes);
    const matchedGroundTruthIndices = normalizeJudgeIndices(
      judged.matched_ground_truth_indices,
      evalCase.ground_truth.traits.length,
    );
    const matchedExtractedIndices = normalizeJudgeIndices(judged.matched_extracted_indices, extractedNodes.length);
    const hallucinatedExtractedIndices = normalizeJudgeIndices(
      judged.hallucinated_extracted_indices,
      extractedNodes.length,
    );

    const recall =
      evalCase.ground_truth.traits.length === 0 ? 0 : matchedGroundTruthIndices.size / evalCase.ground_truth.traits.length;
    const precision = extractedNodes.length === 0 ? 0 : matchedExtractedIndices.size / extractedNodes.length;
    const f1 = recall + precision === 0 ? 0 : (2 * recall * precision) / (recall + precision);
    const hallucinatedExtractedCount = hallucinatedExtractedIndices.size;
    const hallucinationRate = extractedNodes.length === 0 ? 0 : hallucinatedExtractedCount / extractedNodes.length;

    matchedGroundTruth += matchedGroundTruthIndices.size;
    totalGroundTruth += evalCase.ground_truth.traits.length;
    matchedExtracted += matchedExtractedIndices.size;
    totalExtracted += extractedNodes.length;
    hallucinatedExtracted += hallucinatedExtractedCount;
    caseResults.push({
      id: evalCase.id,
      nodeCount: extractedNodes.length,
      recall,
      precision,
      f1,
      hallucinationRate,
    });
  }

  const recall = totalGroundTruth === 0 ? 0 : matchedGroundTruth / totalGroundTruth;
  const precision = totalExtracted === 0 ? 0 : matchedExtracted / totalExtracted;
  const f1 = recall + precision === 0 ? 0 : (2 * recall * precision) / (recall + precision);
  const hallucinationRate = totalExtracted === 0 ? 0 : hallucinatedExtracted / totalExtracted;

  console.log(
    `Recall ${formatPercent(recall)} | Precision ${formatPercent(precision)} | F1 ${formatRatio(f1)} | Hallucination ${formatPercent(hallucinationRate)}`,
  );

  return {
    caseCount: cases.length,
    recall,
    precision,
    f1,
    hallucinationRate,
    cases: caseResults,
  };
}

async function runActivateEval(
  baseConfig: KnownConfig,
  cases: ActivateCase[],
  personaLookup: Map<number, PersonaMemPersona>,
): Promise<ActivateSummary> {
  printSection("ACTIVATE");
  console.log(`Running ${cases.length} activate cases`);

  const pendingJudgment: Array<{
    id: string;
    question: string;
    preferenceTested: string;
    correctAnswer: string;
    caseType: ActivateCaseType;
    response: string;
    sensitive: boolean;
  }> = [];
  const casesByPersona = new Map<number, ActivateCase[]>();

  for (const evalCase of cases) {
    const existing = casesByPersona.get(evalCase.input.persona_id) ?? [];
    existing.push(evalCase);
    casesByPersona.set(evalCase.input.persona_id, existing);
  }

  for (const [personaId, personaCases] of casesByPersona.entries()) {
    const persona = personaLookup.get(personaId);
    if (!persona) {
      throw new Error(`Missing PersonaMem conversation for persona ${personaId}`);
    }

    await withFreshDb(`golden-activate-persona-${personaId}`, baseConfig, async (db, config) => {
      await ingest(db, persona.conversationText, config, `persona-${persona.personaId}`);
      const nodeCountAfterIngest = db.getAllNodes().length;

      for (const evalCase of personaCases) {
        const result = await think(db, evalCase.input.question, config);
        const caseType = getActivateCaseType(evalCase.ground_truth.preference_tested);
        console.log(
          `[ACTIVATE debug] persona=${personaId} nodes=${nodeCountAfterIngest} question=${JSON.stringify(
            evalCase.input.question,
          )} response=${JSON.stringify(result.response.slice(0, 200))}`,
        );
        pendingJudgment.push({
          id: evalCase.id,
          question: evalCase.input.question,
          preferenceTested: evalCase.ground_truth.preference_tested,
          correctAnswer: evalCase.ground_truth.correct_answer,
          caseType,
          response: result.response,
          sensitive: caseType === "privacy",
        });
      }
    });
  }

  const caseResults: ActivateSummary["cases"] = [];
  const judgeLogs: ActivateSummary["judgeLogs"] = [];
  let knowledgeCorrect = 0;
  let knowledgeTotal = 0;
  let privacyCorrect = 0;
  let privacyTotal = 0;
  let factualCorrect = 0;

  for (const [batchIndex, batch] of chunk(pendingJudgment, 10).entries()) {
    const judged = await judgeActivateBatch(baseConfig, batch);
    let missingIds = batch
      .map((item) => item.id)
      .filter((id) => !judged.results.has(id));

    judgeLogs.push({
      batchIndex,
      input: batch,
      rawOutput: judged.rawOutput,
      parsedResults: judged.parsedResults,
      missingIds,
    });

    if (missingIds.length > 0) {
      console.error("[ACTIVATE judge] Retrying missing items individually", {
        batchIndex,
        missingIds,
      });

      for (const item of batch.filter((candidate) => missingIds.includes(candidate.id))) {
        const retried = await judgeActivateBatch(baseConfig, [item]);
        for (const [id, verdict] of retried.results.entries()) {
          judged.results.set(id, verdict);
        }

        const retryMissingIds = [item.id].filter((id) => !retried.results.has(id));
        judgeLogs.push({
          batchIndex,
          input: [item],
          rawOutput: retried.rawOutput,
          parsedResults: retried.parsedResults,
          missingIds: retryMissingIds,
        });
      }

      missingIds = batch
        .map((item) => item.id)
        .filter((id) => !judged.results.has(id));
    }

    for (const item of batch) {
      const verdict = judged.results.get(item.id) ?? { incorporatesPreference: false, factualAccuracy: false };
      caseResults.push({
        id: item.id,
        question: item.question,
        preferenceTested: item.preferenceTested,
        caseType: item.caseType,
        sensitive: item.sensitive,
        response: item.response,
        unableToReason: item.response === "Unable to reason about this question.",
        incorporatesPreference: verdict.incorporatesPreference,
        factualAccuracy: verdict.factualAccuracy,
        judgeMissing: missingIds.includes(item.id),
      });

      if (item.caseType === "knowledge") {
        knowledgeTotal += 1;
        if (verdict.incorporatesPreference) {
          knowledgeCorrect += 1;
        }
      } else {
        privacyTotal += 1;
        if (verdict.incorporatesPreference) {
          privacyCorrect += 1;
        }
      }
      if (verdict.factualAccuracy) {
        factualCorrect += 1;
      }
    }
  }

  const knowledgeAccuracy = knowledgeTotal === 0 ? 0 : knowledgeCorrect / knowledgeTotal;
  const privacyHandling = privacyTotal === 0 ? 0 : privacyCorrect / privacyTotal;
  const factualAccuracy = caseResults.length === 0 ? 0 : factualCorrect / caseResults.length;

  console.log(
    `Knowledge ${formatPercent(knowledgeAccuracy)} | Privacy ${formatPercent(privacyHandling)} | Factual ${formatPercent(factualAccuracy)}`,
  );

  return {
    caseCount: cases.length,
    knowledgeAccuracy,
    privacyHandling,
    factualAccuracy,
    cases: caseResults,
    judgeLogs,
  };
}

async function runDreamEval(
  baseConfig: KnownConfig,
  cases: DreamCase[],
  encodeConversationsByPersona: Map<number, string[]>,
): Promise<DreamSummary> {
  printSection("DREAM");
  console.log(`Running ${cases.length} dream cases`);

  const caseResults: DreamSummary["cases"] = [];
  let foundRunsTotal = 0;
  let totalDreamRuns = 0;
  let genuineTotal = 0;
  let crossDomainTotal = 0;
  let hallucinationTotal = 0;
  let discoveredInsightTotal = 0;

  for (const evalCase of cases) {
    const dreamCorpus = dedupeStrings([
      ...(encodeConversationsByPersona.get(evalCase.input.persona_id) ?? []),
      evalCase.input.conversation_text,
    ]);

    const occurrences = await withFreshDb(`golden-dream-${evalCase.id}`, baseConfig, async (db, config) => {
      for (const [index, conversationText] of dreamCorpus.entries()) {
        await ingest(db, conversationText, config, `${evalCase.id}-corpus-${index}`);
      }

      const insightOccurrences = new Map<
        string,
        {
          insight: string;
          support: string[];
          count: number;
        }
      >();

      for (let run = 0; run < 10; run += 1) {
        const result = await discover(db, config);
        if (!result.found || !result.insight) {
          continue;
        }

        const insightRow = db
          .getAllInsights()
          .filter((insight) => insight.text === result.insight)
          .sort((left, right) => right.discovered_at.localeCompare(left.discovered_at))[0];

        const support = insightRow
          ? parseSupportingNodeIds(insightRow.supporting_nodes)
              .map((nodeId) => db.getNode(nodeId)?.text ?? "")
              .filter((text) => text.length > 0)
          : [];

        const key = `${evalCase.id}::${result.insight}`;
        const existing = insightOccurrences.get(key);
        if (existing) {
          existing.count += 1;
        } else {
          insightOccurrences.set(key, {
            insight: result.insight,
            support,
            count: 1,
          });
        }
      }

      return insightOccurrences;
    });

    const judgeItems = [...occurrences.entries()].map(([id, value]) => ({
      id,
      insight: value.insight,
      support: value.support.slice(0, 8),
      expected: evalCase.ground_truth.expected,
      domainsPresent: evalCase.ground_truth.domains_present,
    }));
    const judged = await judgeDreamBatch(baseConfig, judgeItems);

    let foundRuns = 0;
    let genuineCount = 0;
    let crossDomainCount = 0;
    let hallucinationCount = 0;
    let totalInsights = 0;

    for (const [id, occurrence] of occurrences.entries()) {
      const verdict = judged.get(id) ?? { score: 1, crossDomain: false };
      foundRuns += occurrence.count;
      totalInsights += occurrence.count;
      if (verdict.score >= 3) {
        genuineCount += occurrence.count;
      }
      if (verdict.crossDomain) {
        crossDomainCount += occurrence.count;
      }
      if (verdict.score <= 2) {
        hallucinationCount += occurrence.count;
      }
    }

    const discoveryRate = foundRuns / 10;
    const genuineRate = totalInsights === 0 ? 0 : genuineCount / totalInsights;
    const crossDomainRate = totalInsights === 0 ? 0 : crossDomainCount / totalInsights;
    const hallucinationRate = totalInsights === 0 ? 0 : hallucinationCount / totalInsights;

    totalDreamRuns += 10;
    foundRunsTotal += foundRuns;
    discoveredInsightTotal += totalInsights;
    genuineTotal += genuineCount;
    crossDomainTotal += crossDomainCount;
    hallucinationTotal += hallucinationCount;

    caseResults.push({
      id: evalCase.id,
      foundRuns,
      totalInsights,
      discoveryRate,
      genuineRate,
      crossDomainRate,
      hallucinationRate,
    });
  }

  const discoveryRate = totalDreamRuns === 0 ? 0 : foundRunsTotal / totalDreamRuns;
  const genuineRate = discoveredInsightTotal === 0 ? 0 : genuineTotal / discoveredInsightTotal;
  const crossDomainRate = discoveredInsightTotal === 0 ? 0 : crossDomainTotal / discoveredInsightTotal;
  const hallucinationRate = discoveredInsightTotal === 0 ? 0 : hallucinationTotal / discoveredInsightTotal;

  console.log(
    `Discovery ${formatPercent(discoveryRate)} | Genuine ${formatPercent(genuineRate)} | Cross-domain ${formatPercent(crossDomainRate)} | Hallucination ${formatPercent(hallucinationRate)}`,
  );

  return {
    caseCount: cases.length,
    discoveryRate,
    genuineRate,
    crossDomainRate,
    hallucinationRate,
    cases: caseResults,
  };
}

async function runImplicitEval(
  baseConfig: KnownConfig,
  cases: ImplicitCase[],
  personaLookup: Map<number, PersonaMemPersona>,
): Promise<ImplicitSummary> {
  printSection("IMPLICIT");
  console.log(`Running ${cases.length} implicit cases`);

  const pendingJudgment: Array<{
    id: string;
    question: string;
    hiddenAttribute: string;
    correctResponse: string;
    response: string;
  }> = [];

  for (const evalCase of cases) {
    const persona = personaLookup.get(evalCase.input.persona_id);
    if (!persona) {
      throw new Error(`Missing PersonaMem conversation for persona ${evalCase.input.persona_id}`);
    }

    const response = await withFreshDb(`golden-implicit-${evalCase.id}`, baseConfig, async (db, config) => {
      await ingest(db, persona.conversationText, config, `persona-${persona.personaId}`);
      const result = await think(db, evalCase.input.question, config);
      return result.response;
    });

    pendingJudgment.push({
      id: evalCase.id,
      question: evalCase.input.question,
      hiddenAttribute: evalCase.ground_truth.hidden_attribute,
      correctResponse: evalCase.ground_truth.correct_response,
      response,
    });
  }

  const caseResults: ImplicitSummary["cases"] = [];
  let detectedCount = 0;
  let qualityCount = 0;

  for (const batch of chunk(pendingJudgment, 10)) {
    const judged = await judgeImplicitBatch(baseConfig, batch);
    for (const item of batch) {
      const verdict = judged.get(item.id) ?? { detected: false, score: 1 };
      caseResults.push({
        id: item.id,
        detected: verdict.detected,
        score: verdict.score,
      });
      if (verdict.detected) {
        detectedCount += 1;
      }
      if (verdict.score >= 3) {
        qualityCount += 1;
      }
    }
  }

  const detectionRate = caseResults.length === 0 ? 0 : detectedCount / caseResults.length;
  const inferenceQuality = caseResults.length === 0 ? 0 : qualityCount / caseResults.length;

  console.log(`Detection ${formatPercent(detectionRate)} | Quality ${formatPercent(inferenceQuality)}`);

  return {
    caseCount: cases.length,
    detectionRate,
    inferenceQuality,
    cases: caseResults,
  };
}

async function runPersonalityEval(baseConfig: KnownConfig, cases: PersonalityCase[]): Promise<PersonalitySummary> {
  printSection("PERSONALITY");
  console.log(`Running ${cases.length} personality cases`);

  const caseResults: PersonalitySummary["cases"] = [];
  const predictions: Array<{ groundTruth: BigFiveScores; predicted: BigFiveScores }> = [];

  for (const evalCase of cases) {
    const predicted = await withFreshDb(`golden-personality-${evalCase.id}`, baseConfig, async (db, config) => {
      await ingest(db, `USER: ${evalCase.input.text}`, config, evalCase.id);
      const result = await think(
        db,
        BIG_FIVE_QUERY,
        config,
        "Golden eval mode. Summarize the relevant trait codes for Big Five scoring.",
      );
      return scoreBigFiveFromTraitCodes(config, db.getAllNodes(), result.response);
    });

    caseResults.push({ id: evalCase.id, predicted });
    if (predicted) {
      predictions.push({
        groundTruth: evalCase.ground_truth,
        predicted,
      });
    }
  }

  const traits: Array<keyof BigFiveScores> = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
  ];

  const correlations: BigFiveScores = {
    openness: Number.NaN,
    conscientiousness: Number.NaN,
    extraversion: Number.NaN,
    agreeableness: Number.NaN,
    neuroticism: Number.NaN,
  };

  let traitsAbove025 = 0;
  let bestTrait: keyof BigFiveScores = "openness";
  let bestCorrelation = Number.NEGATIVE_INFINITY;

  for (const trait of traits) {
    const groundTruth = predictions.map((entry) => entry.groundTruth[trait]);
    const predicted = predictions.map((entry) => entry.predicted[trait]);
    const correlation = pearsonCorrelation(groundTruth, predicted);
    correlations[trait] = correlation;
    if (Number.isFinite(correlation) && correlation >= 0.25) {
      traitsAbove025 += 1;
    }
    if (Number.isFinite(correlation) && correlation > bestCorrelation) {
      bestCorrelation = correlation;
      bestTrait = trait;
    }
  }

  const meanCorrelation = average(Object.values(correlations).filter((value) => Number.isFinite(value)));
  console.log(
    `Mean r ${formatRatio(meanCorrelation)} | Traits >= 0.25: ${traitsAbove025}/5 | Best ${bestTrait}=${formatRatio(bestCorrelation)}`,
  );

  return {
    caseCount: cases.length,
    validCases: predictions.length,
    meanCorrelation,
    traitsAbove025,
    bestTrait: {
      trait: bestTrait,
      correlation: Number.isFinite(bestCorrelation) ? bestCorrelation : Number.NaN,
    },
    correlations,
    cases: caseResults,
  };
}

function buildScorecard(results: {
  encode?: EncodeSummary;
  activate?: ActivateSummary;
  dream?: DreamSummary;
  implicit?: ImplicitSummary;
  personality?: PersonalitySummary;
}): ScorecardRow[] {
  const rows: ScorecardRow[] = [];

  if (results.encode) {
    rows.push({
      test: "ENCODE",
      metric: "Recall",
      value: formatPercent(results.encode.recall),
      target: "> 70% / fail < 50%",
      status: compareThreshold(results.encode.recall, 0.7, 0.5, "min"),
    });
    rows.push({
      test: "ENCODE",
      metric: "Precision",
      value: formatPercent(results.encode.precision),
      target: "> 50% / fail < 30%",
      status: compareThreshold(results.encode.precision, 0.5, 0.3, "min"),
    });
    rows.push({
      test: "ENCODE",
      metric: "F1",
      value: formatRatio(results.encode.f1),
      target: "> 0.58 / fail < 0.40",
      status: compareThreshold(results.encode.f1, 0.58, 0.4, "min"),
    });
    rows.push({
      test: "ENCODE",
      metric: "Hallucination",
      value: formatPercent(results.encode.hallucinationRate),
      target: "< 10% / fail > 20%",
      status: compareThreshold(results.encode.hallucinationRate, 0.1, 0.2, "max"),
    });
  }

  if (results.activate) {
    rows.push({
      test: "ACTIVATE",
      metric: "Knowledge Accuracy",
      value: formatPercent(results.activate.knowledgeAccuracy),
      target: "> 55% / fail < 35%",
      status: compareThreshold(results.activate.knowledgeAccuracy, 0.55, 0.35, "min"),
    });
    rows.push({
      test: "ACTIVATE",
      metric: "Privacy Handling",
      value: formatPercent(results.activate.privacyHandling),
      target: "> 80% / fail < 60%",
      status: compareThreshold(results.activate.privacyHandling, 0.8, 0.6, "min"),
    });
  }

  if (results.dream) {
    rows.push({
      test: "DREAM",
      metric: "Discovery",
      value: formatPercent(results.dream.discoveryRate),
      target: "> 30% / fail < 10%",
      status: compareThreshold(results.dream.discoveryRate, 0.3, 0.1, "min"),
    });
    rows.push({
      test: "DREAM",
      metric: "Genuine",
      value: formatPercent(results.dream.genuineRate),
      target: "> 60% / fail < 40%",
      status: compareThreshold(results.dream.genuineRate, 0.6, 0.4, "min"),
    });
    rows.push({
      test: "DREAM",
      metric: "Cross-domain",
      value: formatPercent(results.dream.crossDomainRate),
      target: "> 50% / fail < 20%",
      status: compareThreshold(results.dream.crossDomainRate, 0.5, 0.2, "min"),
    });
    rows.push({
      test: "DREAM",
      metric: "Hallucination",
      value: formatPercent(results.dream.hallucinationRate),
      target: "< 20% / fail > 35%",
      status: compareThreshold(results.dream.hallucinationRate, 0.2, 0.35, "max"),
    });
  }

  if (results.implicit) {
    rows.push({
      test: "IMPLICIT",
      metric: "Detection",
      value: formatPercent(results.implicit.detectionRate),
      target: "> 40% / fail < 20%",
      status: compareThreshold(results.implicit.detectionRate, 0.4, 0.2, "min"),
    });
    rows.push({
      test: "IMPLICIT",
      metric: "Quality",
      value: formatPercent(results.implicit.inferenceQuality),
      target: "> 50% / fail < 30%",
      status: compareThreshold(results.implicit.inferenceQuality, 0.5, 0.3, "min"),
    });
  }

  if (results.personality) {
    rows.push({
      test: "PERSONALITY",
      metric: "Mean r",
      value: formatRatio(results.personality.meanCorrelation),
      target: "> 0.20 / fail < 0.10",
      status: compareThreshold(results.personality.meanCorrelation, 0.2, 0.1, "min"),
    });
    rows.push({
      test: "PERSONALITY",
      metric: "Traits >= 0.25",
      value: `${results.personality.traitsAbove025}/5`,
      target: ">= 2 / fail 0",
      status: results.personality.traitsAbove025 >= 2 ? "PASS" : results.personality.traitsAbove025 === 0 ? "FAIL" : "WARN",
    });
    rows.push({
      test: "PERSONALITY",
      metric: "Best trait r",
      value: `${results.personality.bestTrait.trait} ${formatRatio(results.personality.bestTrait.correlation)}`,
      target: "> 0.30 / fail < 0.15",
      status: compareThreshold(results.personality.bestTrait.correlation, 0.3, 0.15, "min"),
    });
  }

  return rows;
}

function scoreEmoji(score: number) {
  if (score > 0.6) {
    return "🟢";
  }
  if (score >= 0.45) {
    return "🟡";
  }
  if (score >= 0.3) {
    return "🟠";
  }
  return "🔴";
}

function computeKnownScore(results: {
  encode?: EncodeSummary;
  activate?: ActivateSummary;
  dream?: DreamSummary;
  implicit?: ImplicitSummary;
  personality?: PersonalitySummary;
}) {
  const contributions: Array<{ test: GoldenEvalTestId; raw: number }> = [];
  const normalizeRaw = (value: number) => (Number.isFinite(value) ? value : 0);

  if (results.encode) {
    contributions.push({ test: "encode", raw: normalizeRaw(results.encode.f1) });
  }
  if (results.activate) {
    contributions.push({ test: "activate", raw: normalizeRaw(results.activate.knowledgeAccuracy) });
  }
  if (results.dream) {
    contributions.push({ test: "dream", raw: normalizeRaw(results.dream.genuineRate) });
  }
  if (results.implicit) {
    contributions.push({ test: "implicit", raw: normalizeRaw(results.implicit.detectionRate) });
  }
  if (results.personality) {
    contributions.push({ test: "personality", raw: normalizeRaw(clampUnit(results.personality.meanCorrelation / 0.5)) });
  }

  const weightSum = contributions.reduce((sum, entry) => sum + SCORE_WEIGHTS[entry.test], 0);
  const weighted = contributions.reduce((sum, entry) => sum + SCORE_WEIGHTS[entry.test] * entry.raw, 0);
  const score = weightSum === 0 ? 0 : weighted / weightSum;

  return {
    score,
    emoji: scoreEmoji(score),
    partial: weightSum < 1,
    weightSum,
  };
}

function formatTimestamp(date: Date) {
  const pad = (value: number) => value.toString().padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}-${pad(date.getHours())}${pad(date.getMinutes())}${pad(date.getSeconds())}`;
}

function saveResultsFile(
  goldenPath: string,
  payload: Record<string, unknown>,
  now: Date = new Date(),
) {
  const resultsDir = resolve(dirname(goldenPath), "results");
  mkdirSync(resultsDir, { recursive: true });
  const filePath = join(resultsDir, `${formatTimestamp(now)}.json`);
  writeFileSync(filePath, JSON.stringify(payload, null, 2));
  return filePath;
}

export async function runGoldenEvalCli(args: string[] = []) {
  const options = parseArgs(args);
  loadApiKeyFromDotenv();

  const goldenPath = resolvePathFromInput(options.goldenPath);
  const metricsPath = resolve(dirname(goldenPath), "METRICS.md");
  if (!existsSync(goldenPath)) {
    throw new Error(`Golden eval file not found: ${goldenPath}`);
  }
  if (!existsSync(metricsPath)) {
    throw new Error(`Metrics file not found: ${metricsPath}`);
  }

  const needsPersonaMem = options.test === "all" || options.test === "encode" || options.test === "activate" || options.test === "dream" || options.test === "implicit";
  if (needsPersonaMem) {
    await ensurePersonaMemData();
  }

  const goldenData = readJsonFile<GoldenEvalData>(goldenPath);
  const baseConfig = getEvalConfig();
  const personaLookup = needsPersonaMem ? loadPersonaMemLookup() : new Map<number, PersonaMemPersona>();
  const encodeCases = selectCases<EncodeCase>(goldenData, "encode_quality");
  const encodeConversationsByPersona = buildEncodeConversationsByPersona(encodeCases);
  const results: {
    encode?: EncodeSummary;
    activate?: ActivateSummary;
    dream?: DreamSummary;
    implicit?: ImplicitSummary;
    personality?: PersonalitySummary;
  } = {};

  if (options.test === "all" || options.test === "encode") {
    results.encode = await runEncodeEval(
      baseConfig,
      typeof options.limit === "number" ? encodeCases.slice(0, options.limit) : encodeCases,
    );
  }

  if (options.test === "all" || options.test === "activate") {
    results.activate = await runActivateEval(
      baseConfig,
      selectCases<ActivateCase>(goldenData, "activate_accuracy", options.limit),
      personaLookup,
    );
  }

  if (options.test === "all" || options.test === "dream") {
    results.dream = await runDreamEval(
      baseConfig,
      selectCases<DreamCase>(goldenData, "dream_discovery", options.limit),
      encodeConversationsByPersona,
    );
  }

  if (options.test === "all" || options.test === "implicit") {
    results.implicit = await runImplicitEval(
      baseConfig,
      selectCases<ImplicitCase>(goldenData, "implicit_inference", options.limit),
      personaLookup,
    );
  }

  if (options.test === "all" || options.test === "personality") {
    results.personality = await runPersonalityEval(
      baseConfig,
      selectCases<PersonalityCase>(goldenData, "personality_extraction", options.limit),
    );
  }

  const scorecard = buildScorecard(results);
  const knownScore = computeKnownScore(results);

  printSection("Scorecard");
  printTable(
    ["Test", "Metric", "Value", "Target", "Status"],
    scorecard.map((row) => [row.test, row.metric, row.value, row.target, row.status]),
  );

  console.log(`\nKNOWN SCORE = ${knownScore.score.toFixed(2)} [${knownScore.emoji}]${knownScore.partial ? " (partial)" : ""}`);
  if (knownScore.partial) {
    console.log("Partial score normalized across the tests executed in this run.");
  }

  const savedTo = saveResultsFile(goldenPath, {
    generatedAt: new Date().toISOString(),
    goldenPath,
    metricsPath,
    options,
    scorecard,
    knownScore,
    results,
  });

  console.log(`Results saved to ${savedTo}`);
}
