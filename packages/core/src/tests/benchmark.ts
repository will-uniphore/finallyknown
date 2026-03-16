import { execFile } from "node:child_process";
import dotenv from "dotenv";
import { existsSync, readFileSync, rmSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import type { KnownConfig } from "../config.js";
import { getConfig } from "../config.js";
import { KnownDB, type InsightRow, type NodeRow } from "../db.js";
import { discover } from "../discover.js";
import { ingest } from "../ingest.js";
import { getOpenAIClient } from "../openai.js";
import { think } from "../think.js";

type BenchmarkTestId = "1a" | "2a" | "3a" | "4a";

interface BenchmarkOptions {
  test: BenchmarkTestId | "all";
  personas: number;
  pandoraUsers: number;
}

interface PersonaMemQAPair {
  userQuery: string;
  correctAnswer: string;
  preference: string;
  updated: boolean;
  prevPref: string | null;
}

interface PersonaMemPersona {
  personaId: number;
  expandedPersona: string;
  groundTruthTraits: string[];
  snippets: string[];
  conversationText: string;
  uniqueSnippetChars: number;
  qaPairs: PersonaMemQAPair[];
}

interface PersonaMemExtractedData {
  personas: PersonaMemPersona[];
}

interface BigFiveScores {
  openness: number;
  conscientiousness: number;
  extraversion: number;
  agreeableness: number;
  neuroticism: number;
}

interface PandoraUser {
  userId: string;
  text: string;
  textLength: number;
  groundTruth: BigFiveScores;
}

interface PandoraExtractedData {
  users: PandoraUser[];
}

interface PreparedPersona {
  persona: PersonaMemPersona;
  db: KnownDB;
  dbPath: string;
  config: KnownConfig;
  ingestResult: { nodesCreated: number; edgesCreated: number };
  encodedNodes: NodeRow[];
}

interface TraitJudgeResult {
  matched_ground_truth_indices?: number[];
  matched_extracted_indices?: number[];
}

interface BatchJudgeResult {
  results?: Array<{
    id: string;
    verdict?: "yes" | "no";
  }>;
}

interface DreamJudgeResult {
  ratings?: Array<{
    id: string;
    score?: number;
  }>;
}

interface SummaryRow {
  test: string;
  metric: string;
  value: string;
  target: string;
  status: "PASS" | "WARN" | "FAIL";
}

interface Test1AResult {
  recall: number;
  precision: number;
  f1: number;
  matchedGroundTruth: number;
  totalGroundTruth: number;
  matchedExtracted: number;
  totalExtracted: number;
}

interface Test2AResult {
  accuracy: number;
  correct: number;
  total: number;
}

interface Test3AResult {
  genuineRate: number;
  genuineCount: number;
  hallucinationRate: number;
  hallucinationCount: number;
  totalInsights: number;
}

interface Test4AResult {
  meanCorrelation: number;
  traitsAboveTarget: number;
  correlations: BigFiveScores;
  validUsers: number;
}

const PERSONAMEM_JSON_PATH = "/tmp/known-test-data/extracted/personamem-v2-benchmark.json";
const PANDORA_JSON_PATH = "/tmp/known-test-data/extracted/pandora-test.json";
const TEST_IDS: BenchmarkTestId[] = ["1a", "2a", "3a", "4a"];
const QA_JUDGE_BATCH_SIZE = 10;
const BIG_FIVE_QUERY =
  'Rate this person on Big Five personality traits from 0-100. Include scores as: openness: X, conscientiousness: X, extraversion: X, agreeableness: X, neuroticism: X';

function usage() {
  console.log(`Usage:
  known benchmark [--test 1a|2a|3a|4a|all] [--personas 5] [--pandora-users 20]`);
}

function parseNumberFlag(value: string | undefined, fallback: number, label: string): number {
  if (!value) {
    return fallback;
  }

  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`Invalid ${label}: ${value}`);
  }

  return parsed;
}

function parseArgs(args: string[]): BenchmarkOptions {
  let test: BenchmarkOptions["test"] = "all";
  let personas = 5;
  let pandoraUsers = 20;

  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    switch (arg) {
      case "--test": {
        const value = args[index + 1];
        if (!value || (value !== "all" && !TEST_IDS.includes(value as BenchmarkTestId))) {
          throw new Error(`Invalid --test value: ${value ?? "(missing)"}`);
        }
        test = value as BenchmarkOptions["test"];
        index += 1;
        break;
      }

      case "--personas":
        personas = parseNumberFlag(args[index + 1], personas, "--personas");
        index += 1;
        break;

      case "--pandora-users":
        pandoraUsers = parseNumberFlag(args[index + 1], pandoraUsers, "--pandora-users");
        index += 1;
        break;

      case "--help":
      case "-h":
        usage();
        process.exit(0);

      default:
        throw new Error(`Unknown benchmark flag: ${arg}`);
    }
  }

  return { test, personas, pandoraUsers };
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

  throw new Error("Unable to resolve the packages/core root for the benchmark runner.");
}

function extractScriptPath() {
  return join(packageRoot(), "src", "tests", "extract-data.py");
}

function loadApiKeyFromDotenv() {
  dotenv.config({ path: join(homedir(), ".known", ".env") });
}

function getBenchmarkConfig(overrides?: Partial<KnownConfig>): KnownConfig {
  const config = getConfig(overrides);
  if (!config.openaiApiKey) {
    throw new Error(`Missing OPENAI_API_KEY. Set it in ${join(homedir(), ".known", ".env")} or the environment.`);
  }

  return config;
}

function createTempDbPath(label: string) {
  const safeLabel = label.replace(/[^a-z0-9_-]+/gi, "-").replace(/^-+|-+$/g, "").toLowerCase() || "benchmark";
  return join(tmpdir(), `known-test-${safeLabel}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}.db`);
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

async function ensureExtractedData(needsPersonas: boolean, needsPandora: boolean) {
  const missingPersonaMem = needsPersonas && !existsSync(PERSONAMEM_JSON_PATH);
  const missingPandora = needsPandora && !existsSync(PANDORA_JSON_PATH);

  if (!missingPersonaMem && !missingPandora) {
    return;
  }

  console.log("Preparing extracted benchmark data...");
  await runExtractor();
}

function deterministicShuffle<T>(items: T[], seed: number): T[] {
  let state = seed >>> 0;
  const next = () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 0x100000000;
  };

  const copy = [...items];
  for (let index = copy.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(next() * (index + 1));
    [copy[index], copy[swapIndex]] = [copy[swapIndex]!, copy[index]!];
  }

  return copy;
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

function clampPercent(value: number) {
  return Math.max(0, Math.min(100, value));
}

function formatRatio(value: number) {
  return Number.isFinite(value) ? value.toFixed(3) : "n/a";
}

function formatPercent(value: number) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "n/a";
}

function formatTargetThreshold(value: number, type: "min" | "max") {
  return `${type === "min" ? ">" : "<"} ${value.toFixed(2)}`;
}

function compareThreshold(value: number, target: number, fail: number, type: "min" | "max"): "PASS" | "WARN" | "FAIL" {
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

function pearsonCorrelation(xs: number[], ys: number[]) {
  if (xs.length !== ys.length || xs.length < 2) {
    return Number.NaN;
  }

  const meanX = xs.reduce((sum, value) => sum + value, 0) / xs.length;
  const meanY = ys.reduce((sum, value) => sum + value, 0) / ys.length;

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

function average(values: number[]) {
  if (values.length === 0) {
    return Number.NaN;
  }

  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function parseSupportingNodeIds(raw: string) {
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter((value): value is string => typeof value === "string") : [];
  } catch {
    return [];
  }
}

function parseBigFiveResponse(text: string): BigFiveScores | null {
  // Try parsing the full text as JSON first
  let parsed = parseJsonObject<Record<string, unknown>>(text);
  
  // If that fails, try to extract JSON from within the text (think() returns natural language)
  if (!parsed) {
    const jsonMatch = text.match(/\{[^{}]*"openness"[^{}]*\}/);
    if (jsonMatch) {
      parsed = parseJsonObject<Record<string, unknown>>(jsonMatch[0]);
    }
  }
  
  // Last resort: extract numbers after trait names
  if (!parsed) {
    const extract = (trait: string): number => {
      const re = new RegExp(`${trait}[^0-9]*?(\\d+)`, 'i');
      const m = text.match(re);
      return m ? Number.parseInt(m[1], 10) : Number.NaN;
    };
    parsed = {
      openness: extract('openness'),
      conscientiousness: extract('conscientiousness'),
      extraversion: extract('extraversion'),
      agreeableness: extract('agreeableness'),
      neuroticism: extract('neuroticism'),
    };
  }

  if (!parsed) {
    return null;
  }

  const normalize = (key: keyof BigFiveScores) => {
    const rawValue = parsed[key];
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

async function judgeTraitExtraction(
  config: KnownConfig,
  personaId: number,
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

Return JSON:
{
  "matched_ground_truth_indices": [0],
  "matched_extracted_indices": [1]
}`,
      },
      {
        role: "user",
        content: JSON.stringify({
          personaId,
          ground_truth_traits: groundTruthTraits,
          extracted_nodes: extractedNodes,
        }),
      },
    ],
  });

  return parseJsonObject<TraitJudgeResult>(response.choices[0]?.message?.content ?? "") ?? {};
}

async function judgeQABatch(
  config: KnownConfig,
  batch: Array<{
    id: string;
    personaId: number;
    question: string;
    preference: string;
    correctAnswer: string;
    response: string;
  }>,
): Promise<Map<string, boolean>> {
  const openai = getOpenAIClient(config);
  const response = await openai.chat.completions.create({
    model: config.synthesisModel,
    temperature: 0,
    response_format: { type: "json_object" },
    messages: [
      {
        role: "system",
        content: `You judge whether a memory-aware response uses the correct user preference.

Answer "yes" only if the response clearly incorporates the target preference in a way that is consistent with the question and the reference answer.
Paraphrases count. Generic advice that ignores the preference is "no".

Return JSON:
{
  "results": [
    { "id": "item-1", "verdict": "yes" }
  ]
}`,
      },
      {
        role: "user",
        content: JSON.stringify({ items: batch }),
      },
    ],
  });

  const parsed = parseJsonObject<BatchJudgeResult>(response.choices[0]?.message?.content ?? "") ?? {};
  const verdicts = new Map<string, boolean>();

  for (const result of parsed.results ?? []) {
    verdicts.set(result.id, result.verdict === "yes");
  }

  return verdicts;
}

async function judgeDreamInsights(
  config: KnownConfig,
  insights: Array<{
    id: string;
    insight: string;
    support: string[];
  }>,
): Promise<Map<string, number>> {
  if (insights.length === 0) {
    return new Map<string, number>();
  }

  const openai = getOpenAIClient(config);
  const response = await openai.chat.completions.create({
    model: config.synthesisModel,
    temperature: 0,
    response_format: { type: "json_object" },
    messages: [
      {
        role: "system",
        content: `You judge whether a discovered insight is a genuine structural connection.

Score each item from 1 to 5:
1 = hallucinated or unsupported
2 = weak or mostly topical
3 = supported but modest
4 = clearly genuine structural link
5 = strong, non-obvious, well-supported structural insight

Return JSON:
{
  "ratings": [
    { "id": "item-1", "score": 4 }
  ]
}`,
      },
      {
        role: "user",
        content: JSON.stringify({ items: insights }),
      },
    ],
  });

  const parsed = parseJsonObject<DreamJudgeResult>(response.choices[0]?.message?.content ?? "") ?? {};
  const ratings = new Map<string, number>();

  for (const rating of parsed.ratings ?? []) {
    if (typeof rating.score === "number") {
      ratings.set(rating.id, rating.score);
    }
  }

  return ratings;
}

async function preparePersonaMemPersonas(baseConfig: KnownConfig, personaCount: number): Promise<PreparedPersona[]> {
  const extracted = readJsonFile<PersonaMemExtractedData>(PERSONAMEM_JSON_PATH);
  const selected = extracted.personas.slice(0, personaCount);
  const prepared: PreparedPersona[] = [];

  console.log(`Selected PersonaMem personas: ${selected.map((persona) => persona.personaId).join(", ")}`);

  for (const persona of selected) {
    const dbPath = createTempDbPath(`personamem-${persona.personaId}`);
    cleanupDbFiles(dbPath);
    const config = getBenchmarkConfig({ ...baseConfig, dbPath });
    const db = new KnownDB(dbPath);

    try {
      const ingestResult = await ingest(db, persona.conversationText, config, `personamem-${persona.personaId}`);
      prepared.push({
        persona,
        db,
        dbPath,
        config,
        ingestResult,
        encodedNodes: db.getAllNodes(),
      });
    } catch (error) {
      db.close();
      cleanupDbFiles(dbPath);
      throw error;
    }
  }

  return prepared;
}

function cleanupPreparedPersonas(personas: PreparedPersona[]) {
  for (const persona of personas) {
    persona.db.close();
    cleanupDbFiles(persona.dbPath);
  }
}

async function runTest1A(prepared: PreparedPersona[], baseConfig: KnownConfig): Promise<Test1AResult> {
  printSection("Test 1A: Trait Extraction");

  let matchedGroundTruth = 0;
  let totalGroundTruth = 0;
  let matchedExtracted = 0;
  let totalExtracted = 0;

  for (const item of prepared) {
    const groundTruthTraits = item.persona.groundTruthTraits;
    const extractedNodes = item.encodedNodes.map((node) => node.text);
    const judged = await judgeTraitExtraction(baseConfig, item.persona.personaId, groundTruthTraits, extractedNodes);

    const matchedGroundTruthIndices = new Set(judged.matched_ground_truth_indices ?? []);
    const matchedExtractedIndices = new Set(judged.matched_extracted_indices ?? []);

    const recall = groundTruthTraits.length === 0 ? 0 : matchedGroundTruthIndices.size / groundTruthTraits.length;
    const precision = extractedNodes.length === 0 ? 0 : matchedExtractedIndices.size / extractedNodes.length;
    const f1 = recall + precision === 0 ? 0 : (2 * recall * precision) / (recall + precision);

    matchedGroundTruth += matchedGroundTruthIndices.size;
    totalGroundTruth += groundTruthTraits.length;
    matchedExtracted += matchedExtractedIndices.size;
    totalExtracted += extractedNodes.length;

    console.log(
      `Persona ${item.persona.personaId}: recall ${formatPercent(recall)}, precision ${formatPercent(precision)}, F1 ${formatRatio(f1)} (${groundTruthTraits.length} gt / ${extractedNodes.length} nodes)`,
    );
  }

  const recall = totalGroundTruth === 0 ? 0 : matchedGroundTruth / totalGroundTruth;
  const precision = totalExtracted === 0 ? 0 : matchedExtracted / totalExtracted;
  const f1 = recall + precision === 0 ? 0 : (2 * recall * precision) / (recall + precision);

  console.log(
    `Overall: recall ${formatPercent(recall)}, precision ${formatPercent(precision)}, F1 ${formatRatio(f1)} (${matchedGroundTruth}/${totalGroundTruth} gt matches, ${matchedExtracted}/${totalExtracted} node matches)`,
  );

  return {
    recall,
    precision,
    f1,
    matchedGroundTruth,
    totalGroundTruth,
    matchedExtracted,
    totalExtracted,
  };
}

async function runTest2A(prepared: PreparedPersona[], baseConfig: KnownConfig): Promise<Test2AResult> {
  printSection("Test 2A: PersonaMem QA Accuracy");

  let correct = 0;
  let total = 0;

  for (const item of prepared) {
    const personaEvaluations: Array<{
      id: string;
      personaId: number;
      question: string;
      preference: string;
      correctAnswer: string;
      response: string;
    }> = [];
    let personaCorrect = 0;

    for (const [index, qaPair] of item.persona.qaPairs.entries()) {
      const result = await think(item.db, qaPair.userQuery, item.config);
      personaEvaluations.push({
        id: `${item.persona.personaId}-${index}`,
        personaId: item.persona.personaId,
        question: qaPair.userQuery,
        preference: qaPair.preference,
        correctAnswer: qaPair.correctAnswer,
        response: result.response,
      });
    }

    for (const batch of chunk(personaEvaluations, QA_JUDGE_BATCH_SIZE)) {
      const verdicts = await judgeQABatch(baseConfig, batch);
      for (const entry of batch) {
        if (verdicts.get(entry.id)) {
          personaCorrect += 1;
        }
      }
    }

    correct += personaCorrect;
    total += personaEvaluations.length;
    const personaAccuracy = item.persona.qaPairs.length === 0 ? 0 : personaCorrect / item.persona.qaPairs.length;
    console.log(
      `Persona ${item.persona.personaId}: accuracy ${formatPercent(personaAccuracy)} (${personaCorrect}/${item.persona.qaPairs.length})`,
    );
  }

  const accuracy = total === 0 ? 0 : correct / total;
  console.log(`Overall: accuracy ${formatPercent(accuracy)} (${correct}/${total})`);

  return { accuracy, correct, total };
}

async function runTest3A(prepared: PreparedPersona[], _baseConfig: KnownConfig): Promise<Test3AResult> {
  printSection("Test 3A: Dream Validity");

  const richest = [...prepared]
    .sort((left, right) => right.db.getStats().nodeCount - left.db.getStats().nodeCount)
    .slice(0, 2);

  console.log(`Dream personas: ${richest.map((item) => item.persona.personaId).join(", ")}`);

  const insightOccurrences = new Map<
    string,
    {
      personaId: number;
      insight: string;
      support: string[];
      count: number;
    }
  >();

  for (const item of richest) {
    let found = 0;
    for (let run = 0; run < 10; run += 1) {
      const result = await discover(item.db, item.config);
      if (!result.found || !result.insight) {
        continue;
      }

      found += 1;
      const insightRow = item.db
        .getAllInsights()
        .filter((insight) => insight.text === result.insight)
        .sort((left, right) => right.discovered_at.localeCompare(left.discovered_at))[0];

      const support = insightRow
        ? parseSupportingNodeIds(insightRow.supporting_nodes)
            .map((nodeId) => item.db.getNode(nodeId)?.text ?? "")
            .filter((text) => text.length > 0)
        : [];

      const key = `${item.persona.personaId}::${result.insight}`;
      const existing = insightOccurrences.get(key);
      if (existing) {
        existing.count += 1;
      } else {
        insightOccurrences.set(key, {
          personaId: item.persona.personaId,
          insight: result.insight,
          support,
          count: 1,
        });
      }
    }

    console.log(`Persona ${item.persona.personaId}: ${found} insights found across 10 DREAM runs`);
  }

  const uniqueInsights = [...insightOccurrences.entries()].map(([id, value]) => ({
    id,
    insight: value.insight,
    support: value.support.slice(0, 8),
  }));
  const ratings = await judgeDreamInsights(richest[0]?.config ?? getBenchmarkConfig(), uniqueInsights);

  let genuineCount = 0;
  let hallucinationCount = 0;
  let totalInsights = 0;
  for (const [id, occurrence] of insightOccurrences.entries()) {
    const score = ratings.get(id) ?? 1;
    totalInsights += occurrence.count;
    if (score >= 3) {
      genuineCount += occurrence.count;
    }
    if (score <= 2) {
      hallucinationCount += occurrence.count;
    }
  }

  const genuineRate = totalInsights === 0 ? 0 : genuineCount / totalInsights;
  const hallucinationRate = totalInsights === 0 ? 0 : hallucinationCount / totalInsights;
  console.log(
    `Overall: genuine rate ${formatPercent(genuineRate)} (${genuineCount}/${totalInsights}), hallucination rate ${formatPercent(hallucinationRate)} (${hallucinationCount}/${totalInsights})`,
  );

  return {
    genuineRate,
    genuineCount,
    hallucinationRate,
    hallucinationCount,
    totalInsights,
  };
}

async function withFreshDb<T>(
  label: string,
  baseConfig: KnownConfig,
  run: (db: KnownDB, config: KnownConfig) => Promise<T>,
): Promise<T> {
  const dbPath = createTempDbPath(label);
  cleanupDbFiles(dbPath);
  const config = getBenchmarkConfig({ ...baseConfig, dbPath });
  const db = new KnownDB(dbPath);

  try {
    return await run(db, config);
  } finally {
    db.close();
    cleanupDbFiles(dbPath);
  }
}

async function runTest4A(baseConfig: KnownConfig, pandoraUsers: number): Promise<Test4AResult> {
  printSection("Test 4A: Big Five");

  const extracted = readJsonFile<PandoraExtractedData>(PANDORA_JSON_PATH);
  const eligibleUsers = extracted.users.filter((user) => user.textLength >= 500);
  const sampledUsers = deterministicShuffle(eligibleUsers, 42).slice(0, pandoraUsers);

  console.log(`Sampled PANDORA users: ${sampledUsers.length} of ${eligibleUsers.length} eligible (>= 500 chars)`);

  const predictions: Array<{ groundTruth: BigFiveScores; predicted: BigFiveScores }> = [];

  for (const user of sampledUsers) {
    const predicted = await withFreshDb(`pandora-${user.userId}`, baseConfig, async (db, config) => {
      await ingest(db, `USER: ${user.text}`, config, user.userId);
      const result = await think(
        db,
        BIG_FIVE_QUERY,
        config,
        "Benchmark mode. Return only minified JSON with numeric keys openness, conscientiousness, extraversion, agreeableness, and neuroticism.",
      );
      return parseBigFiveResponse(result.response);
    });

    if (predicted) {
      predictions.push({ groundTruth: user.groundTruth, predicted });
    }
  }

  const traits: Array<keyof BigFiveScores> = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
  ];

  const correlations = {
    openness: Number.NaN,
    conscientiousness: Number.NaN,
    extraversion: Number.NaN,
    agreeableness: Number.NaN,
    neuroticism: Number.NaN,
  };

  let traitsAboveTarget = 0;
  for (const trait of traits) {
    const groundTruth = predictions.map((entry) => entry.groundTruth[trait]);
    const predicted = predictions.map((entry) => entry.predicted[trait]);
    const correlation = pearsonCorrelation(groundTruth, predicted);
    correlations[trait] = correlation;
    if (Number.isFinite(correlation) && correlation >= 0.25) {
      traitsAboveTarget += 1;
    }
    console.log(`${trait}: r=${formatRatio(correlation)}`);
  }

  const meanCorrelation = average(Object.values(correlations).filter((value) => Number.isFinite(value)));
  console.log(`Overall: mean r=${formatRatio(meanCorrelation)} (${traitsAboveTarget}/5 traits >= 0.25, ${predictions.length} valid users)`);

  return {
    meanCorrelation,
    traitsAboveTarget,
    correlations,
    validUsers: predictions.length,
  };
}

function buildSummaryRows(results: {
  test1A?: Test1AResult;
  test2A?: Test2AResult;
  test3A?: Test3AResult;
  test4A?: Test4AResult;
}): SummaryRow[] {
  const rows: SummaryRow[] = [];

  if (results.test1A) {
    rows.push({
      test: "1A Trait Extraction",
      metric: "F1",
      value: formatRatio(results.test1A.f1),
      target: `${formatTargetThreshold(0.55, "min")} / fail < 0.40`,
      status: compareThreshold(results.test1A.f1, 0.55, 0.4, "min"),
    });
  }

  if (results.test2A) {
    rows.push({
      test: "2A PersonaMem QA",
      metric: "Accuracy",
      value: formatPercent(results.test2A.accuracy),
      target: `${formatTargetThreshold(0.5, "min")} / fail < 0.35`,
      status: compareThreshold(results.test2A.accuracy, 0.5, 0.35, "min"),
    });
  }

  if (results.test3A) {
    rows.push({
      test: "3A Dream Genuine",
      metric: "Rate",
      value: formatPercent(results.test3A.genuineRate),
      target: `${formatTargetThreshold(0.6, "min")} / fail < 0.40`,
      status: compareThreshold(results.test3A.genuineRate, 0.6, 0.4, "min"),
    });
    rows.push({
      test: "3A Dream Hallucination",
      metric: "Rate",
      value: formatPercent(results.test3A.hallucinationRate),
      target: `${formatTargetThreshold(0.2, "max")} / fail > 0.35`,
      status: compareThreshold(results.test3A.hallucinationRate, 0.2, 0.35, "max"),
    });
  }

  if (results.test4A) {
    const status =
      results.test4A.traitsAboveTarget >= 3
        ? "PASS"
        : compareThreshold(results.test4A.meanCorrelation, 0.25, 0.15, "min");

    rows.push({
      test: "4A Big Five",
      metric: "Mean r",
      value: `${formatRatio(results.test4A.meanCorrelation)} (${results.test4A.traitsAboveTarget}/5 >= 0.25)`,
      target: "> 0.25 for 3 of 5 / fail mean < 0.15",
      status,
    });
  }

  return rows;
}

export async function runBenchmarkCli(args: string[] = []) {
  const options = parseArgs(args);
  loadApiKeyFromDotenv();

  const needsPersonas = options.test === "all" || options.test === "1a" || options.test === "2a" || options.test === "3a";
  const needsPandora = options.test === "all" || options.test === "4a";

  await ensureExtractedData(needsPersonas, needsPandora);

  const baseConfig = getBenchmarkConfig();
  const results: {
    test1A?: Test1AResult;
    test2A?: Test2AResult;
    test3A?: Test3AResult;
    test4A?: Test4AResult;
  } = {};

  let preparedPersonas: PreparedPersona[] = [];

  try {
    if (needsPersonas) {
      preparedPersonas = await preparePersonaMemPersonas(baseConfig, options.personas);
    }

    if (options.test === "all" || options.test === "1a") {
      results.test1A = await runTest1A(preparedPersonas, baseConfig);
    }

    if (options.test === "all" || options.test === "2a") {
      results.test2A = await runTest2A(preparedPersonas, baseConfig);
    }

    if (options.test === "all" || options.test === "3a") {
      results.test3A = await runTest3A(preparedPersonas, baseConfig);
    }

    if (options.test === "all" || options.test === "4a") {
      results.test4A = await runTest4A(baseConfig, options.pandoraUsers);
    }
  } finally {
    cleanupPreparedPersonas(preparedPersonas);
  }

  const summaryRows = buildSummaryRows(results);
  printSection("Summary");
  printTable(
    ["Test", "Metric", "Value", "Target", "Status"],
    summaryRows.map((row) => [row.test, row.metric, row.value, row.target, row.status]),
  );
}
