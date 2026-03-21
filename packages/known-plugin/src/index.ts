import { homedir } from "node:os";
import path from "node:path";
import { KnownDB, discover, getConfig, getInitiationCandidate, ingest, maintain, think } from "known";
import type { KnownConfig, NodeRow } from "known";

type Logger = {
  debug?: (message: string) => void;
  info: (message: string) => void;
  warn: (message: string) => void;
  error: (message: string) => void;
};

type KnownPluginConfig = {
  openaiApiKey?: unknown;
  dbPath?: unknown;
  extractionModel?: unknown;
  synthesisModel?: unknown;
  discoverIntervalHours?: unknown;
  maintainIntervalHours?: unknown;
  autoIngest?: unknown;
  autoContext?: unknown;
};

type AgentContext = {
  sessionId?: string;
  sessionKey?: string;
};

type SessionContext = {
  sessionId: string;
  sessionKey?: string;
};

type AgentEndEvent = {
  messages: unknown[];
};

type SessionEndEvent = {
  sessionId: string;
  sessionKey?: string;
  messageCount: number;
};

type BeforePromptBuildEvent = {
  prompt: string;
  messages: unknown[];
};

type ToolResult = {
  content: Array<{ type: "text"; text: string }>;
  details?: unknown;
};

type ToolDefinition = {
  name: string;
  label: string;
  description: string;
  parameters: Record<string, unknown>;
  execute: (toolCallId: string, params: Record<string, unknown>) => Promise<ToolResult>;
};

type ServiceContext = {
  logger: Logger;
};

type PluginApi = {
  pluginConfig?: Record<string, unknown>;
  logger: Logger;
  resolvePath: (input: string) => string;
  on: (
    hookName: string,
    handler: (event: unknown, ctx: unknown) => Promise<unknown> | unknown,
    opts?: { priority?: number }
  ) => void;
  registerTool: (tool: ToolDefinition, opts?: { name?: string; names?: string[]; optional?: boolean }) => void;
  registerService: (service: {
    id: string;
    start: (ctx: ServiceContext) => Promise<void> | void;
    stop?: (ctx: ServiceContext) => Promise<void> | void;
  }) => void;
};

type ResolvedSettings = {
  knownConfig: KnownConfig;
  autoIngest: boolean;
  autoContext: boolean;
  discoverIntervalMs: number;
  maintainIntervalMs: number;
};

const DEFAULT_DISCOVER_INTERVAL_HOURS = 4;
const DEFAULT_MAINTAIN_INTERVAL_HOURS = 24;
const INITIATION_CHECK_INTERVAL_MS = 60 * 60 * 1000;
const INITIATION_COOLDOWN_MS = 7 * 24 * 60 * 60 * 1000;
const KNOWN_CONTEXT_PREAMBLE =
  "Known context about the user. Treat this as background guidance, not ground truth, and only use details that help with the current conversation.";

function asString(value: unknown): string | undefined {
  if (typeof value !== "string") {
    return undefined;
  }

  const trimmed = value.trim();
  return trimmed ? trimmed : undefined;
}

function asBoolean(value: unknown, fallback: boolean): boolean {
  return typeof value === "boolean" ? value : fallback;
}

function asNonNegativeNumber(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : fallback;
}

function expandPath(api: PluginApi, rawPath?: string): string | undefined {
  if (!rawPath) {
    return undefined;
  }

  if (rawPath === "~") {
    return homedir();
  }

  if (rawPath.startsWith("~/")) {
    return path.join(homedir(), rawPath.slice(2));
  }

  if (path.isAbsolute(rawPath)) {
    return rawPath;
  }

  return api.resolvePath(rawPath);
}

function resolveSettings(api: PluginApi): ResolvedSettings {
  const pluginConfig = (api.pluginConfig ?? {}) as KnownPluginConfig;
  const knownConfig = getConfig({
    openaiApiKey: asString(pluginConfig.openaiApiKey),
    dbPath: expandPath(api, asString(pluginConfig.dbPath)),
    extractionModel: asString(pluginConfig.extractionModel),
    synthesisModel: asString(pluginConfig.synthesisModel),
  });

  return {
    knownConfig,
    autoIngest: asBoolean(pluginConfig.autoIngest, true),
    autoContext: asBoolean(pluginConfig.autoContext, true),
    discoverIntervalMs: asNonNegativeNumber(pluginConfig.discoverIntervalHours, DEFAULT_DISCOVER_INTERVAL_HOURS) * 60 * 60 * 1000,
    maintainIntervalMs: asNonNegativeNumber(pluginConfig.maintainIntervalHours, DEFAULT_MAINTAIN_INTERVAL_HOURS) * 60 * 60 * 1000,
  };
}

function openDatabase(api: PluginApi) {
  const settings = resolveSettings(api);
  return {
    settings,
    db: new KnownDB(settings.knownConfig.dbPath),
  };
}

async function withDatabase<T>(api: PluginApi, action: (db: KnownDB, settings: ResolvedSettings) => Promise<T> | T): Promise<T> {
  const { db, settings } = openDatabase(api);
  try {
    return await action(db, settings);
  } finally {
    db.close();
  }
}

function formatError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function readRequiredString(params: Record<string, unknown>, key: string): string | null {
  const value = asString(params[key]);
  return value ?? null;
}

function getSessionKeys(sessionKey?: string, sessionId?: string): string[] {
  return [sessionKey, sessionId].filter((value): value is string => Boolean(value));
}

function storeSessionSnapshot(store: Map<string, unknown[]>, sessionKey: string | undefined, sessionId: string | undefined, messages: unknown[]) {
  for (const key of getSessionKeys(sessionKey, sessionId)) {
    store.set(key, messages);
  }
}

function readSessionSnapshot(store: Map<string, unknown[]>, sessionKey: string | undefined, sessionId: string | undefined): unknown[] | undefined {
  for (const key of getSessionKeys(sessionKey, sessionId)) {
    const snapshot = store.get(key);
    if (snapshot) {
      return snapshot;
    }
  }
  return undefined;
}

function clearSessionSnapshot(store: Map<string, unknown[]>, sessionKey: string | undefined, sessionId: string | undefined) {
  for (const key of getSessionKeys(sessionKey, sessionId)) {
    store.delete(key);
  }
}

function extractTextBlocks(content: unknown): string[] {
  if (typeof content === "string") {
    const trimmed = content.trim();
    return trimmed ? [trimmed] : [];
  }

  if (!Array.isArray(content)) {
    return [];
  }

  return content.flatMap((block) => {
    if (!block || typeof block !== "object") {
      return [];
    }

    const record = block as Record<string, unknown>;
    if (record.type !== "text" || typeof record.text !== "string") {
      return [];
    }

    const text = record.text.trim();
    return text ? [text] : [];
  });
}

function serializeSessionMessages(messages: unknown[]): string {
  return messages
    .flatMap((message) => {
      if (!message || typeof message !== "object") {
        return [];
      }

      const record = message as Record<string, unknown>;
      const role = record.role;
      if (role !== "user" && role !== "assistant") {
        return [];
      }

      const text = extractTextBlocks(record.content).join("\n").trim();
      if (!text) {
        return [];
      }

      return [`${String(role).toUpperCase()}:\n${text}`];
    })
    .join("\n\n");
}

function buildContextQuestion(prompt: string): string {
  return [
    "What should I know about this user for this conversation?",
    "",
    `Current prompt: ${prompt}`,
  ].join("\n");
}

function buildContextInjection(text: string): string {
  return `${KNOWN_CONTEXT_PREAMBLE}\n\n${text.trim()}`;
}

function buildInitiationMessage(insightText: string): string {
  const normalizedInsight = insightText.trim();
  const trailingPunctuation = /[.!?]$/.test(normalizedInsight) ? "" : ".";
  return `I've noticed something about you that I think you should know. ${normalizedInsight}${trailingPunctuation} This is something I've observed across multiple conversations. Does this resonate?`;
}

function buildInitiationInjection(message: string): string {
  return [
    "Known has a pending initiation for the user.",
    "In your next response, lead with this exact message verbatim and then wait for the user's reaction before continuing:",
    "",
    message,
  ].join("\n");
}

function normalizeForMatch(value: string): string {
  return value.trim().toLowerCase();
}

function tokenize(value: string): string[] {
  return normalizeForMatch(value)
    .split(/[^a-z0-9]+/i)
    .filter((token) => token.length >= 3);
}

function matchesForgetQuery(node: NodeRow, query: string, tokens: string[]): boolean {
  const haystack = normalizeForMatch(`${node.type} ${node.text}`);
  if (haystack.includes(query)) {
    return true;
  }

  return tokens.length > 0 && tokens.every((token) => haystack.includes(token));
}

function textResult(text: string, details?: unknown): ToolResult {
  return {
    content: [{ type: "text", text }],
    details,
  };
}

async function safeDiscover(api: PluginApi, logger: Logger) {
  try {
    await withDatabase(api, async (db, settings) => {
      if (!settings.knownConfig.openaiApiKey) {
        logger.warn("Known discover skipped: OPENAI_API_KEY is not configured.");
        return;
      }

      const result = await discover(db, settings.knownConfig);
      if (result.found && result.insight) {
        logger.info(`Known discover stored insight: ${result.insight}`);
      }
    });
  } catch (error) {
    logger.error(`Known discover failed: ${formatError(error)}`);
  }
}

async function safeMaintain(api: PluginApi, logger: Logger) {
  try {
    await withDatabase(api, (db, settings) => {
      const result = maintain(db, settings.knownConfig);
      logger.info(
        `Known maintain completed: ${result.nodesDecayed} decayed, ${result.nodesPruned} pruned, ${result.insightsPruned} insights pruned, ${result.nodesMerged} merged.`,
      );
    });
  } catch (error) {
    logger.error(`Known maintain failed: ${formatError(error)}`);
  }
}

async function safeCheckInitiation(
  api: PluginApi,
  logger: Logger,
  hasPendingInitiation: () => boolean,
  queuePendingInitiation: (message: string) => void,
) {
  if (hasPendingInitiation()) {
    return;
  }

  try {
    await withDatabase(api, (db) => {
      const mostRecentInitiatedAt = db.getMostRecentInsightInitiatedAt();
      if (mostRecentInitiatedAt) {
        const initiatedAt = new Date(mostRecentInitiatedAt);
        if (!Number.isNaN(initiatedAt.getTime()) && Date.now() - initiatedAt.getTime() < INITIATION_COOLDOWN_MS) {
          return;
        }
      }

      const candidate = getInitiationCandidate(db);
      if (!candidate) {
        return;
      }

      const message = buildInitiationMessage(candidate.text);
      db.markInsightInitiated(candidate.id);
      queuePendingInitiation(message);
      logger.info(`Known queued initiation from insight ${candidate.id}.`);
    });
  } catch (error) {
    logger.error(`Known initiation check failed: ${formatError(error)}`);
  }
}

export default function register(api: PluginApi) {
  const sessionSnapshots = new Map<string, unknown[]>();
  let discoverTimer: ReturnType<typeof setInterval> | undefined;
  let maintainTimer: ReturnType<typeof setInterval> | undefined;
  let initiationTimer: ReturnType<typeof setInterval> | undefined;
  let pendingInitiationMessage: string | undefined;

  api.on("agent_end", async (event, ctx) => {
    const payload = event as AgentEndEvent;
    const context = ctx as AgentContext;
    if (!Array.isArray(payload.messages)) {
      return;
    }

    storeSessionSnapshot(sessionSnapshots, context.sessionKey, context.sessionId, payload.messages);
  });

  api.on("session_end", async (event, ctx) => {
    const payload = event as SessionEndEvent;
    const context = ctx as SessionContext;
    const settings = resolveSettings(api);

    if (!settings.autoIngest || payload.messageCount === 0) {
      clearSessionSnapshot(sessionSnapshots, context.sessionKey, context.sessionId);
      return;
    }

    const snapshot = readSessionSnapshot(sessionSnapshots, context.sessionKey, context.sessionId);
    clearSessionSnapshot(sessionSnapshots, context.sessionKey, context.sessionId);

    if (!snapshot || snapshot.length === 0) {
      api.logger.debug?.(`Known auto-ingest skipped: no transcript snapshot for session ${payload.sessionId}.`);
      return;
    }

    const sessionText = serializeSessionMessages(snapshot);
    if (!sessionText.trim()) {
      return;
    }

    try {
      await withDatabase(api, async (db, activeSettings) => {
        await ingest(db, sessionText, activeSettings.knownConfig, payload.sessionId);
      });
    } catch (error) {
      api.logger.error(`Known auto-ingest failed for session ${payload.sessionId}: ${formatError(error)}`);
    }
  });

  api.on("before_prompt_build", async (event) => {
    const payload = event as BeforePromptBuildEvent;
    const settings = resolveSettings(api);
    const prompt = payload.prompt?.trim();
    const prependContexts: string[] = [];

    if (pendingInitiationMessage) {
      prependContexts.push(buildInitiationInjection(pendingInitiationMessage));
      pendingInitiationMessage = undefined;
    }

    if (settings.autoContext && prompt) {
      try {
        const contextInjection = await withDatabase(api, async (db, activeSettings) => {
          if (db.getStats().nodeCount === 0) {
            return null;
          }

          const result = await think(
            db,
            buildContextQuestion(prompt),
            activeSettings.knownConfig,
            "Respond with compact agent-facing context that is directly useful for the current conversation."
          );

          const response = result.response.trim();
          if (!response || response === "Unable to reason about this question.") {
            return null;
          }

          return buildContextInjection(response);
        });

        if (contextInjection) {
          prependContexts.push(contextInjection);
        }
      } catch (error) {
        api.logger.error(`Known auto-context failed: ${formatError(error)}`);
      }
    }

    if (prependContexts.length === 0) {
      return;
    }

    return { prependSystemContext: prependContexts.join("\n\n") };
  });

  api.registerTool({
    name: "query_known",
    label: "Query Known",
    description:
      "Query Known for deep understanding about the user. Use when you need specific context about the user's patterns, preferences, relationships, or history.",
    parameters: {
      type: "object",
      additionalProperties: false,
      properties: {
        question: {
          type: "string",
          description: "The question to ask Known about the user.",
        },
        agentContext: {
          type: "string",
          description: "Optional extra context about why this question matters for the current conversation.",
        },
      },
      required: ["question"],
    },
    async execute(_toolCallId, params) {
      const question = readRequiredString(params, "question");
      if (!question) {
        return textResult("Error: question is required.", { error: true });
      }

      const agentContext = asString(params.agentContext);

      try {
        const result = await withDatabase(api, async (db, settings) => {
          return think(db, question, settings.knownConfig, agentContext);
        });

        return textResult(result.response, result);
      } catch (error) {
        return textResult(`Error querying Known: ${formatError(error)}`, { error: true });
      }
    },
  });

  api.registerTool(
    {
      name: "forget_known",
      label: "Forget Known",
      description: "Remove specific knowledge about the user from Known. Use only when the user explicitly asks to forget something.",
      parameters: {
        type: "object",
        additionalProperties: false,
        properties: {
          what: {
            type: "string",
            description: "The memory or topic to remove from Known.",
          },
        },
        required: ["what"],
      },
      async execute(_toolCallId, params) {
        const what = readRequiredString(params, "what");
        if (!what) {
          return textResult("Error: what is required.", { error: true });
        }

        const normalizedQuery = normalizeForMatch(what);
        const tokens = tokenize(what);

        try {
          return await withDatabase(api, async (db) => {
            const matches = db
              .getAllNodes()
              .filter((node) => matchesForgetQuery(node, normalizedQuery, tokens))
              .sort((left, right) => right.confidence - left.confidence);

            if (matches.length === 0) {
              return textResult(`No matching Known memories found for "${what}".`, { deleted: 0 });
            }

            for (const node of matches) {
              db.deleteNode(node.id);
            }

            const summary = matches.slice(0, 10).map((node) => `- ${node.text}`).join("\n");
            return textResult(
              `Removed ${matches.length} matching Known memories.\n\n${summary}`,
              {
                deleted: matches.length,
                nodeIds: matches.map((node) => node.id),
              },
            );
          });
        } catch (error) {
          return textResult(`Error forgetting Known memory: ${formatError(error)}`, { error: true });
        }
      },
    },
    { optional: true },
  );

  api.registerService({
    id: "known-thinker",
    start(ctx) {
      const settings = resolveSettings(api);

      void safeCheckInitiation(
        api,
        ctx.logger,
        () => Boolean(pendingInitiationMessage),
        (message) => {
          pendingInitiationMessage = message;
        },
      );
      initiationTimer = setInterval(() => {
        void safeCheckInitiation(
          api,
          ctx.logger,
          () => Boolean(pendingInitiationMessage),
          (message) => {
            pendingInitiationMessage = message;
          },
        );
      }, INITIATION_CHECK_INTERVAL_MS);
      initiationTimer.unref?.();

      if (settings.discoverIntervalMs > 0) {
        discoverTimer = setInterval(() => {
          void safeDiscover(api, ctx.logger);
        }, settings.discoverIntervalMs);
        discoverTimer.unref?.();
      }

      if (settings.maintainIntervalMs > 0) {
        maintainTimer = setInterval(() => {
          void safeMaintain(api, ctx.logger);
        }, settings.maintainIntervalMs);
        maintainTimer.unref?.();
      }

      ctx.logger.info(
        `Known thinker started (discover every ${settings.discoverIntervalMs / 3600000}h, maintain every ${settings.maintainIntervalMs / 3600000}h, initiation checks every ${INITIATION_CHECK_INTERVAL_MS / 3600000}h).`,
      );
    },
    stop(ctx) {
      if (initiationTimer) {
        clearInterval(initiationTimer);
        initiationTimer = undefined;
      }

      if (discoverTimer) {
        clearInterval(discoverTimer);
        discoverTimer = undefined;
      }

      if (maintainTimer) {
        clearInterval(maintainTimer);
        maintainTimer = undefined;
      }

      ctx.logger.info("Known thinker stopped.");
    },
  });
}
