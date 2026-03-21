import dotenv from "dotenv";
import { serve } from "@hono/node-server";
import { homedir } from "node:os";
import { join } from "node:path";
import { Hono } from "hono";
import { pathToFileURL } from "url";
import { discover, getConfig, ingest, KnownDB, think } from "known";

dotenv.config({ path: join(homedir(), ".known", ".env") });

export function createApp(db: KnownDB, config = getConfig()) {
  const app = new Hono();

  app.post("/ingest", async (c) => {
    try {
      const body = await c.req.json<{ text?: string; sessionText?: string; sessionId?: string }>();
      const text = body.text?.trim() ?? body.sessionText?.trim();
      if (!text) {
        return c.json({ error: "text is required" }, 400);
      }

      return c.json(await ingest(db, text, config, body.sessionId));
    } catch (error) {
      return c.json({ error: error instanceof Error ? error.message : "Unknown error" }, 500);
    }
  });

  app.post("/query", async (c) => {
    try {
      const body = await c.req.json<{ question?: string; agentContext?: string }>();
      if (!body.question?.trim()) {
        return c.json({ error: "question is required" }, 400);
      }

      return c.json(await think(db, body.question, config, body.agentContext));
    } catch (error) {
      return c.json({ error: error instanceof Error ? error.message : "Unknown error" }, 500);
    }
  });

  app.post("/discover", async (c) => {
    try {
      return c.json(await discover(db, config));
    } catch (error) {
      return c.json({ error: error instanceof Error ? error.message : "Unknown error" }, 500);
    }
  });

  app.get("/stats", (c) => {
    try {
      return c.json(db.getStats());
    } catch (error) {
      return c.json({ error: error instanceof Error ? error.message : "Unknown error" }, 500);
    }
  });

  return app;
}

export function startServer(options: { port?: number } = {}) {
  const config = getConfig();
  const db = new KnownDB(config.dbPath);
  const app = createApp(db, config);
  const port = options.port ?? Number.parseInt(process.env.KNOWN_PORT ?? "3456", 10);

  const server = serve({ fetch: app.fetch, port }, () => {
    console.log(`Known API running on http://localhost:${port}`);
  });

  const shutdown = () => {
    server.close();
    db.close();
    process.exit(0);
  };

  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}

const entryUrl = process.argv[1] ? pathToFileURL(process.argv[1]).href : "";
if (import.meta.url === entryUrl) {
  startServer();
}
