import { serve } from "@hono/node-server";
import { Hono } from "hono";
import { pathToFileURL } from "url";
import { discover, getConfig, ingest, KnownDB, maintain, think } from "known";

export function createApp(db: KnownDB, config = getConfig()) {
  const app = new Hono();

  app.post("/ingest", async (c) => {
    try {
      const body = await c.req.json<{ sessionText?: string; sessionId?: string }>();
      if (!body.sessionText?.trim()) {
        return c.json({ error: "sessionText is required" }, 400);
      }

      return c.json(await ingest(db, body.sessionText, config, body.sessionId));
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

  app.post("/maintain", (c) => {
    try {
      return c.json(maintain(db, config));
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

export function startServer() {
  const config = getConfig();
  const db = new KnownDB(config.dbPath);
  const app = createApp(db, config);
  const port = Number.parseInt(process.env.KNOWN_PORT ?? "7777", 10);

  const server = serve({ fetch: app.fetch, port }, () => {
    console.log(`Known brain listening at http://localhost:${port}`);
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
