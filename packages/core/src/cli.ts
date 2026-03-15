#!/usr/bin/env node
import { readFileSync } from "fs";
import { stdin as input } from "process";
import { getConfig } from "./config.js";
import { KnownDB } from "./db.js";
import { discover } from "./discover.js";
import { ingest } from "./ingest.js";
import { maintain } from "./maintain.js";
import { think } from "./think.js";

function usage() {
  console.log(`Usage:
  known ingest <file> [--session <session-id>]
  known ingest --text "<session text>" [--session <session-id>]
  cat session.txt | known ingest [--session <session-id>]

  known query "<question>" [--context "<agent context>"]
  known discover
  known maintain
  known stats`);
}

function parseFlag(args: string[], name: string) {
  const index = args.indexOf(name);
  if (index === -1) {
    return { value: undefined, rest: args };
  }

  const value = args[index + 1];
  const rest = args.slice(0, index).concat(args.slice(index + 2));
  return { value, rest };
}

async function readStdin(): Promise<string> {
  if (input.isTTY) {
    return "";
  }

  const chunks: string[] = [];
  for await (const chunk of input) {
    chunks.push(chunk.toString());
  }
  return chunks.join("").trim();
}

async function main() {
  const [, , command, ...rawArgs] = process.argv;
  if (!command) {
    usage();
    process.exit(0);
  }

  const config = getConfig();
  const db = new KnownDB(config.dbPath);

  try {
    switch (command) {
      case "ingest": {
        const sessionFlag = parseFlag(rawArgs, "--session");
        const textFlag = parseFlag(sessionFlag.rest, "--text");
        const filePath = textFlag.rest[0];
        const stdinText = await readStdin();
        const sessionText = textFlag.value ?? (filePath ? readFileSync(filePath, "utf8") : stdinText);

        if (!sessionText) {
          console.error("Usage error: provide a file, --text, or piped stdin for ingest.");
          process.exit(1);
        }

        const result = await ingest(db, sessionText, config, sessionFlag.value);
        console.log(`Ingested session: ${result.nodesCreated} nodes created, ${result.edgesCreated} edges created.`);
        break;
      }

      case "query": {
        const contextFlag = parseFlag(rawArgs, "--context");
        const question = contextFlag.rest.join(" ").trim();
        if (!question) {
          console.error('Usage error: known query "<question>" [--context "<agent context>"]');
          process.exit(1);
        }

        const result = await think(db, question, config, contextFlag.value);
        console.log(result.response);
        console.log(`\nInsights used: ${result.insightsUsed}`);
        console.log(`New insights cached: ${result.newInsights}`);
        break;
      }

      case "discover": {
        const result = await discover(db, config);
        if (!result.found) {
          console.log("No non-obvious cross-domain connection found.");
          break;
        }

        console.log(result.insight);
        console.log(result.strengthened ? "\nInsight strengthened." : "\nInsight cached.");
        break;
      }

      case "maintain": {
        const result = maintain(db, config);
        console.log(`Nodes decayed: ${result.nodesDecayed}`);
        console.log(`Nodes pruned: ${result.nodesPruned}`);
        console.log(`Insights pruned: ${result.insightsPruned}`);
        console.log(`Nodes merged: ${result.nodesMerged}`);
        break;
      }

      case "stats": {
        const stats = db.getStats();
        console.log(`Nodes: ${stats.nodeCount}`);
        console.log(`Edges: ${stats.edgeCount}`);
        console.log(`Insights: ${stats.insightCount}`);
        if (stats.topInsights.length > 0) {
          console.log("\nTop insights:");
          for (const insight of stats.topInsights) {
            console.log(
              `- [${insight.confidence.toFixed(2)}] ${insight.text} (used ${insight.times_used}x, rediscovered ${insight.times_rediscovered}x)`
            );
          }
        }
        break;
      }

      default:
        console.error(`Unknown command: ${command}`);
        usage();
        process.exit(1);
    }
  } catch (error) {
    console.error(error instanceof Error ? error.message : error);
    process.exit(1);
  } finally {
    db.close();
  }
}

main();
