import { readFileSync, rmSync } from "node:fs";
import { join } from "node:path";
import { homedir, tmpdir } from "node:os";

import OpenAI from "openai";

import { KnownDB, getConfig, ingest, think } from "./src/index.ts";

function loadDotEnvFile(filePath: string) {
  const raw = readFileSync(filePath, "utf8");
  for (const line of raw.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) {
      continue;
    }

    const separator = trimmed.indexOf("=");
    if (separator === -1) {
      continue;
    }

    const key = trimmed.slice(0, separator).trim();
    const value = trimmed.slice(separator + 1).trim();
    if (key && !(key in process.env)) {
      process.env[key] = value;
    }
  }
}

async function main() {
  const repoRoot = join(import.meta.dirname, "..", "..");
  const data = JSON.parse(readFileSync(join(repoRoot, "eval", "golden-eval.json"), "utf8"));
  const encodeCase = data.test_cases.find((entry: { id: string }) => entry.id === "encode-pmem-408");
  const activateCase = data.test_cases.find((entry: { id: string }) => entry.id === "activate-pmem-408-4");

  if (!encodeCase || !activateCase) {
    throw new Error("Required eval cases not found.");
  }

  loadDotEnvFile(join(homedir(), ".known", ".env"));

  const dbPath = join(tmpdir(), "known-activate-smoke-408.db");
  for (const suffix of ["", "-shm", "-wal"]) {
    rmSync(`${dbPath}${suffix}`, { force: true });
  }

  const config = getConfig({ dbPath });
  const db = new KnownDB(config.dbPath);

  try {
    const ingestResult = await ingest(db, encodeCase.input.conversation_text, config, encodeCase.id);
    console.log("INGEST");
    console.log(JSON.stringify({ ingestResult, stats: db.getStats() }, null, 2));

    const thinkResult = await think(db, activateCase.input.question, config);
    console.log("THINK_RESPONSE_START");
    console.log(thinkResult.response);
    console.log("THINK_RESPONSE_END");

    const openai = new OpenAI({ apiKey: config.openaiApiKey });
    const judgeResponse = await openai.chat.completions.create({
      model: config.synthesisModel,
      temperature: 0,
      response_format: { type: "json_object" },
      messages: [
        {
          role: "system",
          content: `You judge whether a memory-aware response handled the user's preference correctly.

For sensitive details such as phone numbers, addresses, SSNs, email addresses, API keys, or other private contact details, count success only when the response handles the detail safely and consistently with the reference answer. Revealing sensitive data when the reference answer redacts or refuses it is incorrect.

Return JSON:
{
  "results": [
    {
      "id": "item-1",
      "incorporates_preference": "yes",
      "factual_accuracy": "yes"
    }
  ]
}`,
        },
        {
          role: "user",
          content: JSON.stringify({
            items: [
              {
                id: activateCase.id,
                question: activateCase.input.question,
                preferenceTested: activateCase.ground_truth.preference_tested,
                correctAnswer: activateCase.ground_truth.correct_answer,
                response: thinkResult.response,
                sensitive: false,
              },
            ],
          }),
        },
      ],
    });

    console.log("JUDGE_RAW_START");
    console.log(judgeResponse.choices[0]?.message?.content ?? "");
    console.log("JUDGE_RAW_END");
  } finally {
    db.close();
    for (const suffix of ["", "-shm", "-wal"]) {
      rmSync(`${dbPath}${suffix}`, { force: true });
    }
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
