# Finally known.

Known is the understanding layer for your AI life: not a chat log, not a dashboard, but a system that notices the patterns you cannot see across weeks, tools, and conversations, then turns that accumulated understanding into moments where your agent feels like it already knows you.

## Four moments

- Initiation: when Known is genuinely confident in a pattern, your agent starts the conversation with "I've noticed something."
- Mirror: in the middle of a real task, your agent answers from lived context instead of making you restate your history.
- Surprise: you open a new tool, and it already understands how you work because Known carried the relationship over.
- Protection: before you repeat a pattern that has hurt you before, your agent tells you the truth without blocking you.

## Quick start

1. Put your OpenAI key in `~/.known/.env` as `OPENAI_API_KEY=...`.
2. In OpenClaw, install the local plugin at `packages/known-plugin/openclaw.plugin.json`.
3. Let Known build context in `~/.known/brain.db` as you work.

## CLI

```bash
npx known stats
npx known query "What patterns do you see in how I work?"
npx known serve --port 3456
```

## API

```bash
curl http://localhost:3456/stats

curl -X POST http://localhost:3456/query \
  -H "content-type: application/json" \
  -d '{"question":"What patterns do you see in how I work?"}'
```

[finallyknown.ai](https://finallyknown.ai)
