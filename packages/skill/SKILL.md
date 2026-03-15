# Known

Known is a local brain for AI agents. It stores explicit observations about the user, reasons over them at query time, discovers cross-domain insights in the background, and prunes stale memory over time.

## Server

Run the Known server locally:

```bash
pnpm --filter @known/server start
```

Default address:

```text
http://localhost:7777
```

## When To Use Known

### After a meaningful session
Ingest the session transcript so Known can extract nodes and explicit edges:

```bash
curl -X POST http://localhost:7777/ingest \
  -H "Content-Type: application/json" \
  -d '{"sessionText":"<full session transcript>","sessionId":"session-2026-03-15"}'
```

### Before important reasoning
Ask Known for synthesized understanding instead of raw retrieval:

```bash
curl -X POST http://localhost:7777/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What should I know before this board presentation?","agentContext":"The user just asked for prep help"}'
```

### Daily background discovery
Trigger subconscious cross-domain thinking:

```bash
curl -X POST http://localhost:7777/discover
```

### Daily maintenance
Run neuroplasticity to decay stale knowledge, prune dead memory, and merge near-duplicates:

```bash
curl -X POST http://localhost:7777/maintain
```

### Inspect the brain

```bash
curl http://localhost:7777/stats
```

## Operating Principles

1. Ingest stores only explicit observations and explicit relationships.
2. Query-time thinking is where new connections are discovered.
3. Re-discovered or used insights strengthen over time.
4. Stale observations decay and weak memory gets pruned.
