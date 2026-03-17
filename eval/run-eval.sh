#!/bin/bash
# Known Golden Eval Runner
# Usage: ./eval/run-eval.sh [--test encode|activate|dream|implicit|personality|all] [--limit N]
set -e

cd "$(dirname "$0")/.."
source ~/.known/.env 2>/dev/null || true
export OPENAI_API_KEY

TEST="${1:-all}"
LIMIT="${2:-999}"

echo "╔══════════════════════════════════════════════════════╗"
echo "║         Known Golden Eval — v2 Algorithm             ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "Test: $TEST | Limit: $LIMIT"
echo "Brain: $(npx known stats 2>&1 | head -3)"
echo ""

# Run the TypeScript eval runner
pnpm --filter known exec node dist/cli.js eval \
  --golden eval/golden-eval.json \
  --test "$TEST" \
  --limit "$LIMIT" \
  2>&1

echo ""
echo "Done. Results saved to eval/results/"
