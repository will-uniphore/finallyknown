# Known — Golden Eval Metrics

## How to evaluate: `./eval/run-eval.sh [--test TYPE] [--limit N]`

## Scoring Rubric

### 1. ENCODE Quality (10 test cases)
**Input:** Conversation text → ENCODE → extracted nodes
**Ground truth:** PersonaMem expanded persona traits + hobbies

| Metric | Formula | Target | Fail |
|---|---|---|---|
| **Recall** | GT traits found in nodes / total GT traits | > 70% | < 50% |
| **Precision** | Nodes matching a GT trait / total nodes | > 50% | < 30% |
| **F1** | 2 × (P × R) / (P + R) | > 0.58 | < 0.40 |
| **Hallucination Rate** | Nodes with no evidence support / total nodes | < 10% | > 20% |
| **Compression Ratio** | Nodes / conversation chunks | 5-15 per chunk | <3 or >25 |

**Judge:** LLM-as-judge compares each extracted node against GT traits.
A node "matches" if it captures the same underlying concept even in different words.

### 2. ACTIVATE Accuracy (50 test cases)
**Input:** Question about user → ACTIVATE → response
**Ground truth:** Correct answer + specific preference being tested

| Metric | Formula | Target | Fail |
|---|---|---|---|
| **Preference Incorporation** | Responses incorporating the tested preference / total | > 55% | < 35% |
| **Non-sensitive accuracy** | Same but excluding privacy test questions | > 65% | < 45% |
| **Factual accuracy** | Correct specific facts (names, hobbies, events) | > 50% | < 30% |

**Judge:** LLM-as-judge: "Does this response incorporate the preference '{X}'? yes/no"
**Special handling:** Questions testing sensitive data (phone, SSN, address) are scored separately — Known SHOULD refuse to reveal these.

### 3. DREAM Discovery (1-5 test cases)
**Input:** Full conversation for a multi-domain persona → ENCODE → DREAM × 10
**Ground truth:** Known domains present (work, hobbies, personality)

| Metric | Formula | Target | Fail |
|---|---|---|---|
| **Discovery Rate** | Dreams that found something / total dreams | > 30% | < 10% |
| **Genuine Rate** | Genuine insights / all discovered insights | > 60% | < 40% |
| **Cross-domain Rate** | Insights connecting 2+ domains / all insights | > 50% | < 20% |
| **Hallucination Rate** | Fabricated insights / all insights | < 20% | > 35% |

**Judge:** LLM-as-judge rates each insight 1-5:
- 5: Genuine, non-obvious, cross-domain, supported by evidence
- 4: Genuine, somewhat obvious but still useful
- 3: Plausible but weak evidence
- 2: Surface-level or trivially obvious
- 1: Hallucinated / not supported

Score ≥ 3 = genuine. Score ≥ 4 = non-obvious.

### 4. Implicit Inference (8 test cases)
**Input:** Question where the answer requires inferring something NOT explicitly stated
**Ground truth:** The hidden attribute that should be inferred

| Metric | Formula | Target | Fail |
|---|---|---|---|
| **Detection Rate** | Responses that reference the hidden attribute / total | > 40% | < 20% |
| **Inference Quality** | Responses rated 3+ by judge / total | > 50% | < 30% |

**Judge:** LLM-as-judge: "The hidden attribute is '{X}'. Does the response demonstrate awareness of this attribute, even indirectly? Rate 1-5."

### 5. Personality Extraction (20 test cases)
**Input:** Text from a user → ENCODE → ACTIVATE "Rate Big Five"
**Ground truth:** Validated Big Five scores (0-100)

| Metric | Formula | Target | Fail |
|---|---|---|---|
| **Mean Pearson r** | Average correlation across 5 traits | > 0.20 | < 0.10 |
| **Traits above r=0.25** | Count of traits with r ≥ 0.25 | ≥ 2 of 5 | 0 of 5 |
| **Best trait r** | Highest single-trait correlation | > 0.30 | < 0.15 |

---

## Overall Score

```
KNOWN SCORE = weighted average of:
  30% × ENCODE F1
  30% × ACTIVATE preference incorporation  
  15% × DREAM genuine rate
  10% × Implicit detection rate
  15% × Personality mean r (normalized to 0-1 by dividing by 0.5)
```

| Overall Score | Rating |
|---|---|
| > 0.60 | 🟢 Ship-ready |
| 0.45 - 0.60 | 🟡 Usable but needs iteration |
| 0.30 - 0.45 | 🟠 Significant gaps |
| < 0.30 | 🔴 Not ready |

## Running the Eval

```bash
# Full eval (all 89 test cases)
./eval/run-eval.sh all

# Just ENCODE (10 cases, fast)
./eval/run-eval.sh encode

# Just ACTIVATE (50 cases, ~15 min)
./eval/run-eval.sh activate

# Limit to N cases per type
./eval/run-eval.sh all 3
```

Results are saved to `eval/results/YYYY-MM-DD-HHMMSS.json` for tracking progress across algorithm iterations.
