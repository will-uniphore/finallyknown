#!/usr/bin/env python3

import ast
import json
from pathlib import Path

from datasets import load_from_disk


DATA_ROOT = Path("/tmp/known-test-data")
OUTPUT_ROOT = DATA_ROOT / "extracted"
PERSONAMEM_OUTPUT = OUTPUT_ROOT / "personamem-v2-benchmark.json"
PANDORA_OUTPUT = OUTPUT_ROOT / "pandora-test.json"

TRAIT_ROOTS = {
    "health",
    "hobbies_interests",
    "personality",
    "speaking_style_with_chatbot",
    "values_beliefs",
}


def parse_structured(raw):
    if raw is None or isinstance(raw, (dict, list, int, float, bool)):
        return raw

    if not isinstance(raw, str):
        return raw

    text = raw.strip()
    if not text:
        return text

    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except Exception:
            continue

    return raw


def clean_text(value):
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").strip()


def dedupe_strings(values):
    seen = set()
    ordered = []
    for value in values:
        text = clean_text(value)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(text)
    return ordered


def format_messages(messages):
    if not isinstance(messages, list):
        return ""

    lines = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = clean_text(message.get("role", "message")).upper() or "MESSAGE"
        content = clean_text(message.get("content", ""))
        if not content:
            continue
        lines.append(f"{role}: {content}")

    return "\n".join(lines)


def normalize_query(raw_query):
    parsed = parse_structured(raw_query)
    if isinstance(parsed, dict):
        return clean_text(parsed.get("content", raw_query))
    return clean_text(parsed)


def collect_trait_strings(value, path=()):
    collected = []

    if isinstance(value, dict):
        for key, nested in value.items():
            collected.extend(collect_trait_strings(nested, path + (str(key),)))
        return collected

    if isinstance(value, list):
        for nested in value:
            collected.extend(collect_trait_strings(nested, path))
        return collected

    if not isinstance(value, str):
        return collected

    if any(segment in TRAIT_ROOTS for segment in path):
        collected.append(clean_text(value))

    return collected


def extract_personamem():
    dataset_path = DATA_ROOT / "personamem-v2"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing PersonaMem v2 dataset at {dataset_path}")

    dataset = load_from_disk(str(dataset_path))["benchmark_text"]
    grouped = {}

    for row in dataset:
        persona_id = int(row["persona_id"])
        persona = grouped.setdefault(
            persona_id,
            {
                "personaId": persona_id,
                "expandedPersona": clean_text(row.get("expanded_persona", "")),
                "snippets": [],
                "conversationText": "",
                "uniqueSnippetChars": 0,
                "groundTruthTraits": [],
                "qaPairs": [],
            },
        )

        snippet_text = format_messages(parse_structured(row.get("related_conversation_snippet", "")))
        if snippet_text:
            persona["snippets"].append(snippet_text)

        persona["qaPairs"].append(
            {
                "userQuery": normalize_query(row.get("user_query", "")),
                "correctAnswer": clean_text(row.get("correct_answer", "")),
                "preference": clean_text(row.get("preference", "")),
                "updated": bool(row.get("updated", False)),
                "prevPref": clean_text(row.get("prev_pref", "")) or None,
            }
        )

    personas = []
    for persona in grouped.values():
        expanded = parse_structured(persona["expandedPersona"])
        ground_truth_traits = []
        if isinstance(expanded, dict):
            ground_truth_traits = dedupe_strings(collect_trait_strings(expanded))

        deduped_snippets = dedupe_strings(persona["snippets"])
        conversation_text = "\n\n".join(deduped_snippets)

        personas.append(
            {
                "personaId": persona["personaId"],
                "expandedPersona": persona["expandedPersona"],
                "groundTruthTraits": ground_truth_traits,
                "snippets": deduped_snippets,
                "conversationText": conversation_text,
                "uniqueSnippetChars": len(conversation_text),
                "qaPairs": persona["qaPairs"],
            }
        )

    personas.sort(key=lambda item: (-item["uniqueSnippetChars"], -len(item["qaPairs"]), item["personaId"]))
    return {"personas": personas}


def extract_pandora():
    dataset_path = DATA_ROOT / "pandora"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing PANDORA dataset at {dataset_path}")

    dataset = load_from_disk(str(dataset_path))["test"]
    users = []

    for index, row in enumerate(dataset):
        text = clean_text(row.get("text", ""))
        users.append(
            {
                "userId": f"pandora-{index}",
                "text": text,
                "textLength": len(text),
                "groundTruth": {
                    "openness": float(row.get("openness", 0.0)),
                    "conscientiousness": float(row.get("conscientiousness", 0.0)),
                    "extraversion": float(row.get("extraversion", 0.0)),
                    "agreeableness": float(row.get("agreeableness", 0.0)),
                    "neuroticism": float(row.get("neuroticism", 0.0)),
                },
            }
        )

    return {"users": users}


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    with PERSONAMEM_OUTPUT.open("w", encoding="utf-8") as handle:
        json.dump(extract_personamem(), handle, ensure_ascii=True)

    with PANDORA_OUTPUT.open("w", encoding="utf-8") as handle:
        json.dump(extract_pandora(), handle, ensure_ascii=True)

    print(PERSONAMEM_OUTPUT)
    print(PANDORA_OUTPUT)


if __name__ == "__main__":
    main()
