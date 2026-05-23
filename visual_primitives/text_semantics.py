"""Lexicon-based semantic text correction for visual primitive programs."""
from __future__ import annotations

import copy
from typing import Any


TEXT_SEMANTICS_VERSION = "text-semantics-0.1"


DEFAULT_REPLACEMENTS = {
    "Al-Enabled": "AI-Enabled",
    "Al-Enhanced": "AI-Enhanced",
    "Al-Driven": "AI-Driven",
    "coloreled": "colored",
    "Input Metrocs": "Input Metrics",
    "Phase LLM+RL": "Phase I: LLM+RL",
    "PyZX+Hand Layout)": "PyZX+Hand Layout",
    "Soundness hrrc": "Soundness hmc",
    "GNN Surrog:": "GNN Surrogate",
    "b-unitaries": "Sub-unitaries",
    "(Select Truncation": "Select Truncation",
    "jodel": "Model",
    "auge": "Gauge",
    "Benenmarking": "Benchmarking",
    "benenmarking": "benchmarking",
    "Fiter": "Filter",
    "fiter": "filter",
    "Dittusion": "Diffusion",
    "dittusion": "diffusion",
    "analysisfnput": "analysis/Input",
}


def apply_text_replacements(
    program: dict[str, Any],
    replacements: dict[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply exact text replacements from a domain lexicon."""
    updated = copy.deepcopy(program)
    mapping = replacements or DEFAULT_REPLACEMENTS
    operations = []
    for primitive in updated.get("primitives", []):
        if primitive.get("type") != "text":
            continue
        text = str(primitive.get("text", ""))
        new_text = text
        for old, new in mapping.items():
            if old in new and new in new_text:
                continue
            new_text = new_text.replace(old, new)
        if new_text == text:
            continue
        primitive["text"] = new_text
        operations.append({
            "action": "replace_text",
            "primitive_id": primitive.get("id"),
            "before": text,
            "after": new_text,
            "bbox": primitive.get("bbox"),
        })
    report = {
        "version": TEXT_SEMANTICS_VERSION,
        "counts": {
            "operations": len(operations),
            "replacements": len(operations),
        },
        "operations": operations,
    }
    updated.setdefault("metadata", {})["text_semantics"] = report["counts"]
    return updated, report
