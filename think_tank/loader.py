"""YAML loader and validator for panels and debate specs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import yaml

from think_tank.schemas import Expert, Panel, RoundSpec, DebateSpec


def _resolve_path(path: str) -> Path:
    """Resolve a path relative to the package root if not absolute."""
    p = Path(path)
    if p.is_absolute():
        return p
    # Try relative to CWD first, then package root
    if p.exists():
        return p
    pkg_root = Path(__file__).parent.parent
    candidate = pkg_root / p
    if candidate.exists():
        return candidate
    return p  # Return original, let caller handle missing


def load_panel(path: str) -> Panel:
    """Load an expert panel from a YAML file.

    Supports two formats:
    - WW3 format: id, name, role, bias, lens
    - Cluster format: id, name, title, background
    """
    p = _resolve_path(path)
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    panel_name = data.get("name", p.stem)
    description = data.get("description", "")
    experts_raw = data.get("experts", [])

    experts = []
    for e in experts_raw:
        expert = Expert(
            id=str(e.get("id", e.get("name", "").lower().replace(" ", "_"))),
            name=e.get("name", "Unknown"),
            title=e.get("title", e.get("role", "")),
            background=e.get("background", ""),
            bias=e.get("bias", ""),
            lens=e.get("lens", ""),
            domain=e.get("domain", ""),
        )
        experts.append(expert)

    return Panel(name=panel_name, description=description, experts=experts)


def load_spec(path: str) -> DebateSpec:
    """Load a debate specification from a YAML file."""
    p = _resolve_path(path)
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    title = data.get("title", p.stem)
    context = data.get("context", "")
    synth_prompt = data.get("synthesizer_prompt", "")

    rounds = []
    for r in data.get("rounds", []):
        rounds.append(RoundSpec(
            number=r.get("number", len(rounds) + 1),
            focus=r.get("focus", ""),
            question=r.get("question", ""),
            agents=r.get("agents", []),
        ))

    return DebateSpec(
        title=title,
        context=context,
        rounds=rounds,
        synthesizer_prompt=synth_prompt,
    )


def validate_spec_against_panel(spec: DebateSpec, panel: Panel) -> List[str]:
    """Check that all agents referenced in the spec exist in the panel.

    Returns a list of warning messages (empty = valid).
    """
    panel_ids = set(panel.list_ids())
    warnings = []

    for rnd in spec.rounds:
        for agent_id in rnd.agents:
            if agent_id == "synthesizer":
                continue
            if agent_id not in panel_ids:
                warnings.append(
                    f"Round {rnd.number}: agent '{agent_id}' not found in panel "
                    f"'{panel.name}' (available: {len(panel_ids)} experts)"
                )

    return warnings


def discover_files(directory: str, suffix: str = ".yaml") -> List[Tuple[str, str]]:
    """Find all YAML files in a directory. Returns [(path, stem), ...]."""
    d = _resolve_path(directory)
    if not d.is_dir():
        return []
    results = []
    for f in sorted(d.iterdir()):
        if f.suffix in (suffix, ".yml") and not f.name.startswith("_"):
            results.append((str(f), f.stem))
    return results
