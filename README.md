# Think Tank

LLM-powered multi-expert structured debate tool. Run debates with any number of expert personas on any topic, producing structured claims with confidence levels, evidence citations, and a self-improving memory system.

## Features

- **Selective Participation**: 4-5 agents per round (not all agents every round) — keeps cost at ~$2-3 per full debate
- **Structured Claims**: JSON output with confidence, evidence, assumptions, and stance
- **Self-Development**: Post-debate lesson extraction, forecast tracking with Brier scores, per-expert performance metrics
- **Flexible Panels**: Define expert personas in YAML — bring any domain expertise
- **Cost Estimation**: `--dry-run` mode shows estimated cost before running
- **Sonnet + Opus**: Regular agents use Sonnet, synthesis uses Opus (configurable)

## Installation

```bash
pip install -e .
```

Requires an [Anthropic API key](https://console.anthropic.com/).

## Quick Start

```bash
# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...

# List available panels and specs
python -m think_tank list

# Estimate cost before running
python -m think_tank run \
  --spec specs/ww3_project_review.yaml \
  --panel panels/geopolitical_42.yaml \
  --dry-run

# Run a full debate
python -m think_tank run \
  --spec specs/ww3_project_review.yaml \
  --panel panels/geopolitical_42.yaml

# Check forecast tracking
python -m think_tank check-forecasts

# Replay a completed debate
python -m think_tank replay runs/WW3_*/debate_state.json
```

## Architecture

```
think_tank/
├── cli.py          # CLI: run, list, check-forecasts, replay, resolve
├── agent.py        # DebateAgent — LLM-powered expert persona
├── runner.py       # DebateRunner — multi-round orchestrator
├── schemas.py      # Dataclasses: Move, Claim, Expert, Forecast, Lesson
├── loader.py       # YAML panel/spec loading + validation
├── report.py       # Markdown report generator
├── memory.py       # Self-development: lessons, forecasts, performance
└── cost.py         # Token counting and cost estimation
```

### How a Debate Works

1. Load a **panel** (expert personas) and **spec** (problem + round structure)
2. For each round, select 4-5 agents based on the spec
3. Each agent receives: problem context, memory (lessons from prior debates), prior moves, and the round question
4. Agents respond with structured JSON: `move_type`, `content`, `claims[]`, `targets[]`
5. Final round uses a stronger model (Opus) for synthesis
6. Post-debate: extract lessons → save to memory → load into future debates

### Self-Development

After each debate, the system:
- Extracts **lessons** (methodology, domain, bias, process insights)
- Tracks **forecasts** with deadlines for Brier score validation
- Records **per-expert performance** (claims, confidence, challenges)

Lessons are injected into future debates, making the system improve over time.

## Creating Custom Panels

```yaml
# panels/my_team.yaml
name: "My Expert Panel"
description: "Domain experts for my problem"

experts:
  - id: analyst
    name: "Dr. Jane Smith"
    title: "Senior Analyst"
    background: "20 years in the field..."
    bias: "Evidence-based, quantitative"
    lens: "What does the data actually say?"

  - id: skeptic
    name: "Dr. John Doe"
    title: "Red Team Lead"
    bias: "Adversarial, contrarian"
    lens: "What are the failure modes?"
```

## Creating Custom Specs

```yaml
# specs/my_problem.yaml
title: "My Problem Assessment"
context: |
  Describe the problem, system, or question here...

rounds:
  - number: 1
    focus: "Assessment"
    question: "What are the key issues?"
    agents: [analyst, skeptic]

  - number: 2
    focus: "Synthesis"
    question: "Synthesize findings into recommendations."
    agents: [synthesizer]
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `run --spec S --panel P` | Run a debate |
| `run --dry-run` | Estimate cost without running |
| `run --no-memory` | Disable self-development |
| `list` | List available panels and specs |
| `check-forecasts` | Show forecast tracking status |
| `replay <state.json>` | Replay a completed debate |
| `resolve <id> <yes/no>` | Resolve a forecast |

## Output

Each debate run creates a timestamped directory in `runs/` containing:
- `debate_state.json` — Full state with all moves and claims
- `report.md` — Markdown report with per-round analysis and claim index
- `cost.json` — Actual token usage and cost breakdown
- `move_XX_RY_agent.json` — Individual move files

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT
