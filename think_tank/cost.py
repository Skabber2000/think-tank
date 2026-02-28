"""Token counting and cost estimation for debate runs."""

from __future__ import annotations

from typing import Dict, List, Tuple

from think_tank.schemas import DebateSpec, Panel, DebateState

# Anthropic pricing per million tokens (as of Feb 2026)
MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    # (input_per_M, output_per_M)
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-6": (15.0, 75.0),
    "claude-haiku-4-5-20251001": (0.80, 4.0),
}

# Rough token estimates per component
AVG_SYSTEM_PROMPT_TOKENS = 300
AVG_PROBLEM_CONTEXT_TOKENS = 2000
AVG_PRIOR_MOVES_TOKENS = 1500  # ~6 moves × 250 tokens each
AVG_ROUND_PROMPT_TOKENS = 200
AVG_OUTPUT_TOKENS = 1200


def estimate_cost(
    spec: DebateSpec,
    model: str = "claude-sonnet-4-6",
    synth_model: str = "claude-opus-4-6",
) -> Dict:
    """Estimate the cost of running a debate without making API calls.

    Returns dict with per-round and total cost breakdown.
    """
    rounds_detail = []
    total_input = 0
    total_output = 0
    total_synth_input = 0
    total_synth_output = 0

    for rnd in spec.rounds:
        is_synth = rnd.agents == ["synthesizer"]
        n_agents = len(rnd.agents)

        if is_synth:
            # Synthesizer sees all prior moves
            input_est = (
                AVG_SYSTEM_PROMPT_TOKENS
                + AVG_PROBLEM_CONTEXT_TOKENS
                + total_output  # all prior output becomes synth input
                + AVG_ROUND_PROMPT_TOKENS
            )
            output_est = AVG_OUTPUT_TOKENS * 3  # synthesis is longer
            total_synth_input += input_est
            total_synth_output += output_est
        else:
            input_per_agent = (
                AVG_SYSTEM_PROMPT_TOKENS
                + AVG_PROBLEM_CONTEXT_TOKENS
                + AVG_PRIOR_MOVES_TOKENS
                + AVG_ROUND_PROMPT_TOKENS
            )
            output_per_agent = AVG_OUTPUT_TOKENS
            total_input += input_per_agent * n_agents
            total_output += output_per_agent * n_agents

        rounds_detail.append({
            "round": rnd.number,
            "focus": rnd.focus,
            "agents": n_agents,
            "is_synthesis": is_synth,
        })

    # Calculate costs
    model_price = MODEL_PRICING.get(model, (3.0, 15.0))
    synth_price = MODEL_PRICING.get(synth_model, (15.0, 75.0))

    agent_cost = (
        (total_input / 1_000_000) * model_price[0]
        + (total_output / 1_000_000) * model_price[1]
    )
    synth_cost = (
        (total_synth_input / 1_000_000) * synth_price[0]
        + (total_synth_output / 1_000_000) * synth_price[1]
    )

    total_api_calls = sum(len(r.agents) for r in spec.rounds)

    return {
        "model": model,
        "synth_model": synth_model,
        "total_rounds": len(spec.rounds),
        "total_api_calls": total_api_calls,
        "estimated_input_tokens": total_input + total_synth_input,
        "estimated_output_tokens": total_output + total_synth_output,
        "estimated_agent_cost_usd": round(agent_cost, 2),
        "estimated_synth_cost_usd": round(synth_cost, 2),
        "estimated_total_cost_usd": round(agent_cost + synth_cost, 2),
        "rounds": rounds_detail,
    }


def compute_actual_cost(state: DebateState) -> Dict:
    """Compute actual cost from a completed debate state."""
    model_price = MODEL_PRICING.get(state.model, (3.0, 15.0))
    synth_price = MODEL_PRICING.get(state.synth_model, (15.0, 75.0))

    agent_input = 0
    agent_output = 0
    synth_input = 0
    synth_output = 0

    for move in state.moves:
        if move.move_type == "synthesize":
            synth_input += move.input_tokens
            synth_output += move.output_tokens
        else:
            agent_input += move.input_tokens
            agent_output += move.output_tokens

    agent_cost = (
        (agent_input / 1_000_000) * model_price[0]
        + (agent_output / 1_000_000) * model_price[1]
    )
    synth_cost = (
        (synth_input / 1_000_000) * synth_price[0]
        + (synth_output / 1_000_000) * synth_price[1]
    )

    return {
        "model": state.model,
        "synth_model": state.synth_model,
        "agent_input_tokens": agent_input,
        "agent_output_tokens": agent_output,
        "synth_input_tokens": synth_input,
        "synth_output_tokens": synth_output,
        "agent_cost_usd": round(agent_cost, 4),
        "synth_cost_usd": round(synth_cost, 4),
        "total_cost_usd": round(agent_cost + synth_cost, 4),
    }
