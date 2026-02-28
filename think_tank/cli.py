"""CLI for the Think Tank debate tool."""

from __future__ import annotations

import argparse
import json
import os
import sys


def cmd_run(args):
    """Run a debate."""
    from think_tank.loader import load_panel, load_spec, validate_spec_against_panel
    from think_tank.runner import DebateRunner
    from think_tank.cost import estimate_cost

    spec = load_spec(args.spec)
    panel = load_panel(args.panel)

    # Validate
    warnings = validate_spec_against_panel(spec, panel)
    for w in warnings:
        print(f"[WARN] {w}")

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")

    # Dry run — just show cost estimate
    if args.dry_run:
        est = estimate_cost(spec, args.model, args.synth_model)
        print(f"\n{'=' * 60}")
        print(f"DRY RUN: {spec.title}")
        print(f"{'=' * 60}")
        print(f"Panel: {panel.name} ({len(panel.experts)} experts)")
        print(f"Rounds: {est['total_rounds']}")
        print(f"API calls: {est['total_api_calls']}")
        print(f"Est. input tokens: {est['estimated_input_tokens']:,}")
        print(f"Est. output tokens: {est['estimated_output_tokens']:,}")
        print(f"Est. agent cost: ${est['estimated_agent_cost_usd']:.2f}")
        print(f"Est. synth cost: ${est['estimated_synth_cost_usd']:.2f}")
        print(f"Est. TOTAL cost: ${est['estimated_total_cost_usd']:.2f}")
        print(f"\nRound breakdown:")
        for r in est["rounds"]:
            tag = " [SYNTH]" if r["is_synthesis"] else ""
            print(f"  R{r['round']}: {r['agents']} agents — {r['focus']}{tag}")
        return

    if not api_key:
        print("ERROR: No API key. Use --api-key or set ANTHROPIC_API_KEY.")
        sys.exit(1)

    runner = DebateRunner(
        spec=spec,
        panel=panel,
        api_key=api_key,
        model=args.model,
        synth_model=args.synth_model,
        output_dir=args.output_dir,
        use_memory=not args.no_memory,
    )
    runner.run()


def cmd_list(args):
    """List available panels and specs."""
    from think_tank.loader import discover_files, load_panel, load_spec

    pkg_root = os.path.dirname(os.path.dirname(__file__))

    panels_dir = args.panels_dir or os.path.join(pkg_root, "panels")
    specs_dir = args.specs_dir or os.path.join(pkg_root, "specs")

    print(f"\n{'=' * 60}")
    print("AVAILABLE PANELS")
    print(f"{'=' * 60}")
    panels = discover_files(panels_dir)
    if panels:
        for path, stem in panels:
            try:
                panel = load_panel(path)
                print(f"  {stem}: {panel.name} ({len(panel.experts)} experts)")
                if panel.description:
                    print(f"    {panel.description[:80]}")
            except Exception as e:
                print(f"  {stem}: [ERROR] {e}")
    else:
        print("  (none found)")

    print(f"\n{'=' * 60}")
    print("AVAILABLE SPECS")
    print(f"{'=' * 60}")
    specs = discover_files(specs_dir)
    if specs:
        for path, stem in specs:
            try:
                spec = load_spec(path)
                print(f"  {stem}: {spec.title} ({len(spec.rounds)} rounds)")
            except Exception as e:
                print(f"  {stem}: [ERROR] {e}")
    else:
        print("  (none found)")

    print()


def cmd_check_forecasts(args):
    """Check forecast tracking status."""
    from think_tank.memory import MemoryManager

    pkg_root = os.path.dirname(os.path.dirname(__file__))
    memory_dir = args.memory_dir or os.path.join(pkg_root, "memory")
    mm = MemoryManager(memory_dir)
    print(mm.check_forecasts())


def cmd_replay(args):
    """Replay a completed debate from its state file."""
    from think_tank.schemas import DebateState
    from think_tank.cost import compute_actual_cost

    with open(args.state_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    state = DebateState.from_dict(data)

    print(f"\n{'=' * 60}")
    print(f"REPLAY: {state.spec_title}")
    print(f"{'=' * 60}")
    print(f"Panel: {state.panel_name} ({state.num_experts} experts)")
    print(f"Rounds: {state.num_rounds}")
    print(f"Moves: {len(state.moves)}")
    print(f"Claims: {state.total_claims}")
    print(f"Started: {state.started_at}")
    print(f"Finished: {state.finished_at}")

    cost = compute_actual_cost(state)
    print(f"Cost: ${cost['total_cost_usd']:.4f}\n")

    current_round = 0
    for move in state.moves:
        if move.round != current_round:
            current_round = move.round
            print(f"\n--- Round {current_round} ---\n")

        print(f"[{move.agent_title}] ({move.move_type})")
        print(f"  {move.content[:200]}...")
        if move.claims:
            print(f"  Claims: {len(move.claims)}")
            for c in move.claims[:3]:
                print(f"    [{c.confidence:.2f}] {c.text[:100]}")
        print()


def cmd_resolve(args):
    """Resolve a forecast."""
    from think_tank.memory import MemoryManager

    pkg_root = os.path.dirname(os.path.dirname(__file__))
    memory_dir = args.memory_dir or os.path.join(pkg_root, "memory")
    mm = MemoryManager(memory_dir)

    outcome = args.outcome.lower() in ("true", "yes", "1", "y")
    mm.resolve_forecast(args.forecast_id, outcome)
    print(f"Resolved forecast '{args.forecast_id}' as {'YES' if outcome else 'NO'}")
    print(mm.check_forecasts())


def main():
    parser = argparse.ArgumentParser(
        prog="think-tank",
        description="LLM-powered multi-expert structured debate tool",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # ── run ──
    p_run = sub.add_parser("run", help="Run a debate")
    p_run.add_argument("--spec", required=True, help="Path to debate spec YAML")
    p_run.add_argument("--panel", required=True, help="Path to expert panel YAML")
    p_run.add_argument("--api-key", default=None,
                       help="Anthropic API key (or ANTHROPIC_API_KEY env)")
    p_run.add_argument("--model", default="claude-sonnet-4-6",
                       help="Model for agents (default: claude-sonnet-4-6)")
    p_run.add_argument("--synth-model", default="claude-opus-4-6",
                       help="Model for synthesis (default: claude-opus-4-6)")
    p_run.add_argument("--output-dir", default=None, help="Output directory")
    p_run.add_argument("--dry-run", action="store_true",
                       help="Estimate cost without running")
    p_run.add_argument("--no-memory", action="store_true",
                       help="Disable self-development memory")
    p_run.set_defaults(func=cmd_run)

    # ── list ──
    p_list = sub.add_parser("list", help="List available panels and specs")
    p_list.add_argument("--panels-dir", default=None)
    p_list.add_argument("--specs-dir", default=None)
    p_list.set_defaults(func=cmd_list)

    # ── check-forecasts ──
    p_fc = sub.add_parser("check-forecasts", help="Check forecast tracking")
    p_fc.add_argument("--memory-dir", default=None)
    p_fc.set_defaults(func=cmd_check_forecasts)

    # ── replay ──
    p_replay = sub.add_parser("replay", help="Replay a completed debate")
    p_replay.add_argument("state_file", help="Path to debate_state.json")
    p_replay.set_defaults(func=cmd_replay)

    # ── resolve ──
    p_resolve = sub.add_parser("resolve", help="Resolve a forecast")
    p_resolve.add_argument("forecast_id", help="Forecast ID to resolve")
    p_resolve.add_argument("outcome", help="Outcome: true/false/yes/no")
    p_resolve.add_argument("--memory-dir", default=None)
    p_resolve.set_defaults(func=cmd_resolve)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
