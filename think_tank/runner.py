"""DebateRunner — orchestrates multi-round structured debates."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional

from anthropic import Anthropic

from think_tank.schemas import (
    Expert, Panel, DebateSpec, DebateState, Move,
)
from think_tank.agent import DebateAgent
from think_tank.memory import MemoryManager
from think_tank.report import generate_report
from think_tank.cost import compute_actual_cost


SYNTHESIZER_EXPERT = Expert(
    id="synthesizer",
    name="Synthesis Engine",
    title="Debate Synthesizer",
    background="Neutral facilitator that consolidates all expert contributions.",
)


class DebateRunner:
    """Runs a multi-round structured debate with selective participation."""

    def __init__(
        self,
        spec: DebateSpec,
        panel: Panel,
        api_key: str,
        model: str = "claude-sonnet-4-6",
        synth_model: str = "claude-opus-4-6",
        output_dir: Optional[str] = None,
        memory_dir: Optional[str] = None,
        use_memory: bool = True,
    ):
        self.spec = spec
        self.panel = panel
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.synth_model = synth_model
        self.use_memory = use_memory

        # Output directory
        if output_dir:
            self.output_dir = output_dir
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            slug = spec.title[:30].replace(" ", "_").replace("/", "-")
            self.output_dir = os.path.join("runs", f"{slug}_{ts}")
        os.makedirs(self.output_dir, exist_ok=True)

        # Memory
        if memory_dir is None:
            memory_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "memory"
            )
        self.memory = MemoryManager(memory_dir) if use_memory else None

        # Initialize agents
        self.agents: dict[str, DebateAgent] = {}
        for expert in panel.experts:
            self.agents[expert.id] = DebateAgent(
                expert=expert,
                client=self.client,
                model=model,
            )
        # Synthesizer on stronger model
        self.agents["synthesizer"] = DebateAgent(
            expert=SYNTHESIZER_EXPERT,
            client=self.client,
            model=synth_model,
            is_synthesizer=True,
        )

    def run(self) -> DebateState:
        """Execute the full debate and return the final state."""

        state = DebateState(
            spec_title=self.spec.title,
            panel_name=self.panel.name,
            model=self.model,
            synth_model=self.synth_model,
            num_experts=len(self.panel.experts),
            num_rounds=len(self.spec.rounds),
        )

        # Load memory context
        memory_context = ""
        if self.memory:
            memory_context = self.memory.load_context()

        print(f"\n{'=' * 80}")
        print(f"THINK TANK: {self.spec.title}")
        print(f"{'=' * 80}")
        print(f"Panel: {self.panel.name} ({len(self.panel.experts)} experts)")
        print(f"Rounds: {len(self.spec.rounds)}")
        print(f"Model: {self.model} (synthesis: {self.synth_model})")
        print(f"Output: {self.output_dir}")
        if memory_context:
            print(f"Memory: {len(memory_context)} chars loaded")
        print()

        move_counter = 1

        for rnd in self.spec.rounds:
            print(f"\n{'=' * 80}")
            print(f"ROUND {rnd.number}/{len(self.spec.rounds)}: {rnd.focus}")
            print(f"Agents: {', '.join(rnd.agents)}")
            print(f"{'=' * 80}\n")

            for agent_id in rnd.agents:
                agent = self.agents.get(agent_id)
                if agent is None:
                    print(f"  [SKIP] Agent '{agent_id}' not found in panel")
                    continue

                move_id = f"M{move_counter:03d}"
                title = agent.expert.display_title()
                print(f"  [{title}] Deliberating...")

                try:
                    move = agent.make_move(
                        round_spec=rnd,
                        problem_context=self.spec.context,
                        prior_moves=state.moves,
                        move_id=move_id,
                        memory_context=memory_context,
                        synthesizer_prompt=self.spec.synthesizer_prompt,
                    )
                    state.moves.append(move)
                    state.total_input_tokens += move.input_tokens
                    state.total_output_tokens += move.output_tokens

                    # Save individual move
                    move_file = os.path.join(
                        self.output_dir,
                        f"move_{move_counter:02d}_R{rnd.number}_{agent_id}.json",
                    )
                    with open(move_file, "w", encoding="utf-8") as f:
                        json.dump(move.to_dict(), f, indent=2, ensure_ascii=False)

                    n_claims = len(move.claims)
                    preview = move.content[:120]
                    print(f"  [{title}] {n_claims} claims | {preview}...")
                    print(
                        f"    tokens: {move.input_tokens} in / "
                        f"{move.output_tokens} out"
                    )

                except Exception as e:
                    print(f"  [{title}] ERROR: {e}")
                    error_move = Move(
                        move_id=move_id,
                        agent_id=agent_id,
                        agent_title=title,
                        round=rnd.number,
                        move_type="error",
                        content=f"Agent error: {e!s}",
                    )
                    state.moves.append(error_move)

                move_counter += 1

            total_claims = state.total_claims
            print(
                f"\n  Round {rnd.number} complete. "
                f"Total moves: {len(state.moves)}, "
                f"Total claims: {total_claims}"
            )

        # Finalize
        state.finished_at = datetime.now().isoformat()

        print(f"\n{'=' * 80}")
        print("DEBATE COMPLETE")
        print(f"{'=' * 80}")

        # Save full state
        state_file = os.path.join(self.output_dir, "debate_state.json")
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, indent=2, ensure_ascii=False)

        # Generate report
        report_file = os.path.join(self.output_dir, "report.md")
        report_text = generate_report(state, self.spec, self.panel)
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_text)

        # Compute and save cost
        cost = compute_actual_cost(state)
        cost_file = os.path.join(self.output_dir, "cost.json")
        with open(cost_file, "w", encoding="utf-8") as f:
            json.dump(cost, f, indent=2)

        print(f"\nCost: ${cost['total_cost_usd']:.4f}")
        print(f"  Agent: ${cost['agent_cost_usd']:.4f}")
        print(f"  Synth: ${cost['synth_cost_usd']:.4f}")

        # Post-debate: extract lessons
        if self.memory:
            print("\nExtracting lessons from debate...")
            try:
                self.memory.extract_lessons_from_debate(
                    state, self.client, self.synth_model
                )
                self.memory.update_panel_performance(state)
                print("  Lessons saved to memory/")
            except Exception as e:
                print(f"  [WARN] Lesson extraction failed: {e}")

        print(f"\n[SAVED] Results in: {self.output_dir}")
        print(f"  - debate_state.json ({len(state.moves)} moves)")
        print(f"  - report.md")
        print(f"  - cost.json")
        print(f"  - move_XX_RY_agent.json (per-move details)")

        return state
