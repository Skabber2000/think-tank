"""DebateAgent — LLM-powered expert persona for structured debate."""

from __future__ import annotations

import json
from datetime import datetime
from typing import List, Optional

from anthropic import Anthropic

from think_tank.schemas import Expert, Move, Claim, Evidence, RoundSpec


class DebateAgent:
    """A think tank expert agent that participates in structured debate."""

    def __init__(
        self,
        expert: Expert,
        client: Anthropic,
        model: str = "claude-sonnet-4-6",
        is_synthesizer: bool = False,
    ):
        self.expert = expert
        self.client = client
        self.model = model
        self.is_synthesizer = is_synthesizer

    @property
    def agent_id(self) -> str:
        return self.expert.id

    def make_move(
        self,
        round_spec: RoundSpec,
        problem_context: str,
        prior_moves: List[Move],
        move_id: str,
        memory_context: str = "",
        synthesizer_prompt: str = "",
    ) -> Move:
        """Generate a debate contribution with structured claims."""

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            round_spec, problem_context, prior_moves, move_id,
            memory_context, synthesizer_prompt,
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=6000,
            temperature=0.4,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        text = response.content[0].text
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        move = self._parse_response(text, move_id, round_spec.number)
        move.input_tokens = input_tokens
        move.output_tokens = output_tokens
        return move

    def _build_system_prompt(self) -> str:
        if self.is_synthesizer:
            return (
                "You are the DEBATE SYNTHESIZER. Your role is to consolidate "
                "all prior expert contributions into a unified, actionable "
                "report. You are neutral, rigorous, and focused on practical "
                "output. Identify consensus, dissensus, and critical "
                "uncertainties. Produce structured recommendations."
            )

        e = self.expert
        parts = [f"You are {e.name}, {e.title}."]

        if e.background:
            parts.append(f"\nBackground: {e.background}")
        if e.bias:
            parts.append(f"\nAnalytical bias: {e.bias}")
        if e.lens:
            parts.append(f"\nYour critical lens: {e.lens}")

        parts.append(
            "\n\nYou are participating in a structured think tank debate. "
            "RULES:\n"
            "- Be specific, quantitative, and evidence-based\n"
            "- Reference real institutions, regulations, and programmes by name\n"
            "- Challenge prior claims if you disagree, citing evidence\n"
            "- Provide realistic timelines based on your direct experience\n"
            "- Distinguish between what is proven and what is aspirational\n"
            "- When giving estimates, provide ranges, not point values"
        )

        return "\n".join(parts)

    def _build_user_prompt(
        self,
        round_spec: RoundSpec,
        problem_context: str,
        prior_moves: List[Move],
        move_id: str,
        memory_context: str = "",
        synthesizer_prompt: str = "",
    ) -> str:
        parts = [problem_context]

        # Inject memory context (lessons from prior debates)
        if memory_context:
            parts.append(f"\n# LESSONS FROM PRIOR DEBATES\n{memory_context}")

        # Prior debate moves
        if prior_moves:
            parts.append("\n# PRIOR DEBATE MOVES\n")
            limit = len(prior_moves) if self.is_synthesizer else 6
            for m in prior_moves[-limit:]:
                parts.append(
                    f"\n## [{m.agent_title}] Round {m.round} — {m.move_type}"
                )
                parts.append(m.content[:1500])
                if m.claims:
                    for c in m.claims[:4]:
                        parts.append(
                            f"  - [{c.confidence:.2f}] {c.text[:250]}"
                        )

        parts.append(f"\n# ROUND {round_spec.number}: {round_spec.focus}\n")
        parts.append(f"**Question**: {round_spec.question}\n")

        if self.is_synthesizer:
            prompt = synthesizer_prompt or (
                "Produce the FINAL SYNTHESIS as a comprehensive Markdown document. "
                "Be concrete — specific recommendations, specific timelines, "
                "specific evidence. This is the final output."
            )
            parts.append(prompt)
        else:
            parts.append(
                "Respond with a JSON object containing:\n"
                "```json\n"
                "{\n"
                '  "move_type": "claim" | "object" | "defend",\n'
                '  "content": "Your main analysis (500-1000 words)",\n'
                '  "claims": [\n'
                "    {\n"
                f'      "id": "{move_id}_C1",\n'
                '      "text": "Specific, falsifiable claim",\n'
                '      "confidence": 0.0-1.0,\n'
                '      "evidence": [{"source": "...", "quote": "..."}],\n'
                '      "assumptions": ["..."],\n'
                '      "stance": "pro" | "contra" | "neutral"\n'
                "    }\n"
                "  ],\n"
                '  "targets": ["claim_id to challenge, if any"]\n'
                "}\n"
                "```\n"
                "Provide 3-6 claims per move. Be specific and evidence-based."
            )

        return "\n".join(parts)

    def _parse_response(self, text: str, move_id: str, round_num: int) -> Move:
        """Parse agent response into a structured Move."""

        if self.is_synthesizer:
            return Move(
                move_id=move_id,
                agent_id=self.agent_id,
                agent_title="Debate Synthesizer",
                round=round_num,
                move_type="synthesize",
                content=text,
            )

        # Try to parse JSON from response
        try:
            json_text = text
            if "```json" in text:
                start = text.index("```json") + 7
                end = text.index("```", start)
                json_text = text[start:end].strip()
            elif "```" in text:
                start = text.index("```") + 3
                end = text.index("```", start)
                json_text = text[start:end].strip()

            data = json.loads(json_text)

            claims = []
            for c in data.get("claims", []):
                evidence = [
                    Evidence(source=e.get("source", ""), quote=e.get("quote", ""))
                    for e in c.get("evidence", [])
                ]
                claims.append(Claim(
                    id=c.get("id", ""),
                    text=c.get("text", ""),
                    confidence=float(c.get("confidence", 0.7)),
                    evidence=evidence,
                    assumptions=c.get("assumptions", []),
                    stance=c.get("stance", "neutral"),
                ))

            return Move(
                move_id=move_id,
                agent_id=self.agent_id,
                agent_title=self.expert.display_title(),
                round=round_num,
                move_type=data.get("move_type", "claim"),
                content=data.get("content", text[:2000]),
                claims=claims,
                targets=data.get("targets", []),
            )

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"    [WARN] JSON parse failed for {self.agent_id}: {e}")
            return Move(
                move_id=move_id,
                agent_id=self.agent_id,
                agent_title=self.expert.display_title(),
                round=round_num,
                move_type="claim",
                content=text[:3000],
            )
