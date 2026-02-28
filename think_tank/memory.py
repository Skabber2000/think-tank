"""Self-development system: lessons, forecasts, and panel performance tracking."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import List, Optional

from think_tank.schemas import (
    DebateState, Lesson, Forecast, ExpertPerformance,
)


class MemoryManager:
    """Manages persistent memory across debate runs."""

    def __init__(self, memory_dir: str):
        self.memory_dir = memory_dir
        os.makedirs(memory_dir, exist_ok=True)

        self.lessons_file = os.path.join(memory_dir, "lessons.json")
        self.forecasts_file = os.path.join(memory_dir, "forecasts.json")
        self.performance_file = os.path.join(memory_dir, "performance.json")
        self.bootstrap_file = os.path.join(memory_dir, "_bootstrap.json")

    # ── Lessons ────────────────────────────────────────────

    def load_lessons(self) -> List[Lesson]:
        """Load all lessons from disk."""
        lessons = []

        # Bootstrap lessons (seed for cold start)
        if os.path.exists(self.bootstrap_file):
            lessons.extend(self._load_json_list(self.bootstrap_file, Lesson))

        # Learned lessons
        if os.path.exists(self.lessons_file):
            lessons.extend(self._load_json_list(self.lessons_file, Lesson))

        return lessons

    def save_lessons(self, lessons: List[Lesson]):
        """Save lessons to disk (overwrites)."""
        data = [l.to_dict() for l in lessons]
        with open(self.lessons_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_context(self) -> str:
        """Build a memory context string for injection into agent prompts."""
        lessons = self.load_lessons()
        if not lessons:
            return ""

        parts = []
        for l in lessons[-20:]:  # Last 20 lessons, most recent
            parts.append(f"- [{l.category}] {l.text}")
        return "\n".join(parts)

    def extract_lessons_from_debate(
        self,
        state: DebateState,
        client,
        model: str = "claude-opus-4-6",
    ):
        """Use an LLM to extract lessons from a completed debate."""

        # Build summary of the debate
        summary_parts = [
            f"Debate: {state.spec_title}",
            f"Panel: {state.panel_name} ({state.num_experts} experts, "
            f"{state.num_rounds} rounds)",
            f"Total claims: {state.total_claims}",
            "",
        ]

        for move in state.moves:
            if move.move_type == "synthesize":
                summary_parts.append(
                    f"[SYNTHESIS] {move.content[:2000]}"
                )
            elif move.claims:
                for c in move.claims[:3]:
                    summary_parts.append(
                        f"[{move.agent_id}] [{c.confidence:.2f}] {c.text[:200]}"
                    )

        summary = "\n".join(summary_parts)

        # LLM call to extract lessons
        prompt = (
            "You are a meta-analyst reviewing a completed think tank debate. "
            "Extract 3-8 lessons that should inform future debates.\n\n"
            "For each lesson, provide:\n"
            '- "id": unique string\n'
            '- "text": the lesson (1-2 sentences)\n'
            '- "category": one of [methodology, domain, bias, process]\n'
            '- "confidence": 0.0-1.0\n\n'
            "Focus on:\n"
            "- Methodological insights (what worked, what didn't)\n"
            "- Domain knowledge that should be carried forward\n"
            "- Biases that were identified or that emerged\n"
            "- Process improvements for future debates\n\n"
            f"DEBATE SUMMARY:\n{summary[:8000]}\n\n"
            "Respond with a JSON array of lessons."
        )

        response = client.messages.create(
            model=model,
            max_tokens=2000,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text

        # Parse lessons
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

            raw_lessons = json.loads(json_text)
            if not isinstance(raw_lessons, list):
                raw_lessons = [raw_lessons]

            new_lessons = []
            for rl in raw_lessons:
                lesson = Lesson(
                    id=rl.get("id", f"L_{datetime.now().strftime('%Y%m%d%H%M%S')}"),
                    text=rl.get("text", ""),
                    source_debate=state.spec_title,
                    category=rl.get("category", "methodology"),
                    confidence=float(rl.get("confidence", 0.7)),
                )
                new_lessons.append(lesson)

            # Merge with existing
            existing = self.load_lessons()
            # Don't load bootstrap again — load_lessons already includes them
            existing_from_file = (
                self._load_json_list(self.lessons_file, Lesson)
                if os.path.exists(self.lessons_file)
                else []
            )
            existing_from_file.extend(new_lessons)
            self.save_lessons(existing_from_file)

            print(f"  Extracted {len(new_lessons)} lessons")

        except (json.JSONDecodeError, ValueError) as e:
            print(f"  [WARN] Lesson parsing failed: {e}")

    # ── Forecasts ──────────────────────────────────────────

    def load_forecasts(self) -> List[Forecast]:
        if not os.path.exists(self.forecasts_file):
            return []
        return self._load_json_list(self.forecasts_file, Forecast)

    def save_forecasts(self, forecasts: List[Forecast]):
        data = [f.to_dict() for f in forecasts]
        with open(self.forecasts_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def add_forecast(self, forecast: Forecast):
        forecasts = self.load_forecasts()
        forecasts.append(forecast)
        self.save_forecasts(forecasts)

    def resolve_forecast(self, forecast_id: str, outcome: bool):
        """Resolve a forecast and compute its Brier score."""
        forecasts = self.load_forecasts()
        for f in forecasts:
            if f.id == forecast_id:
                f.resolved = True
                f.outcome = outcome
                f.resolved_at = datetime.now().isoformat()
                break
        self.save_forecasts(forecasts)

    def check_forecasts(self) -> str:
        """Return a formatted report of all forecasts and their scores."""
        forecasts = self.load_forecasts()
        if not forecasts:
            return "No forecasts recorded."

        lines = ["# Forecast Tracker\n"]
        resolved = [f for f in forecasts if f.resolved]
        pending = [f for f in forecasts if not f.resolved]

        if resolved:
            scores = [f.brier_score for f in resolved if f.brier_score is not None]
            avg_brier = sum(scores) / len(scores) if scores else 0.0
            lines.append(f"## Resolved ({len(resolved)})")
            lines.append(f"**Average Brier Score**: {avg_brier:.4f} "
                         f"(0 = perfect, 1 = worst)\n")
            for f in resolved:
                outcome_str = "YES" if f.outcome else "NO"
                lines.append(
                    f"- [{f.brier_score:.4f}] {f.text} "
                    f"(predicted {f.probability:.0%}, actual: {outcome_str})"
                )
            lines.append("")

        if pending:
            lines.append(f"## Pending ({len(pending)})")
            for f in pending:
                lines.append(
                    f"- {f.text} (predicted {f.probability:.0%}, "
                    f"deadline: {f.deadline})"
                )

        return "\n".join(lines)

    # ── Panel Performance ──────────────────────────────────

    def load_performance(self) -> dict[str, ExpertPerformance]:
        if not os.path.exists(self.performance_file):
            return {}
        with open(self.performance_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: ExpertPerformance.from_dict(v) for k, v in data.items()}

    def save_performance(self, perf: dict[str, ExpertPerformance]):
        data = {k: v.to_dict() for k, v in perf.items()}
        with open(self.performance_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def update_panel_performance(self, state: DebateState):
        """Update expert performance metrics from a completed debate."""
        perf = self.load_performance()

        # Count challenges per agent
        all_targets = {}
        for move in state.moves:
            for target in move.targets:
                # Target format: M001_C1 → agent who made M001
                all_targets[target] = move.agent_id

        for move in state.moves:
            if move.move_type in ("error", "synthesize"):
                continue

            eid = move.agent_id
            if eid not in perf:
                perf[eid] = ExpertPerformance(expert_id=eid)

            p = perf[eid]
            p.debates_participated += 1
            n_claims = len(move.claims)
            p.total_claims += n_claims

            if n_claims > 0:
                avg_conf = sum(c.confidence for c in move.claims) / n_claims
                # Running average
                total = p.total_claims
                prev = total - n_claims
                if total > 0:
                    p.avg_confidence = (
                        (p.avg_confidence * prev + avg_conf * n_claims) / total
                    )

            p.challenges_made += len(move.targets)

            # Count how many times this agent's claims were targeted
            for claim in move.claims:
                if claim.id in all_targets:
                    p.challenges_received += 1

        self.save_performance(perf)

    # ── Helpers ────────────────────────────────────────────

    @staticmethod
    def _load_json_list(path: str, cls):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [cls.from_dict(d) for d in data]
        return []
