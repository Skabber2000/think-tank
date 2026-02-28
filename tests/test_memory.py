"""Tests for think_tank.memory."""

import json
import os

from think_tank.memory import MemoryManager
from think_tank.schemas import (
    Lesson, Forecast, ExpertPerformance, DebateState, Move, Claim,
)


class TestMemoryManager:
    def test_load_empty(self, tmp_path):
        mm = MemoryManager(str(tmp_path))
        assert mm.load_lessons() == []
        assert mm.load_forecasts() == []

    def test_load_bootstrap(self, tmp_path):
        bootstrap = [
            {"id": "b1", "text": "Lesson 1", "source_debate": "boot",
             "category": "methodology", "confidence": 0.9,
             "created_at": "2026-01-01T00:00:00"},
        ]
        with open(tmp_path / "_bootstrap.json", "w") as f:
            json.dump(bootstrap, f)

        mm = MemoryManager(str(tmp_path))
        lessons = mm.load_lessons()
        assert len(lessons) == 1
        assert lessons[0].text == "Lesson 1"

    def test_save_and_load_lessons(self, tmp_path):
        mm = MemoryManager(str(tmp_path))
        lessons = [
            Lesson(id="L1", text="Test lesson", source_debate="Test",
                   category="methodology"),
        ]
        mm.save_lessons(lessons)
        loaded = mm.load_lessons()
        assert len(loaded) == 1
        assert loaded[0].id == "L1"

    def test_load_context(self, tmp_path):
        mm = MemoryManager(str(tmp_path))
        lessons = [
            Lesson(id="L1", text="Lesson A", source_debate="Test",
                   category="methodology"),
            Lesson(id="L2", text="Lesson B", source_debate="Test",
                   category="domain"),
        ]
        mm.save_lessons(lessons)
        ctx = mm.load_context()
        assert "methodology" in ctx
        assert "Lesson A" in ctx
        assert "Lesson B" in ctx

    def test_load_context_empty(self, tmp_path):
        mm = MemoryManager(str(tmp_path))
        assert mm.load_context() == ""


class TestForecasts:
    def test_add_and_load(self, tmp_path):
        mm = MemoryManager(str(tmp_path))
        f = Forecast(
            id="F1", text="Test forecast", probability=0.7,
            deadline="2026-06-01", source_debate="Test",
        )
        mm.add_forecast(f)
        loaded = mm.load_forecasts()
        assert len(loaded) == 1
        assert loaded[0].id == "F1"

    def test_resolve(self, tmp_path):
        mm = MemoryManager(str(tmp_path))
        f = Forecast(
            id="F1", text="Test", probability=0.8,
            deadline="2026-06-01", source_debate="Test",
        )
        mm.add_forecast(f)
        mm.resolve_forecast("F1", True)
        loaded = mm.load_forecasts()
        assert loaded[0].resolved is True
        assert loaded[0].outcome is True
        assert loaded[0].brier_score is not None

    def test_check_forecasts_empty(self, tmp_path):
        mm = MemoryManager(str(tmp_path))
        result = mm.check_forecasts()
        assert "No forecasts" in result

    def test_check_forecasts_with_data(self, tmp_path):
        mm = MemoryManager(str(tmp_path))
        mm.add_forecast(Forecast(
            id="F1", text="Pending forecast", probability=0.6,
            deadline="2026-12-01", source_debate="Test",
        ))
        mm.add_forecast(Forecast(
            id="F2", text="Resolved forecast", probability=0.9,
            deadline="2026-06-01", source_debate="Test",
            resolved=True, outcome=True,
        ))
        result = mm.check_forecasts()
        assert "Resolved" in result
        assert "Pending" in result


class TestPerformance:
    def test_update_panel_performance(self, tmp_path):
        mm = MemoryManager(str(tmp_path))
        state = DebateState(
            spec_title="Test", panel_name="Test", model="test",
            synth_model="test", num_experts=2, num_rounds=1,
        )
        state.moves = [
            Move(
                move_id="M1", agent_id="expert_a", agent_title="A",
                round=1, claims=[
                    Claim(id="M1_C1", text="Claim 1", confidence=0.8),
                    Claim(id="M1_C2", text="Claim 2", confidence=0.6),
                ], targets=[],
            ),
            Move(
                move_id="M2", agent_id="expert_b", agent_title="B",
                round=1, claims=[
                    Claim(id="M2_C1", text="Counter", confidence=0.9),
                ], targets=["M1_C1"],
            ),
        ]
        mm.update_panel_performance(state)
        perf = mm.load_performance()
        assert "expert_a" in perf
        assert "expert_b" in perf
        assert perf["expert_a"].total_claims == 2
        assert perf["expert_b"].total_claims == 1
        assert perf["expert_b"].challenges_made == 1
