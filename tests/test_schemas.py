"""Tests for think_tank.schemas."""

import json

from think_tank.schemas import (
    Expert, Evidence, Claim, Move, RoundSpec, DebateSpec,
    Panel, DebateState, Lesson, Forecast, ExpertPerformance,
)


class TestExpert:
    def test_create(self):
        e = Expert(id="test", name="Dr. Test", title="Testing Expert")
        assert e.id == "test"
        assert e.name == "Dr. Test"

    def test_display_title(self):
        e = Expert(id="test", name="Dr. Test", title="Testing Expert")
        assert e.display_title() == "Dr. Test (Testing Expert)"

    def test_to_dict(self):
        e = Expert(id="test", name="Dr. Test", title="Expert", bias="testing")
        d = e.to_dict()
        assert d["id"] == "test"
        assert d["bias"] == "testing"


class TestClaim:
    def test_create_with_evidence(self):
        ev = Evidence(source="Paper X", quote="Finding Y")
        c = Claim(
            id="C1", text="Test claim", confidence=0.9,
            evidence=[ev], assumptions=["A1"], stance="pro",
        )
        assert c.confidence == 0.9
        assert len(c.evidence) == 1

    def test_to_dict(self):
        c = Claim(id="C1", text="Test", confidence=0.5)
        d = c.to_dict()
        assert d["id"] == "C1"
        assert d["confidence"] == 0.5
        assert isinstance(d["evidence"], list)


class TestMove:
    def test_create(self):
        m = Move(
            move_id="M001", agent_id="test", agent_title="Test Agent",
            round=1, content="Test content",
        )
        assert m.move_id == "M001"
        assert m.round == 1

    def test_roundtrip(self):
        c = Claim(
            id="C1", text="Claim text", confidence=0.8,
            evidence=[Evidence(source="S1", quote="Q1")],
            assumptions=["A1"], stance="pro",
        )
        m = Move(
            move_id="M001", agent_id="test", agent_title="Test Agent",
            round=1, content="Content", claims=[c], targets=["C0"],
            input_tokens=100, output_tokens=50,
        )
        d = m.to_dict()
        m2 = Move.from_dict(d)
        assert m2.move_id == "M001"
        assert len(m2.claims) == 1
        assert m2.claims[0].text == "Claim text"
        assert m2.claims[0].evidence[0].source == "S1"
        assert m2.input_tokens == 100


class TestPanel:
    def test_get_expert(self):
        experts = [
            Expert(id="a", name="A", title="A"),
            Expert(id="b", name="B", title="B"),
        ]
        p = Panel(name="Test", experts=experts)
        assert p.get_expert("a").name == "A"
        assert p.get_expert("c") is None

    def test_list_ids(self):
        experts = [
            Expert(id="x", name="X", title="X"),
            Expert(id="y", name="Y", title="Y"),
        ]
        p = Panel(name="Test", experts=experts)
        assert p.list_ids() == ["x", "y"]


class TestDebateState:
    def test_total_claims(self):
        m1 = Move(
            move_id="M1", agent_id="a", agent_title="A", round=1,
            claims=[Claim(id="C1", text="T1"), Claim(id="C2", text="T2")],
        )
        m2 = Move(
            move_id="M2", agent_id="b", agent_title="B", round=1,
            claims=[Claim(id="C3", text="T3")],
        )
        state = DebateState(
            spec_title="Test", panel_name="Test", model="test",
            synth_model="test", num_experts=2, num_rounds=1,
            moves=[m1, m2],
        )
        assert state.total_claims == 3

    def test_roundtrip(self):
        m = Move(
            move_id="M1", agent_id="a", agent_title="A", round=1,
            claims=[Claim(id="C1", text="T1", confidence=0.9)],
        )
        state = DebateState(
            spec_title="Test", panel_name="Panel", model="sonnet",
            synth_model="opus", num_experts=5, num_rounds=3,
            moves=[m],
        )
        d = state.to_dict()
        s2 = DebateState.from_dict(d)
        assert s2.spec_title == "Test"
        assert len(s2.moves) == 1
        assert s2.moves[0].claims[0].confidence == 0.9


class TestLesson:
    def test_roundtrip(self):
        l = Lesson(
            id="L1", text="Lesson text", source_debate="Test",
            category="methodology", confidence=0.85,
        )
        d = l.to_dict()
        l2 = Lesson.from_dict(d)
        assert l2.id == "L1"
        assert l2.text == "Lesson text"


class TestForecast:
    def test_brier_score_unresolved(self):
        f = Forecast(
            id="F1", text="Test", probability=0.8,
            deadline="2026-06-01", source_debate="Test",
        )
        assert f.brier_score is None

    def test_brier_score_correct(self):
        f = Forecast(
            id="F1", text="Test", probability=0.9,
            deadline="2026-06-01", source_debate="Test",
            resolved=True, outcome=True,
        )
        assert abs(f.brier_score - 0.01) < 1e-6

    def test_brier_score_wrong(self):
        f = Forecast(
            id="F1", text="Test", probability=0.9,
            deadline="2026-06-01", source_debate="Test",
            resolved=True, outcome=False,
        )
        assert abs(f.brier_score - 0.81) < 1e-6


class TestExpertPerformance:
    def test_roundtrip(self):
        p = ExpertPerformance(
            expert_id="test", debates_participated=3,
            total_claims=15, avg_confidence=0.75,
        )
        d = p.to_dict()
        p2 = ExpertPerformance.from_dict(d)
        assert p2.expert_id == "test"
        assert p2.total_claims == 15
