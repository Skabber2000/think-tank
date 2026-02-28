"""Data models for the Think Tank debate system."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Optional


# ── Expert definition ──────────────────────────────────────

@dataclass
class Expert:
    """An expert persona that participates in debate."""
    id: str
    name: str
    title: str
    background: str = ""
    bias: str = ""
    lens: str = ""
    domain: str = ""

    def display_title(self) -> str:
        return f"{self.name} ({self.title})"

    def to_dict(self) -> dict:
        return asdict(self)


# ── Claim / Evidence ───────────────────────────────────────

@dataclass
class Evidence:
    source: str
    quote: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Claim:
    """A specific, falsifiable claim made during debate."""
    id: str
    text: str
    confidence: float = 0.7
    evidence: List[Evidence] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    stance: str = "neutral"  # pro | contra | neutral

    def to_dict(self) -> dict:
        d = asdict(self)
        d["evidence"] = [e.to_dict() for e in self.evidence]
        return d


# ── Move (single agent contribution) ──────────────────────

@dataclass
class Move:
    """A single debate contribution from one agent in one round."""
    move_id: str
    agent_id: str
    agent_title: str
    round: int
    move_type: str = "claim"  # claim | object | defend | synthesize | error
    content: str = ""
    claims: List[Claim] = field(default_factory=list)
    targets: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["claims"] = [c.to_dict() for c in self.claims]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Move:
        claims = [
            Claim(
                id=c.get("id", ""),
                text=c.get("text", ""),
                confidence=c.get("confidence", 0.7),
                evidence=[Evidence(**e) for e in c.get("evidence", [])],
                assumptions=c.get("assumptions", []),
                stance=c.get("stance", "neutral"),
            )
            for c in d.get("claims", [])
        ]
        return cls(
            move_id=d.get("move_id", ""),
            agent_id=d.get("agent_id", ""),
            agent_title=d.get("agent_title", ""),
            round=d.get("round", 0),
            move_type=d.get("move_type", "claim"),
            content=d.get("content", ""),
            claims=claims,
            targets=d.get("targets", []),
            timestamp=d.get("timestamp", ""),
            input_tokens=d.get("input_tokens", 0),
            output_tokens=d.get("output_tokens", 0),
        )


# ── Round definition ───────────────────────────────────────

@dataclass
class RoundSpec:
    """A single debate round with focus question and assigned agents."""
    number: int
    focus: str
    question: str
    agents: List[str]  # agent IDs

    def to_dict(self) -> dict:
        return asdict(self)


# ── Debate specification ───────────────────────────────────

@dataclass
class DebateSpec:
    """Full problem specification for a debate."""
    title: str
    context: str
    rounds: List[RoundSpec]
    synthesizer_prompt: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["rounds"] = [r.to_dict() for r in self.rounds]
        return d


# ── Panel (collection of experts) ─────────────────────────

@dataclass
class Panel:
    """A collection of expert personas."""
    name: str
    description: str = ""
    experts: List[Expert] = field(default_factory=list)

    def get_expert(self, expert_id: str) -> Optional[Expert]:
        for e in self.experts:
            if e.id == expert_id:
                return e
        return None

    def list_ids(self) -> List[str]:
        return [e.id for e in self.experts]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["experts"] = [e.to_dict() for e in self.experts]
        return d


# ── Debate state (full run) ───────────────────────────────

@dataclass
class DebateState:
    """Complete state of a debate run."""
    spec_title: str
    panel_name: str
    model: str
    synth_model: str
    num_experts: int
    num_rounds: int
    moves: List[Move] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    finished_at: str = ""
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    @property
    def total_claims(self) -> int:
        return sum(len(m.claims) for m in self.moves)

    def to_dict(self) -> dict:
        return {
            "spec_title": self.spec_title,
            "panel_name": self.panel_name,
            "model": self.model,
            "synth_model": self.synth_model,
            "num_experts": self.num_experts,
            "num_rounds": self.num_rounds,
            "total_moves": len(self.moves),
            "total_claims": self.total_claims,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "moves": [m.to_dict() for m in self.moves],
        }

    @classmethod
    def from_dict(cls, d: dict) -> DebateState:
        state = cls(
            spec_title=d.get("spec_title", ""),
            panel_name=d.get("panel_name", ""),
            model=d.get("model", ""),
            synth_model=d.get("synth_model", ""),
            num_experts=d.get("num_experts", 0),
            num_rounds=d.get("num_rounds", 0),
            started_at=d.get("started_at", ""),
            finished_at=d.get("finished_at", ""),
            total_input_tokens=d.get("total_input_tokens", 0),
            total_output_tokens=d.get("total_output_tokens", 0),
        )
        state.moves = [Move.from_dict(m) for m in d.get("moves", [])]
        return state


# ── Self-development models ────────────────────────────────

@dataclass
class Lesson:
    """A lesson extracted from a completed debate."""
    id: str
    text: str
    source_debate: str  # spec title
    source_round: int = 0
    category: str = ""  # methodology | domain | bias | process
    confidence: float = 0.8
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Lesson:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Forecast:
    """A falsifiable prediction with a deadline for Brier score tracking."""
    id: str
    text: str
    probability: float
    deadline: str  # ISO date
    source_debate: str
    source_claim_id: str = ""
    resolved: bool = False
    outcome: Optional[bool] = None  # True = happened, False = didn't
    resolved_at: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def brier_score(self) -> Optional[float]:
        if not self.resolved or self.outcome is None:
            return None
        actual = 1.0 if self.outcome else 0.0
        return (self.probability - actual) ** 2

    def to_dict(self) -> dict:
        d = asdict(self)
        d["brier_score"] = self.brier_score
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Forecast:
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


@dataclass
class ExpertPerformance:
    """Tracks per-expert performance across debates."""
    expert_id: str
    debates_participated: int = 0
    total_claims: int = 0
    avg_confidence: float = 0.0
    challenges_received: int = 0
    challenges_made: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ExpertPerformance:
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)
