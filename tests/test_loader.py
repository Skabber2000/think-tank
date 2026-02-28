"""Tests for think_tank.loader."""

import os
import tempfile

import yaml

from think_tank.loader import load_panel, load_spec, validate_spec_against_panel


class TestLoadPanel:
    def test_load_cluster_format(self, tmp_path):
        data = {
            "name": "Test Panel",
            "description": "A test",
            "experts": [
                {"id": "e1", "name": "Expert 1", "title": "Title 1",
                 "background": "Background 1"},
                {"id": "e2", "name": "Expert 2", "title": "Title 2",
                 "background": "Background 2"},
            ],
        }
        path = tmp_path / "panel.yaml"
        with open(path, "w") as f:
            yaml.dump(data, f)

        panel = load_panel(str(path))
        assert panel.name == "Test Panel"
        assert len(panel.experts) == 2
        assert panel.experts[0].id == "e1"
        assert panel.experts[0].background == "Background 1"

    def test_load_ww3_format(self, tmp_path):
        data = {
            "name": "WW3 Panel",
            "experts": [
                {"id": "b1", "name": "Gen. Test", "role": "DIA Director",
                 "bias": "Rigor", "lens": "IC standards?"},
            ],
        }
        path = tmp_path / "ww3.yaml"
        with open(path, "w") as f:
            yaml.dump(data, f)

        panel = load_panel(str(path))
        assert panel.experts[0].title == "DIA Director"
        assert panel.experts[0].bias == "Rigor"
        assert panel.experts[0].lens == "IC standards?"

    def test_load_real_panel(self):
        """Load the actual geopolitical_42.yaml panel."""
        panels_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "panels")
        path = os.path.join(panels_dir, "geopolitical_42.yaml")
        if not os.path.exists(path):
            return  # Skip if not available

        panel = load_panel(path)
        assert panel.name == "Geopolitical 42"
        assert len(panel.experts) == 42


class TestLoadSpec:
    def test_load(self, tmp_path):
        data = {
            "title": "Test Debate",
            "context": "Context here",
            "synthesizer_prompt": "Synthesize it",
            "rounds": [
                {"number": 1, "focus": "Intro", "question": "Q1?",
                 "agents": ["e1", "e2"]},
                {"number": 2, "focus": "Synthesis", "question": "Q2?",
                 "agents": ["synthesizer"]},
            ],
        }
        path = tmp_path / "spec.yaml"
        with open(path, "w") as f:
            yaml.dump(data, f)

        spec = load_spec(str(path))
        assert spec.title == "Test Debate"
        assert len(spec.rounds) == 2
        assert spec.rounds[0].agents == ["e1", "e2"]

    def test_load_real_spec(self):
        """Load the actual ww3_project_review.yaml spec."""
        specs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "specs")
        path = os.path.join(specs_dir, "ww3_project_review.yaml")
        if not os.path.exists(path):
            return

        spec = load_spec(path)
        assert spec.title.startswith("WW3")
        assert len(spec.rounds) == 10


class TestValidation:
    def test_valid(self, tmp_path):
        panel_data = {
            "name": "P",
            "experts": [
                {"id": "e1", "name": "E1", "title": "T1"},
                {"id": "e2", "name": "E2", "title": "T2"},
            ],
        }
        spec_data = {
            "title": "S",
            "context": "C",
            "rounds": [
                {"number": 1, "focus": "F", "question": "Q",
                 "agents": ["e1", "e2"]},
                {"number": 2, "focus": "S", "question": "Q",
                 "agents": ["synthesizer"]},
            ],
        }

        pp = tmp_path / "panel.yaml"
        sp = tmp_path / "spec.yaml"
        with open(pp, "w") as f:
            yaml.dump(panel_data, f)
        with open(sp, "w") as f:
            yaml.dump(spec_data, f)

        panel = load_panel(str(pp))
        spec = load_spec(str(sp))
        warnings = validate_spec_against_panel(spec, panel)
        assert warnings == []

    def test_missing_agent(self, tmp_path):
        panel_data = {
            "name": "P",
            "experts": [{"id": "e1", "name": "E1", "title": "T1"}],
        }
        spec_data = {
            "title": "S",
            "context": "C",
            "rounds": [
                {"number": 1, "focus": "F", "question": "Q",
                 "agents": ["e1", "missing_agent"]},
            ],
        }

        pp = tmp_path / "panel.yaml"
        sp = tmp_path / "spec.yaml"
        with open(pp, "w") as f:
            yaml.dump(panel_data, f)
        with open(sp, "w") as f:
            yaml.dump(spec_data, f)

        panel = load_panel(str(pp))
        spec = load_spec(str(sp))
        warnings = validate_spec_against_panel(spec, panel)
        assert len(warnings) == 1
        assert "missing_agent" in warnings[0]
