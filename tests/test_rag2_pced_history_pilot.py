from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_rag2_pced_history_pilot.py"
SPEC = importlib.util.spec_from_file_location("pced_history_pilot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PcedHistoryPilotTest(unittest.TestCase):
    def test_history_zero_advantage_only_decays(self) -> None:
        current = torch.tensor([1.0, -1.0])
        result = MODULE.update_history(
            current,
            torch.zeros(2),
            decay=0.5,
            temperature=1.0,
            cap=2.0,
        )
        self.assertTrue(torch.allclose(result, torch.tensor([0.5, -0.5])))

    def test_history_uses_no_rag_zero_boundary_and_is_bounded(self) -> None:
        result = MODULE.update_history(
            torch.zeros(3),
            torch.tensor([100.0, 0.0, -100.0]),
            decay=0.95,
            temperature=1.0,
            cap=0.75,
        )
        self.assertTrue(torch.allclose(result, torch.tensor([0.75, 0.0, -0.75])))

    def test_rapid_switchbacks_counts_short_returns_only(self) -> None:
        self.assertEqual(MODULE.rapid_switchbacks([0, 1, 0, 2, 3, 2], window=3), 2)
        self.assertEqual(MODULE.rapid_switchbacks([0, 0, 0, 1], window=3), 0)
