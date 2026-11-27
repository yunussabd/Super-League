"""
Unit tests for src/metrics.py.

Run from the repository folder:  python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metrics import implied_probabilities, rps  # noqa: E402


class TestImpliedProbabilities(unittest.TestCase):
    def test_margin_is_removed(self):
        p_home, p_draw, p_away, margin = implied_probabilities(1.80, 3.60, 4.50)
        total = 1 / 1.80 + 1 / 3.60 + 1 / 4.50
        self.assertAlmostEqual(p_home[0] + p_draw[0] + p_away[0], 1.0)
        self.assertAlmostEqual(p_home[0], (1 / 1.80) / total)
        self.assertAlmostEqual(margin[0], total - 1)

    def test_fair_odds_have_no_margin(self):
        p_home, p_draw, p_away, margin = implied_probabilities(2.0, 4.0, 4.0)
        np.testing.assert_allclose([p_home[0], p_draw[0], p_away[0]], [0.5, 0.25, 0.25])
        self.assertAlmostEqual(margin[0], 0.0)

    def test_works_on_whole_columns(self):
        p_home, p_draw, p_away, _ = implied_probabilities([1.5, 2.5], [4.0, 3.2], [6.0, 2.9])
        self.assertEqual(len(p_home), 2)
        np.testing.assert_allclose(p_home + p_draw + p_away, [1.0, 1.0])

    def test_rejects_impossible_odds(self):
        with self.assertRaises(ValueError):
            implied_probabilities(1.0, 3.0, 4.0)


class TestRPS(unittest.TestCase):
    def test_perfect_forecast_scores_zero(self):
        probs = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        np.testing.assert_allclose(rps(probs, ["H", "D", "A"]), [0, 0, 0])

    def test_opposite_forecast_scores_one(self):
        self.assertAlmostEqual(rps([[0, 0, 1]], ["H"])[0], 1.0)

    def test_known_value(self):
        # cumulative differences 0.5 and 0.8 -> (0.25 + 0.64) / 2 = 0.445
        self.assertAlmostEqual(rps([[0.5, 0.3, 0.2]], ["A"])[0], 0.445)

    def test_draw_is_closer_than_the_other_win(self):
        self.assertLess(rps([[0, 1, 0]], ["H"])[0], rps([[0, 0, 1]], ["H"])[0])

    def test_rejects_unknown_result(self):
        with self.assertRaises(ValueError):
            rps([[0.5, 0.3, 0.2]], ["X"])


if __name__ == "__main__":
    unittest.main()
