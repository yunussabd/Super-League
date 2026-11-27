"""
Validation tests for the cleaned match table, data/processed/matches.csv.

Build the table first (python src/load_data.py), then run from the repository folder:

    python -m unittest discover -s tests -v
"""
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "processed" / "matches.csv"
SEASONS = ["2021/22", "2022/23", "2023/24", "2024/25", "2025/26"]
KEY = ["season", "date", "home", "away", "home_goals", "away_goals", "result",
       "b365_h", "b365_d", "b365_a", "p_home", "p_draw", "p_away", "margin"]


@unittest.skipUnless(DATA.exists(), "data/processed/matches.csv not found: run python src/load_data.py")
class TestMatchTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(DATA, parse_dates=["date"])

    def test_has_the_needed_columns(self):
        missing = set(KEY + ["home_shots", "away_shots", "home_sot", "away_sot"]) - set(self.df.columns)
        self.assertFalse(missing, f"missing columns: {sorted(missing)}")

    def test_five_seasons_including_the_test_season(self):
        self.assertEqual(sorted(self.df.season.unique()), SEASONS)

    def test_each_season_has_a_plausible_number_of_matches(self):
        for season, n in self.df.groupby("season").size().items():
            self.assertTrue(300 <= n <= 380, f"{season}: {n} matches")

    def test_no_missing_values_in_key_columns(self):
        gaps = self.df[KEY].isna().sum()
        self.assertEqual(int(gaps.sum()), 0, f"missing values: {gaps[gaps > 0].to_dict()}")

    def test_result_agrees_with_goals(self):
        hg, ag = self.df.home_goals, self.df.away_goals
        expected = np.select([hg > ag, hg == ag], ["H", "D"], "A")
        self.assertTrue((expected == self.df.result).all())

    def test_odds_are_above_one(self):
        self.assertTrue((self.df[["b365_h", "b365_d", "b365_a"]] > 1).all().all())

    def test_probabilities_add_up_to_one(self):
        total = self.df[["p_home", "p_draw", "p_away"]].sum(axis=1)
        np.testing.assert_allclose(total, 1.0, atol=1e-9)

    def test_no_duplicate_matches(self):
        self.assertFalse(self.df.duplicated(["date", "home", "away"]).any())

    def test_no_team_plays_itself(self):
        self.assertFalse((self.df.home == self.df.away).any())

    def test_matches_fall_inside_their_season(self):
        year = self.df.season.str[:4].astype(int)
        start = pd.to_datetime(year.astype(str) + "-07-01")
        end = pd.to_datetime((year + 1).astype(str) + "-07-01")
        self.assertTrue(((self.df.date >= start) & (self.df.date < end)).all())


if __name__ == "__main__":
    unittest.main()
