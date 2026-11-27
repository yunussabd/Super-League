"""
Step 3 - Elo model (Model A), following Hvattum & Arntzen (2010).

How it works
  1. Every team has a rating (start 1500). After each match the ratings move:
     a surprise result moves them more, a bigger goal difference moves them more.
  2. Before each match we take the rating difference (home - away) and feed it
     to an ordered logit model, which turns it into home / draw / away probabilities.
  3. The settings are tuned on past seasons only (rolling origin):
     predict 2022/23 from 2021/22, predict 2023/24 from 2021/22-2022/23, and so on.
     2025/26 is the locked test season and is NOT predicted here.

    python src/elo.py

Output:
    results/elo_tuning.csv        RPS for every combination of settings tried
    results/elo_validation.csv    best Elo vs bookmaker vs naive, per season
    results/elo_predictions.csv   Elo probabilities for every validation match
    results/elo_ratings.csv       team ratings at the end of 2024/25
"""
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel

warnings.filterwarnings("ignore")  # statsmodels prints convergence chatter

DATA = Path("data/processed/matches.csv")
OUT = Path("results")
TEST_SEASON = "2025/26"
START = 1500

# settings to try (grid search)
GRID = {
    "k": [5, 10, 20, 30],                 # how fast ratings move
    "home": [50, 100, 150],              # home advantage (rating points) used in the update
    "goal_power": [0, 0.5, 1.0, 1.5],   # 0 = ignore goal difference, 1 = strong effect
    "carry": [0.5, 0.6, 0.7, 0.85, 1.0],  # share of rating kept between seasons (rest pulled to 1500)
}


# ------------------------------------------------------------------ RPS
def rps(probs, result):
    """Ranked Probability Score per match. probs columns: home, draw, away."""
    actual = pd.get_dummies(pd.Series(result)).reindex(columns=["H", "D", "A"], fill_value=0).to_numpy(float)
    diff = np.cumsum(np.asarray(probs), axis=1) - np.cumsum(actual, axis=1)
    return (diff[:, :2] ** 2).sum(axis=1) / 2


# ------------------------------------------------------------------ 1. Elo ratings
def elo_diffs(df, k, home, goal_power, carry):
    """Walk through matches in date order. For each match return the PRE-match
    rating difference (home - away), so no match ever uses its own result."""
    ratings = {}
    prev_teams = set()
    diffs = np.empty(len(df))
    season = None
    for i, m in enumerate(df.itertuples(index=False)):
        if m.season != season:  # new season: pull ratings towards 1500, place promoted teams
            teams = set(df.loc[df.season == m.season, "home"]) | set(df.loc[df.season == m.season, "away"])
            ratings = {t: START + carry * (r - START) for t, r in ratings.items()}
            left = prev_teams - teams                         # relegated teams
            newcomer = np.mean([ratings[t] for t in left]) if left else START
            for t in teams - set(ratings):
                ratings[t] = newcomer                          # promoted teams start where relegated ones ended
            prev_teams, season = teams, m.season
        rh, ra = ratings[m.home], ratings[m.away]
        diffs[i] = rh - ra
        expected_home = 1 / (1 + 10 ** (-(rh + home - ra) / 400))
        actual_home = {"H": 1.0, "D": 0.5, "A": 0.0}[m.result]
        change = k * (1 + abs(m.home_goals - m.away_goals)) ** goal_power * (actual_home - expected_home)
        ratings[m.home] = rh + change
        ratings[m.away] = ra - change
    return diffs, ratings


# ------------------------------------------------------------------ 2. ordered logit
def fit_predict(train_x, train_y, test_x):
    """Ordered logit: rating difference -> P(away), P(draw), P(home)."""
    y = pd.Series(pd.Categorical(train_y, categories=["A", "D", "H"], ordered=True))
    model = OrderedModel(y, pd.DataFrame({"diff": train_x / 100}), distr="logit")
    fit = model.fit(method="bfgs", disp=False)
    p = np.asarray(fit.predict(pd.DataFrame({"diff": test_x / 100})))   # columns: A, D, H
    return np.column_stack([p[:, 2], p[:, 1], p[:, 0]])       # reorder to H, D, A


# ------------------------------------------------------------------ 3. rolling-origin validation
def validate(df, diffs, seasons):
    """For each validation season, fit the logit on all EARLIER seasons and predict it."""
    preds = []
    for s in seasons:
        past = (df.season < s).to_numpy()
        cur = (df.season == s).to_numpy()
        p = fit_predict(diffs[past], df.result[past], diffs[cur])
        preds.append(pd.DataFrame({"match_id": df.match_id[cur].to_numpy(), "season": s,
                                   "elo_diff": diffs[cur], "elo_p_home": p[:, 0],
                                   "elo_p_draw": p[:, 1], "elo_p_away": p[:, 2]}))
    return pd.concat(preds, ignore_index=True)


def main():
    df = pd.read_csv(DATA, parse_dates=["date"]).sort_values(["date", "match_id"]).reset_index(drop=True)
    seasons = sorted(df.season.unique())
    val_seasons = [s for s in seasons[1:] if s != TEST_SEASON]   # first season is warm-up only
    train = df[df.season != TEST_SEASON].reset_index(drop=True)  # the test season is never touched
    print(f"Tuning on {', '.join(val_seasons)} ({train.season.isin(val_seasons).sum()} matches); "
          f"{TEST_SEASON} stays locked.\n")

    # --- grid search
    rows = []
    combos = list(itertools.product(*GRID.values()))
    for n, values in enumerate(combos, 1):
        params = dict(zip(GRID, values))
        diffs, _ = elo_diffs(train, **params)
        pred = validate(train, diffs, val_seasons)
        res = train.set_index("match_id").loc[pred.match_id, "result"]
        score = rps(pred[["elo_p_home", "elo_p_draw", "elo_p_away"]], res.to_numpy()).mean()
        rows.append({**params, "rps": score})
        if n % 50 == 0:
            print(f"  tried {n}/{len(combos)} settings")
    tuning = pd.DataFrame(rows).sort_values("rps").reset_index(drop=True)
    best = tuning.iloc[0][list(GRID)].to_dict()
    best = {k: (int(v) if k in ("k", "home") else float(v)) for k, v in best.items()}
    print("\nBest settings:", best, f"-> mean RPS {tuning.rps.iloc[0]:.4f}")
    print("Top 5 settings:\n", tuning.head().round(4).to_string(index=False), sep="")

    # --- best model: per-season comparison with the bookmaker and the naive forecast
    diffs, ratings = elo_diffs(train, **best)
    pred = validate(train, diffs, val_seasons)
    m = pred.merge(train[["match_id", "result", "p_home", "p_draw", "p_away"]], on="match_id")
    m["elo_rps"] = rps(m[["elo_p_home", "elo_p_draw", "elo_p_away"]], m.result)
    m["book_rps"] = rps(m[["p_home", "p_draw", "p_away"]], m.result)
    naive = []
    for s in val_seasons:  # naive: home/draw/away rates of earlier seasons
        rates = train[train.season < s].result.value_counts(normalize=True)
        cur = m.season == s
        naive.append(pd.Series(rps(np.tile([rates["H"], rates["D"], rates["A"]], (cur.sum(), 1)),
                                   m.result[cur]), index=m.index[cur]))
    m["naive_rps"] = pd.concat(naive)

    table = m.groupby("season")[["elo_rps", "book_rps", "naive_rps"]].mean()
    table.loc["all validation"] = m[["elo_rps", "book_rps", "naive_rps"]].mean()
    table["matches"] = list(m.groupby("season").size()) + [len(m)]
    gap = m.elo_rps - m.book_rps
    print("\nRPS per season (lower = better):\n", table.round(4).to_string(), sep="")
    print(f"\nGap Elo - bookmaker: {gap.mean():+.4f} "
          f"(95% CI {gap.mean() - 1.96 * gap.sem():+.4f} to {gap.mean() + 1.96 * gap.sem():+.4f})")
    print("Elo beats naive in", f"{(m.elo_rps < m.naive_rps).mean():.0%}", "of matches; "
          "beats bookmaker in", f"{(m.elo_rps < m.book_rps).mean():.0%}")

    # --- save
    OUT.mkdir(exist_ok=True)
    tuning.to_csv(OUT / "elo_tuning.csv", index=False)
    table.round(4).to_csv(OUT / "elo_validation.csv")
    m.to_csv(OUT / "elo_predictions.csv", index=False)
    last = set(train[train.season == val_seasons[-1]].home)
    r = pd.Series({t: v for t, v in ratings.items() if t in last}).sort_values(ascending=False).round(0)
    r.rename("elo").to_csv(OUT / "elo_ratings.csv", index_label="team")
    print(f"\nTop 5 at the end of {val_seasons[-1]}:\n", r.head().to_string(), sep="")
    print(f"\nSaved elo_tuning.csv, elo_validation.csv, elo_predictions.csv, elo_ratings.csv in {OUT}")


if __name__ == "__main__":
    main()
