"""
Step 2 - descriptive baseline (Week 6: "what is happening?").

Reads data/processed/matches.csv and describes each season:
results, goals, shots on target, bookmaker margin and how often the
bookmaker's favourite wins. It also scores two simple forecasts with RPS
on the training seasons only (2025/26 stays locked as the test season).

    python descriptive.py

Output:
    results/descriptive_by_season.csv   the table printed below
    results/figures/*.png               charts for the report
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save charts to files, no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path("data/processed/matches.csv")
OUT = Path("results")
FIG = OUT / "figures"
TEST_SEASON = "2025/26"


# ------------------------------------------------------------------ RPS
def rps(p_home, p_draw, p_away, result):
    """Ranked Probability Score for each match (0 = perfect, lower = better).
    Outcomes are ordered home -> draw -> away, so we compare cumulative probabilities."""
    actual = pd.get_dummies(result).reindex(columns=["H", "D", "A"], fill_value=0).to_numpy(float)
    pred = np.column_stack([p_home, p_draw, p_away])
    cum_diff = np.cumsum(pred, axis=1) - np.cumsum(actual, axis=1)
    return (cum_diff[:, :2] ** 2).sum(axis=1) / 2


# ------------------------------------------------------------------ table per season
def season_table(df):
    g = df.groupby("season")
    t = pd.DataFrame({
        "matches": g.size(),
        "home_win_%": g.result.apply(lambda r: (r == "H").mean() * 100),
        "draw_%": g.result.apply(lambda r: (r == "D").mean() * 100),
        "away_win_%": g.result.apply(lambda r: (r == "A").mean() * 100),
        "goals_per_match": g.total_goals.mean(),
        "home_goals": g.home_goals.mean(),
        "away_goals": g.away_goals.mean(),
        "home_sot": g.home_sot.mean(),
        "away_sot": g.away_sot.mean(),
        "margin_%": g.margin.mean() * 100,
        "favourite_wins_%": g.fav_won.mean() * 100,
    })
    return t.round(2)


# ------------------------------------------------------------------ RPS of simple forecasts
def baseline_rps(df):
    """Two simple forecasts, scored on training seasons only.
    naive: every match gets the home/draw/away rates of all EARLIER seasons
           (so it never uses information from the season it predicts).
    bookmaker: Bet365 probabilities with the margin removed."""
    train = df[df.season != TEST_SEASON]
    seasons = sorted(train.season.unique())
    rows = []
    for i, s in enumerate(seasons):
        cur = train[train.season == s]
        past = train[train.season.isin(seasons[:i])]
        row = {"season": s, "bookmaker_rps": rps(cur.p_home, cur.p_draw, cur.p_away, cur.result).mean()}
        if len(past):  # the first season has no earlier seasons to learn from
            rates = past.result.value_counts(normalize=True)
            n = len(cur)
            row["naive_rps"] = rps(np.full(n, rates["H"]), np.full(n, rates["D"]),
                                   np.full(n, rates["A"]), cur.result).mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("season").round(4)


# ------------------------------------------------------------------ charts
def charts(df, t):
    FIG.mkdir(parents=True, exist_ok=True)

    # 1. result shares per season (they add up to 100%, so stacked bars)
    ax = t[["home_win_%", "draw_%", "away_win_%"]].plot(
        kind="bar", stacked=True, color=["#2a6f97", "#a9a9a9", "#c8553d"], figsize=(7, 4), rot=0)
    ax.set_ylabel("% of matches")
    ax.set_title("Süper Lig results by season")
    ax.legend(["Home win", "Draw", "Away win"], loc="upper left", bbox_to_anchor=(1.01, 1))
    plt.tight_layout(); plt.savefig(FIG / "results_by_season.png", dpi=150); plt.close()

    # 2. bookmaker margin per season (box plot: spread across matches)
    seasons = sorted(df.season.unique())
    plt.figure(figsize=(7, 4))
    plt.boxplot([df.loc[df.season == s, "margin"].dropna() * 100 for s in seasons],
                tick_labels=seasons, showfliers=False)
    plt.ylabel("Bet365 margin (%)")
    plt.title("Bookmaker margin per match, by season")
    plt.tight_layout(); plt.savefig(FIG / "margin_by_season.png", dpi=150); plt.close()

    # 3. goals per match (skewed count data -> histogram)
    plt.figure(figsize=(7, 4))
    plt.hist(df.total_goals, bins=range(0, int(df.total_goals.max()) + 2), align="left",
             color="#2a6f97", edgecolor="white")
    plt.xlabel("Goals in the match"); plt.ylabel("Matches")
    plt.title("Goals per match, 2021/22–2025/26")
    plt.tight_layout(); plt.savefig(FIG / "goals_per_match.png", dpi=150); plt.close()

    # 4. calibration: when Bet365 says X% home win, how often does the home team win?
    hist = df[df.season != TEST_SEASON].copy()
    hist["bin"] = pd.cut(hist.p_home, bins=np.arange(0, 1.01, 0.1))
    cal = hist.groupby("bin", observed=True).agg(pred=("p_home", "mean"),
                                                 actual=("result", lambda r: (r == "H").mean()),
                                                 n=("result", "size"))
    cal = cal[cal.n >= 20]  # skip bins with too few matches
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    plt.plot(cal.pred, cal.actual, "o-", color="#2a6f97", label="Bet365")
    plt.xlabel("Bet365 home-win probability"); plt.ylabel("Actual home-win rate")
    plt.title("Are the odds calibrated? (2021/22–2024/25)")
    plt.legend(); plt.tight_layout(); plt.savefig(FIG / "calibration_home.png", dpi=150); plt.close()


def main():
    df = pd.read_csv(DATA, parse_dates=["date"])
    df["total_goals"] = df.home_goals + df.away_goals
    # bookmaker favourite = outcome with the highest probability; did it happen?
    fav = df[["p_home", "p_draw", "p_away"]].idxmax(axis=1).map({"p_home": "H", "p_draw": "D", "p_away": "A"})
    df["fav_won"] = fav == df.result

    t = season_table(df)
    b = baseline_rps(df)
    OUT.mkdir(exist_ok=True)
    t.join(b).to_csv(OUT / "descriptive_by_season.csv")
    charts(df, t)

    pd.set_option("display.width", 200)
    print("Per season:\n", t.to_string(), sep="")
    print("\nRPS of simple forecasts (lower = better; test season locked):\n", b.to_string(), sep="")
    print(f"\nAll seasons: {len(df)} matches, home win {(df.result == 'H').mean():.1%}, "
          f"draw {(df.result == 'D').mean():.1%}, away win {(df.result == 'A').mean():.1%}")
    print(f"Saved {OUT / 'descriptive_by_season.csv'} and 4 charts in {FIG}")


if __name__ == "__main__":
    main()
