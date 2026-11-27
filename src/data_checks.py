"""
Step 2 - data checks and first descriptive statistics.

Loads the cleaned match table, data/processed/matches.csv (one row per match,
built from the raw season files by src/load_data.py), and reports:

    1. size          rows, columns, matches and teams per season
    2. data types    the type of every column
    3. missing       missing values per column
    4. checks        duplicates, result vs. goals, odds and probabilities
    5. statistics    mean, median, standard deviation, IQR, min and max
    6. raw files     the season files in data/raw/ and the optional FootyStats xG files

Run from the repository folder:

    python src/data_checks.py

The report is printed and saved to results/data_checks.txt; the statistics
table is saved to results/summary_statistics.csv.
"""
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import implied_probabilities

ROOT = Path(__file__).resolve().parents[1]          # repository folder
DATA = ROOT / "data" / "processed" / "matches.csv"
RAW = ROOT / "data" / "raw"
FOOTYSTATS = RAW / "footystats"                     # licensed xG files, kept out of Git
OUT = ROOT / "results"

ODDS = ["b365_h", "b365_d", "b365_a"]
PROBS = ["p_home", "p_draw", "p_away"]
COUNTS = ["home_goals", "away_goals", "home_shots", "away_shots", "home_sot", "away_sot"]
STATS = ["home_goals", "away_goals", "total_goals", "home_shots", "away_shots",
         "home_sot", "away_sot"] + ODDS + ["margin_pct"]

log = logging.getLogger("data_checks")


def setup_logging():
    """Send every line to the screen and to results/data_checks.txt."""
    OUT.mkdir(exist_ok=True)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # never crash on a character the console can't show
    log.setLevel(logging.INFO)
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(OUT / "data_checks.txt", mode="w", encoding="utf-8")):
        handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(handler)


def heading(title):
    log.info("")
    log.info(title)
    log.info("-" * len(title))


# ------------------------------------------------------------------ 1. size
def report_size(df):
    heading("1. Size")
    log.info(f"{len(df):,} matches and {df.shape[1]} columns, "
             f"{df.date.min():%d %b %Y} to {df.date.max():%d %b %Y}, {df.home.nunique()} different teams")
    per = df.groupby("season").agg(matches=("result", "size"), teams=("home", "nunique"),
                                   first_match=("date", "min"), last_match=("date", "max"))
    per["full_season"] = per.teams * (per.teams - 1)  # every team plays every other team twice
    per["difference"] = per.matches - per.full_season
    log.info(per.to_string())
    log.info("full_season = teams x (teams - 1). A negative difference means matches are missing "
             "or were removed (load_data.py drops awarded results).")


# ------------------------------------------------------------------ 2. types
def report_types(df):
    heading("2. Data types")
    types = df.dtypes.astype(str)
    for dtype in sorted(types.unique()):
        log.info(f"{dtype:<15} {', '.join(types.index[types == dtype])}")


# ------------------------------------------------------------------ 3. missing values
def report_missing(df):
    heading("3. Missing values")
    missing = df.isna().sum()
    table = pd.DataFrame({"missing": missing, "percent": (missing / len(df) * 100).round(1)})
    table = table[table.missing > 0]
    if table.empty:
        log.info("No missing values.")
        return table
    log.info(table.to_string())
    log.info(f"The other {df.shape[1] - len(table)} columns are complete.")
    partly = [c for c in table.index if table.loc[c, "percent"] < 100]
    if partly:
        rows = df[df[partly].isna().any(axis=1)]
        if len(rows) <= 10:
            log.info("Matches with a gap:")
            log.info(rows[["season", "date", "home", "away"] + partly].to_string(index=False))
    return table


# ------------------------------------------------------------------ 4. consistency checks
def run_checks(df):
    """Each check counts the rows that break a rule; 0 rows means PASS."""
    heading("4. Consistency checks")
    from_goals = np.select([df.home_goals > df.away_goals, df.home_goals == df.away_goals], ["H", "D"], "A")
    p_home, p_draw, p_away, margin = implied_probabilities(df.b365_h, df.b365_d, df.b365_a)
    recomputed = np.column_stack([p_home, p_draw, p_away])
    year = df.season.str[:4].astype(int)
    season_start = pd.to_datetime(year.astype(str) + "-07-01")
    season_end = pd.to_datetime((year + 1).astype(str) + "-07-01")
    checks = {
        "no duplicate matches (same date and teams)": df.duplicated(["date", "home", "away"]).sum(),
        "no team plays itself": (df.home == df.away).sum(),
        "result agrees with the goals": (from_goals != df.result).sum(),
        "goals and shots are never negative": (df[COUNTS] < 0).any(axis=1).sum(),
        "shots on target never exceed shots": ((df.home_sot > df.home_shots)
                                               | (df.away_sot > df.away_shots)).sum(),
        "every match falls inside its season (Jul-Jun)": ((df.date < season_start)
                                                          | (df.date >= season_end)).sum(),
        "Bet365 odds are above 1.0": (df[ODDS] <= 1).any(axis=1).sum(),
        "fair probabilities add up to 1": (np.abs(df[PROBS].sum(axis=1) - 1) > 1e-6).sum(),
        "stored probabilities match the odds": (np.abs(recomputed - df[PROBS].to_numpy()) > 1e-6)
                                               .any(axis=1).sum(),
        "stored margin matches the odds": (np.abs(margin - df.margin.to_numpy()) > 1e-6).sum(),
        "bookmaker margin between 0% and 15%": ((df.margin < 0) | (df.margin > 0.15)).sum(),
    }
    for name, bad in checks.items():
        log.info(f"{'PASS' if bad == 0 else 'FAIL'}  {name}" + ("" if bad == 0 else f"  ({bad} rows)"))
    return sum(bad == 0 for bad in checks.values()), len(checks)


# ------------------------------------------------------------------ 5. descriptive statistics
def report_statistics(df):
    heading("5. Descriptive statistics")
    data = df.assign(total_goals=df.home_goals + df.away_goals, margin_pct=df.margin * 100)[STATS]
    table = pd.DataFrame({
        "n": data.count(),
        "mean": data.mean(),
        "median": data.median(),
        "std": data.std(),                                 # dispersion: standard deviation
        "IQR": data.quantile(0.75) - data.quantile(0.25),  # dispersion: spread of the middle 50%
        "min": data.min(),
        "max": data.max(),
    })
    log.info(table.round(2).to_string())
    log.info("std = standard deviation, IQR = interquartile range, margin_pct = Bet365 margin in %")

    shares = df.result.value_counts(normalize=True).reindex(["H", "D", "A"]) * 100
    log.info(f"\nResults: home win {shares['H']:.1f}%, draw {shares['D']:.1f}%, away win {shares['A']:.1f}%")
    g = df.assign(home_win=df.result.eq("H"), draw=df.result.eq("D"), away_win=df.result.eq("A"),
                  goals=df.home_goals + df.away_goals, margin_pct=df.margin * 100).groupby("season")
    by_season = (g[["home_win", "draw", "away_win"]].mean() * 100).join(g[["goals", "margin_pct"]].mean())
    by_season.columns = ["home_win_%", "draw_%", "away_win_%", "goals_per_match", "margin_%"]
    log.info("\nBy season:\n" + by_season.round(2).to_string())
    return table


# ------------------------------------------------------------------ 6. raw files
def report_raw_files():
    heading("6. Raw files")
    files = sorted(RAW.glob("T1_*.csv"))
    if not files:
        log.info("No season files (T1_*.csv) in data/raw/.")
    for path in files:
        raw = pd.read_csv(path, encoding="latin-1").dropna(how="all")
        closing = all(c in raw.columns for c in ("B365CH", "B365CD", "B365CA"))
        log.info(f"{path.name:<13} {len(raw):>4} rows  {raw.shape[1]:>3} columns  "
                 f"closing odds: {'yes' if closing else 'no'}")
    xg_files = sorted(FOOTYSTATS.glob("*.csv")) if FOOTYSTATS.is_dir() else []
    if not xg_files:
        log.info("FootyStats xG: no files in data/raw/footystats/ yet "
                 "(licensed data, so .gitignore keeps it out of the public repository).")
    for path in xg_files:
        fs = pd.read_csv(path, encoding_errors="replace", low_memory=False)
        xg_cols = [c for c in fs.columns if "xg" in c.lower()]
        log.info(f"{path.name}: {len(fs)} rows; xG columns: {', '.join(xg_cols) or 'none found'}")
        for col in xg_cols:
            v = pd.to_numeric(fs[col], errors="coerce")
            log.info(f"    {col:<30} mean {v.mean():.2f}  median {v.median():.2f}  "
                     f"std {v.std():.2f}  missing {v.isna().sum()}")


def main():
    if not DATA.exists():
        sys.exit("data/processed/matches.csv not found. Run  python src/load_data.py  first.")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    setup_logging()
    df = pd.read_csv(DATA, parse_dates=["date"])
    log.info(f"Data checks for data/processed/matches.csv (run {datetime.now():%Y-%m-%d %H:%M})")

    report_size(df)
    report_types(df)
    missing = report_missing(df)
    passed, total = run_checks(df)
    stats = report_statistics(df)
    report_raw_files()

    heading("Summary")
    log.info(f"{len(df):,} matches in {df.season.nunique()} seasons; "
             f"{passed} of {total} consistency checks passed.")
    empty = [c for c in missing.index if missing.loc[c, "percent"] == 100]
    if empty:
        log.info(f"Empty columns (not in the source files): {', '.join(empty)}")
    stats.round(4).to_csv(OUT / "summary_statistics.csv", index_label="variable")
    log.info("Saved results/data_checks.txt and results/summary_statistics.csv")


if __name__ == "__main__":
    main()
