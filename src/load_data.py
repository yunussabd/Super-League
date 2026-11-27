"""Core module for the Super League project.

Step 1: download Turkish Süper Lig match data from Football-Data.co.uk,
clean it and save one tidy table to data/processed/matches.csv.

Usage:
    python src/load_data.py            # download missing files, build matches.csv
    python src/load_data.py --refresh  # also re-download the current season (2526)
"""

import argparse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

SEASONS = ["2122", "2223", "2324", "2425", "2526"]
CURRENT_SEASON = "2526"
URL = "https://www.football-data.co.uk/mmz4281/{season}/T1.csv"

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# source column -> our column name
COLUMNS = {
    "Date": "date",
    "HomeTeam": "home",
    "AwayTeam": "away",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_sot",
    "AST": "away_sot",
    "B365H": "b365_h",
    "B365D": "b365_d",
    "B365A": "b365_a",
    "B365CH": "b365c_h",
    "B365CD": "b365c_d",
    "B365CA": "b365c_a",
    "MaxH": "max_h",
    "MaxD": "max_d",
    "MaxA": "max_a",
}
# older files call the best price BbMx* instead of Max*
FALLBACK_COLUMNS = {"BbMxH": "max_h", "BbMxD": "max_d", "BbMxA": "max_a"}

NUMERIC = [c for c in COLUMNS.values() if c not in ("date", "home", "away", "result")]
ODDS = ["b365_h", "b365_d", "b365_a"]
ALL_ODDS = ODDS + ["b365c_h", "b365c_d", "b365c_a", "max_h", "max_d", "max_a"]


def season_label(season):
    """'2122' -> '2021/22'"""
    return f"20{season[:2]}/{season[2:]}"


def download(season, force=False):
    path = RAW_DIR / f"T1_{season}.csv"
    if path.exists() and not force:
        return path
    url = URL.format(season=season)
    print(f"Downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    hint = f"Download it manually and save it as {path}, then run again."
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            content = response.read()
    except OSError as err:
        raise SystemExit(f"Could not download {url}: {err}\n{hint}") from err
    if content.lstrip().lower().startswith(b"<"):
        raise SystemExit(f"{url} returned an HTML page instead of CSV.\n{hint}")
    path.write_bytes(content)  # saved untouched
    return path


def load_season(path, season):
    raw = pd.read_csv(path, encoding="latin-1")
    raw.columns = raw.columns.str.strip()

    df = pd.DataFrame(index=raw.index)
    for src, dst in COLUMNS.items():
        if src in raw.columns:
            df[dst] = raw[src]
        else:
            df[dst] = np.nan
    for src, dst in FALLBACK_COLUMNS.items():
        if src in raw.columns and df[dst].isna().all():
            df[dst] = raw[src]

    df.insert(0, "season", season_label(season))
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="mixed", errors="coerce")
    df[NUMERIC] = df[NUMERIC].apply(pd.to_numeric, errors="coerce")
    return df


def clean(df):
    n_start = len(df)

    # empty rows (trailing blank lines in the CSV)
    df = df.dropna(subset=["date", "home", "away"])
    n_empty = n_start - len(df)

    # awarded forfeits: no odds at all and a 3-0 / 0-3 score
    no_odds = df[ALL_ODDS].isna().all(axis=1)
    score_3_0 = ((df["home_goals"] == 3) & (df["away_goals"] == 0)) | (
        (df["home_goals"] == 0) & (df["away_goals"] == 3)
    )
    forfeits = df[no_odds & score_3_0]
    df = df[~(no_odds & score_3_0)]

    # keep only valid results
    valid = df["result"].isin(["H", "D", "A"])
    n_invalid = (~valid).sum()
    df = df[valid]

    print(f"  removed {n_empty} empty rows, {len(forfeits)} forfeits, {n_invalid} invalid results")
    for _, row in forfeits.iterrows():
        print(
            f"    forfeit: {row['date']:%Y-%m-%d} {row['home']} "
            f"{row['home_goals']:.0f}-{row['away_goals']:.0f} {row['away']}"
        )
    return df


def add_probabilities(df):
    inverse = 1 / df[ODDS]
    total = inverse.sum(axis=1, min_count=3)
    df = df.copy()
    df["p_home"] = inverse["b365_h"] / total
    df["p_draw"] = inverse["b365_d"] / total
    df["p_away"] = inverse["b365_a"] / total
    df["margin"] = total - 1
    return df


def print_checks(df):
    print("\n=== Matches per season ===")
    print(df.groupby("season").size().to_string())

    print("\n=== Share of matches with data, per season ===")
    coverage = df.groupby("season").agg(
        odds=("b365_h", lambda s: s.notna().mean()),
        shots=("home_shots", lambda s: s.notna().mean()),
        closing_odds=("b365c_h", lambda s: s.notna().mean()),
        best_price=("max_h", lambda s: s.notna().mean()),
    )
    print((coverage * 100).round(1).astype(str).add("%").to_string())

    print("\n=== Average Bet365 margin ===")
    print(df.groupby("season")["margin"].mean().map("{:.2%}".format).to_string())
    print(f"overall: {df['margin'].mean():.2%}")

    teams = sorted(set(df["home"]) | set(df["away"]))
    print(f"\n=== Teams ({len(teams)}) ===")
    print(", ".join(teams))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--refresh", action="store_true", help=f"re-download the current season ({CURRENT_SEASON})"
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    for season in SEASONS:
        path = download(season, force=args.refresh and season == CURRENT_SEASON)
        print(f"{season_label(season)}: {path.name}")
        frames.append(clean(load_season(path, season)))

    df = pd.concat(frames, ignore_index=True)
    df = add_probabilities(df)
    df = df.sort_values(["date", "home"], kind="stable").reset_index(drop=True)
    df.insert(0, "match_id", range(1, len(df) + 1))

    out = PROCESSED_DIR / "matches.csv"
    df.to_csv(out, index=False, date_format="%Y-%m-%d")
    print(f"\nSaved {len(df)} matches to {out}")

    print_checks(df)


if __name__ == "__main__":
    main()
