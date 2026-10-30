# Süper Lig prediction vs. bookmaker odds

## Research question

How close can a prediction model built on free public data come to bookmaker
odds in forecasting Turkish Süper Lig match results, and does it add any
information that the odds miss?

## Data

Match results, shots and odds for the Turkish Süper Lig, seasons 2021/22 to
2025/26, from [Football-Data.co.uk](https://www.football-data.co.uk/).
The raw files are in `data/raw/` exactly as downloaded.

Known issues in the raw data:
- 2022/23: 29 matches awarded 3-0 after the February 2023 earthquake
  (Hatayspor, Gaziantep). These are removed during cleaning.
- 2024/25: the file has 314 of 342 matches; most of the missing ones are
  between late October and late November 2024.
- No season includes Bet365 closing odds.

## Usage

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install pandas
.\.venv\Scripts\python.exe superleague.py            # build data/processed/matches.csv
.\.venv\Scripts\python.exe superleague.py --refresh  # also re-download 2025/26
```
