# Can Free Public Data Beat the Bookmakers?

**Forecasting Turkish Süper Lig results, 2021/22–2025/26**
CS251 Integrative BCS Project I · Schiller International University, Heidelberg · Yunus Abdurahman · Fall 2026

This project measures how close forecasting models built only on free public data come to bookmaker odds when predicting Süper Lig match results (home win, draw or away win), and whether they add information the odds miss. Every method forecasts the same matches and is scored with the Ranked Probability Score (RPS; lower is better). Seasons 2021/22–2024/25 are used to build and tune the models; 2025/26 is a locked test season, predicted once at the end.

Literature, methods and variables: [`docs/midterm_revisions.md`](docs/midterm_revisions.md).

## Status

| Step | What | Script | Status |
| --- | --- | --- | --- |
| 1 | Load and clean the data | `src/load_data.py` | Done |
| 2 | Data checks and descriptive statistics | `src/data_checks.py`, `src/descriptive.py` | Done |
| 3 | Model A: Elo ratings + ordered logit | `src/elo.py` | First version |
| 4 | Model B: Dixon–Coles goals model with shots on target and xG | – | Next |
| 5 | Model C: odds and model combined; final test on 2025/26 | – | Planned |

## Repository structure

```
Super-League/
├── README.md                 project overview, setup and run steps
├── .gitignore                keeps virtual environments, caches and licensed data out of Git
├── requirements.txt          exact package versions
├── docs/
│   └── midterm_revisions.md  literature (6 sources), methods and variables
├── data/
│   ├── raw/                  season files as downloaded, never edited
│   └── processed/            matches.csv, built by src/load_data.py
├── src/
│   ├── load_data.py          step 1: clean the raw files into one table
│   ├── data_checks.py        step 2: row counts, missing values, types, descriptive statistics
│   ├── descriptive.py        step 2: per-season baseline and charts
│   ├── elo.py                step 3: Elo model (Model A)
│   └── metrics.py            shared functions: odds to probabilities, RPS
├── tests/
│   ├── test_metrics.py       unit tests for src/metrics.py
│   └── test_dataset.py       validation tests for matches.csv
└── results/                  tables, charts and reports written by the scripts
```

## Data

| Source | What | Where |
| --- | --- | --- |
| [Football-Data.co.uk](https://www.football-data.co.uk/turkeym.php) (Turkey, T1) | Results, goals, shots, shots on target, Bet365 odds, best market odds | `data/raw/T1_2122.csv` … `data/raw/T1_2526.csv` |
| [FootyStats](https://footystats.org/turkey/super-lig) | Expected goals (xG) per match | `data/raw/footystats/`: local only, because the data is licensed (`.gitignore` excludes it) |

- The raw files are never edited; all cleaning happens in `src/load_data.py`.
- `data/processed/matches.csv` has one row per match: 1,693 matches after removing awarded results.
- Known gaps in the current copy: Bet365 closing odds are empty, and 28 matches of 2024/25 are missing. Both will be fixed by downloading the original files from Football-Data.co.uk.

## Setup

Requires Python 3.12 and Git.

Windows (PowerShell):

```powershell
git clone https://github.com/yunussabd/Super-League.git
cd Super-League
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS / Linux:

```bash
git clone https://github.com/yunussabd/Super-League.git
cd Super-League
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If `pip` fails with `CERTIFICATE_VERIFY_FAILED` (a network that inspects HTTPS traffic), add `--use-feature=truststore` to the install command.

## Run

From the repository folder, with the virtual environment active:

```bash
python src/load_data.py                    # 1. data/raw/ -> data/processed/matches.csv
python src/data_checks.py                  # 2. checks + descriptive statistics -> results/data_checks.txt
python src/descriptive.py                  #    per-season tables and charts -> results/
python src/elo.py                          # 3. Elo model, tuned on 2022/23-2024/25 -> results/elo_*.csv
python -m unittest discover -s tests -v    # unit and data-validation tests
```

`src/data_checks.py` should report 1,693 matches in 5 seasons with all consistency checks passed, and the tests should all pass.

## Reproducibility

- Raw data stays as downloaded; every change happens in code.
- No setting is chosen with 2025/26 data; that season is predicted once, at the end.
- Random steps use a fixed seed (42).
- `requirements.txt` pins the exact package versions.

## AI use

AI assistance (Claude) was used for code, code explanations, debugging and help with documentation.
