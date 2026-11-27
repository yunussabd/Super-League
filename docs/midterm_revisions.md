# Midterm Revisions

**Project:** Can Free Public Data Beat the Bookmakers? Forecasting the Turkish Süper Lig
**Course:** CS251 Integrative BCS Project I · Schiller International University, Heidelberg
**Author:** Yunus Abdurahman · 27 November 2026

This document closes two gaps from the midterm presentation: full citations for the literature (section 1) and the exact methods and variables of the analysis (section 2).

## 1. Literature overview

Six peer-reviewed journal articles, in the order the field developed. Each entry gives the full citation (APA 7) with its DOI, then two sentences on how it links to my problem.

1. **Maher, M. J. (1982).** Modelling association football scores. *Statistica Neerlandica, 36*(3), 109–118. https://doi.org/10.1111/j.1467-9574.1982.tb00782.x

   Maher models the goals each team scores as a Poisson variable that depends on the team's attacking strength and the opponent's defensive strength, and shows that this simple model describes real league scores reasonably well. It is the starting point for the goals model I use to turn team strengths into home-win, draw and away-win probabilities.

2. **Dixon, M. J., & Coles, S. G. (1997).** Modelling association football scores and inefficiencies in the football betting market. *Journal of the Royal Statistical Society: Series C (Applied Statistics), 46*(2), 265–280. https://doi.org/10.1111/1467-9876.00065

   Dixon and Coles turn Maher's idea into a forecasting model by correcting the probabilities of low scores and giving recent matches more weight, and they test it against bookmakers' odds, where a betting strategy based on the model makes a positive return. Their model is my Model B, and their comparison with the odds is the same comparison my project makes for the Süper Lig.

3. **Hvattum, L. M., & Arntzen, H. (2010).** Using ELO ratings for match result prediction in association football. *International Journal of Forecasting, 26*(3), 460–470. https://doi.org/10.1016/j.ijforecast.2009.10.002

   Hvattum and Arntzen turn Elo rating differences into match probabilities with an ordered logit model; their Elo forecasts beat four of six benchmarks but were significantly worse than the two based on bookmaker odds. I use the same Elo plus ordered logit method as Model A, and their result sets the expectation that the odds will stay ahead.

4. **Constantinou, A. C., & Fenton, N. E. (2012).** Solving the problem of inadequate scoring rules for assessing probabilistic football forecast models. *Journal of Quantitative Analysis in Sports, 8*(1), Article 1. https://doi.org/10.1515/1559-0410.1418

   Constantinou and Fenton show that common scoring rules ignore that home win, draw and away win are ordered, and recommend the Ranked Probability Score (RPS), which respects that order. RPS is therefore the dependent variable of my comparison: every forecasting method is judged by its RPS on the same matches.

5. **Angelini, G., & De Angelis, L. (2019).** Efficiency of online football betting markets. *International Journal of Forecasting, 35*(2), 712–721. https://doi.org/10.1016/j.ijforecast.2018.07.008

   Using Football-Data.co.uk odds from 41 bookmakers between 2006 and 2017, Angelini and De Angelis test the betting markets of 11 European leagues and find eight of them, the Turkish Süper Lig included, efficient when the best available odds are used. It is the closest study to mine, but its data ends in 2017 and it tests betting returns rather than forecast accuracy, which is the gap my project addresses.

6. **Wheatcroft, E. (2021).** Forecasting football matches by predicting match statistics. *Journal of Sports Analytics, 7*(2), 77–97. https://doi.org/10.3233/JSA-200462 (preprint: https://arxiv.org/abs/2001.09097)

   Wheatcroft predicts match statistics such as shots, shots on target and corners before kick-off from free Football-Data.co.uk data, and finds that these predictions carry information beyond the odds, with shots on target the most informative of the observed statistics. This supports using shots on target, and xG as a finer measure of chance quality, as inputs to my models.

**The gap.** Together, these studies give me the goals model (Maher; Dixon and Coles), the rating model (Hvattum and Arntzen), the accuracy measure (Constantinou and Fenton) and the inputs and benchmark (Wheatcroft; Angelini and De Angelis). No study I found compares free-data models with the odds on recent Süper Lig seasons, measured with RPS, together with a test of whether the models add information the odds miss.

## 2. Method and variables

### 2.1 Design

The study is quantitative and uses secondary, observational data: 1,693 Süper Lig matches from 2021/22 to 2025/26. It is a **within-subjects computational experiment**: every forecasting method predicts exactly the same matches, so each match acts as its own control, and methods are compared with paired tests. Models are built and tuned on 2021/22–2024/25 only; 2025/26 is a locked test season, predicted once at the end.

### 2.2 Variables in the dataset

| Variable | Columns (`data/processed/matches.csv`) | Scale | Role |
| --- | --- | --- | --- |
| Match result: home win, draw or away win | `result` | Ordinal | **Dependent variable**: what every method predicts |
| Goals scored | `home_goals`, `away_goals` | Ratio | Dependent variable of the goals model (Model B); earlier matches' goals become inputs |
| Shots, shots on target | `home_shots`, `away_shots`, `home_sot`, `away_sot` | Ratio | **Independent variable**: each team's averages over earlier matches |
| Expected goals (xG) | FootyStats match files, kept locally in `data/raw/footystats/` | Ratio | **Independent variable**: each team's averages over earlier matches |
| Elo rating difference | computed by `src/elo.py` from earlier results | Interval | **Independent variable** of Model A |
| Home team, away team | `home`, `away` | Nominal | Identifies the teams; gives the home-advantage term |
| Bet365 odds and fair probabilities | `b365_h`, `b365_d`, `b365_a` → `p_home`, `p_draw`, `p_away` | Ratio | Benchmark forecast; also an input of the combined model (Model C) |
| Bookmaker margin | `margin` | Ratio | Descriptive variable (the bookmaker's built-in cut) |
| Best market odds | `max_h`, `max_d`, `max_a` | Ratio | Robustness check of the benchmark |
| Date, season | `date`, `season` | Interval, ordinal | Order the matches and set the training and test periods (control) |

Every independent variable is built only from matches played **before** kick-off, so no forecast uses its own result.

### 2.3 Variables of the comparison

- **Independent variable: the forecasting method**, with five levels:
  1. Bookmaker: Bet365 probabilities with the margin removed
  2. Naive forecast: the home-win, draw and away-win shares of all earlier seasons
  3. Model A: Elo ratings + ordered logit
  4. Model B: Dixon–Coles goals model
  5. Model C: bookmaker probabilities + model probabilities combined
- **Second independent variable**, for the shots and xG question: the input set of Model B (goals only; goals + shots on target; goals + xG).
- **Dependent variable: RPS per match** (0 = perfect, lower is better). Secondary measures: log loss and Brier score.
- **Controlled:** the same matches, the same training and test seasons, only information available before kick-off, and the same scoring rule for every method.

### 2.4 Methods of analysis

1. **Data preparation** (`src/load_data.py`). Remove awarded results. Turn decimal odds into fair probabilities by removing the margin:

   $$p_i = \frac{1/o_i}{\sum_{j} 1/o_j}, \qquad \text{margin} = \sum_{j} \frac{1}{o_j} - 1$$

2. **Data checks and descriptive statistics** (`src/data_checks.py`, `src/descriptive.py`). Row counts, missing values, data types and consistency checks; mean, median, standard deviation and interquartile range of goals, shots, odds and margin; result shares per season; box plots and a calibration plot of the odds.
3. **Model A: Elo + ordered logit** (`src/elo.py`; Hvattum & Arntzen, 2010). Ratings are updated after every match. The update speed (K), home advantage, goal-difference weight and the share of rating kept between seasons are chosen by grid search on the training seasons. An ordered logit model turns the rating difference into probabilities.
4. **Model B: Dixon–Coles goals model** (Dixon & Coles, 1997). Attack and defence strength for every team, a home-advantage term, the low-score correction and exponential time weighting, fitted by maximum likelihood before each round. Versions estimate team strength from goals only, from goals plus shots on target, and from goals plus xG.
5. **Model C: combined model.** An ordered logit model whose inputs are the bookmaker's probabilities and the model's probabilities, fitted on earlier seasons only. If it scores a lower RPS than the bookmaker alone, my model carries information the odds miss.
6. **Evaluation.** Ranked Probability Score for every match, with r = 3 ordered outcomes:

   $$\text{RPS} = \frac{1}{r-1}\sum_{i=1}^{r-1}\Big(\sum_{j=1}^{i}(p_j - e_j)\Big)^2$$

   where p are the forecast probabilities and e is 1 for the outcome that happened, 0 otherwise. Methods are compared on per-match RPS differences with a paired t-test and a 95% bootstrap confidence interval (2,000 resamples, fixed seed 42), at the 5% significance level. Calibration plots check whether, for example, a forecast of 30% happens about 30% of the time.
7. **Validation.** Rolling origin on the training seasons: train on all earlier seasons, then predict 2022/23, 2023/24 and 2024/25 in turn. The final run on 2025/26 happens once, after all settings are fixed.

### 2.5 Known data gaps

- Bet365 closing odds are empty in the current copy of the data, so the benchmark uses the odds recorded before each round.
- 2024/25 is missing 28 matches in the current copy; downloading the original files from Football-Data.co.uk will fix both gaps.
- 29 awarded results in 2022/23 are removed (two clubs withdrew after the February 2023 earthquake).
- One match, İstanbulspor vs Trabzonspor on 19 December 2023, has no shot statistics; its 0–3 score was awarded after the match was abandoned, so it will be dropped before modelling.
