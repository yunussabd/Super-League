"""
Shared functions used by the scripts in src/ and by the tests.

    implied_probabilities   bookmaker decimal odds -> fair probabilities and margin
    rps                     Ranked Probability Score, the project's main accuracy measure
"""
import numpy as np

OUTCOMES = ("H", "D", "A")  # ordered: home win, draw, away win


def implied_probabilities(odds_home, odds_draw, odds_away):
    """Turn decimal odds into fair probabilities by removing the bookmaker's margin.

    1 / odds is the implied probability of each outcome. The three add up to more
    than 1; the excess is the bookmaker's margin. Dividing each by their sum removes
    it (basic normalisation).

    Takes single numbers or equal-length columns and returns four arrays:
    p_home, p_draw, p_away, margin.
    Example: odds 1.80 / 3.60 / 4.50 -> 0.526 / 0.263 / 0.211, margin 0.056.
    """
    odds = np.column_stack([np.atleast_1d(np.asarray(o, dtype=float))
                            for o in (odds_home, odds_draw, odds_away)])
    if np.any(odds <= 1):
        raise ValueError("decimal odds must be greater than 1")
    inverse = 1.0 / odds
    total = inverse.sum(axis=1)
    probs = inverse / total[:, None]
    return probs[:, 0], probs[:, 1], probs[:, 2], total - 1.0


def rps(probs, results):
    """Ranked Probability Score for each match: 0 = perfect, 1 = worst, lower is better.

    probs   : n x 3 probabilities in the order home win, draw, away win
    results : n results, each "H", "D" or "A"

    The outcomes are ordered, so RPS compares cumulative probabilities: predicting a
    draw when the home team wins is less wrong than predicting an away win
    (Constantinou & Fenton, 2012).

        RPS = 1/(r-1) * sum over i=1..r-1 of ( sum over j<=i of (p_j - e_j) )^2,  r = 3
    """
    probs = np.atleast_2d(np.asarray(probs, dtype=float))
    results = [str(r) for r in np.atleast_1d(np.asarray(results, dtype=object))]
    if probs.shape[1] != len(OUTCOMES) or len(results) != len(probs):
        raise ValueError("need one row of three probabilities per result")
    unknown = set(results) - set(OUTCOMES)
    if unknown:
        raise ValueError(f"results must be H, D or A, got {sorted(unknown)}")
    actual = np.zeros_like(probs)
    actual[np.arange(len(results)), [OUTCOMES.index(r) for r in results]] = 1.0
    cumulative = np.cumsum(probs - actual, axis=1)[:, :-1]
    return (cumulative ** 2).sum(axis=1) / (len(OUTCOMES) - 1)
