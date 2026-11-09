"""Rank-mean ensembling of several models' predictions.

This is how the ONE-NAS paper reads an evolved population. Prior work
takes the single global-best genome; the paper instead averages the
island champions, and on identical runs that difference is worth +4.5%
against +27.5% net return over 2022-2024. The combination happens at
scoring time and does not touch the search, which is what makes the two
readings comparable run-for-run.

Ranks are averaged rather than the predicted values themselves. Members
are on arbitrary and mutually incomparable scales -- one champion may
emit values ten times another's -- and averaging levels would let the
loudest member dominate. Every trading rule here reads only the ordering
of a day's cross-section, so nothing downstream needs the levels.

    from ensemble import rank_mean, combine_prediction_files
"""
import os

import numpy as np
import pandas as pd


def ranks(values):
    """Ascending ranks, 0-based, ties given their average rank."""
    return pd.Series(values).rank(method="average").to_numpy() - 1.0


def rank_mean(cross_sections):
    """Average the centred ranks of several members' predictions for one day.

    `cross_sections` is a sequence of equal-length arrays, one per member.
    Centring on (n-1)/2 makes each member's vote mean-zero, so the result
    is a signed score suitable for rules that key on sign as well as order.
    """
    members = [np.asarray(c, dtype=float) for c in cross_sections]
    if not members:
        raise ValueError("no members to combine")
    n = len(members[0])
    if any(len(c) != n for c in members):
        raise ValueError("members disagree on cross-section length")
    mid = (n - 1) / 2.0
    return np.mean([ranks(c) - mid for c in members], axis=0)


def mean_pairwise_spearman(cross_sections):
    """Mean Spearman correlation over all member pairs for one day.

    This is the rho of the equicorrelated averaging prediction
    m*sqrt(N/(1+(N-1)rho)), which is what tells you whether an ensemble's
    members are diverse enough for averaging to buy anything. Computed via
    the norm of the summed unit vectors, which is O(m*n) rather than the
    O(m^2*n) of forming every pair.
    """
    members = [ranks(c) for c in cross_sections]
    m = len(members)
    if m < 2:
        return float("nan")
    unit = []
    for r in members:
        centred = r - r.mean()
        norm = np.linalg.norm(centred)
        if norm == 0:
            continue                      # a flat member correlates with nothing
        unit.append(centred / norm)
    if len(unit) < 2:
        return float("nan")
    total = np.sum(unit, axis=0)
    m = len(unit)
    return float((np.dot(total, total) - m) / (m * (m - 1)))


def combine_prediction_files(paths, column="predicted_RET", out_path=None):
    """Rank-mean several prediction CSVs into one, row by row.

    Each file is one member and must carry `column` with one row per
    trading day, in the same order. Returns the combined DataFrame and
    writes it to `out_path` if given.
    """
    if not paths:
        raise ValueError("no prediction files given")
    frames = []
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        frames.append(pd.read_csv(p, usecols=[column])[column].to_numpy())
    lengths = {len(f) for f in frames}
    if len(lengths) != 1:
        raise ValueError(f"prediction files differ in length: {sorted(lengths)}")

    # one cross-section per member per day: here each file is a single
    # name's series, so the members are combined position by position
    combined = rank_mean(frames)
    out = pd.DataFrame({column: combined})
    if out_path:
        out.to_csv(out_path, index=False)
    return out
