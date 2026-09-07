#!/usr/bin/env python3
"""Initial values for the learner's Q-table.

Q-learning from an all-zero table does not work on this problem: a good policy
is a specific multi-bundle sequence, the only strong feedback (right/wrong)
arrives at the very end of the episode, and epsilon-greedy essentially never
produces that sequence by chance. Cold-start runs plateau after a single bundle
at 0.597 accuracy. Seeding the table fixes that -- training then refines a
sensible starting point rather than trying to discover one.

one_step_seed returns {(mask, bin): {action: q}} for QLearner.warm_start(),
which only loads values and contains no logic for producing them -- so any
other source of initial values drops in here just as well.

It reads the subset scores that bundle_sweep.py already recorded and asks, for
each state, "what would I get if I acquired this one bundle and then stopped?".
No recursion and no planning: it prices ONE bundle at a time and says nothing
about sequences, combinations, or when to stop, which is the part the agent
still has to learn.
"""

from typing import Dict

import sys
from pathlib import Path
_AGENT = Path(__file__).resolve().parents[1]          # CODE/training/agent
sys.path[:0] = [str(_AGENT / "core"), str(_AGENT.parent / "train")]
from train_classifier import N_BUNDLES


def one_step_seed(scores: Dict[int, float],
                  bundle_costs: Dict[int, float],
                  n_conf_bins: int = 1,
                  terminal_weight: float = 1.0) -> Dict[tuple, Dict[int, float]]:
    """Seed from measured subset scores, looking one acquisition ahead.

        Q[stop]        = beta * score(m)
        Q[acquire i]   = beta * score(m | 1<<i)  -  C_i

    i.e. "stopping here is worth what this classifier scores; acquiring bundle
    i is worth what THAT classifier scores, minus what i costs to get."

    Args:
        scores: {mask: score}. Use a VALIDATION metric -- seeding from a test
            column would leak the test set into the agent's starting point.
        bundle_costs: {bundle index: C_i} = lambda * seconds, already scaled.
        n_conf_bins: confidence bins in the learner's state.
        terminal_weight: beta.

    The scores are not outside knowledge smuggled in: the 1,024 classifiers
    ARE the environment the agent queries, so their measured accuracies are a
    property of that environment rather than a solution handed to the agent.
    """
    seed: Dict[tuple, Dict[int, float]] = {}
    for m in range(1 << N_BUNDLES):
        row = {0: terminal_weight * scores[m]}
        for i in range(N_BUNDLES):
            if not m >> i & 1:
                row[i + 1] = (terminal_weight * scores[m | (1 << i)]
                              - bundle_costs.get(i, 0.0))
        for b in range(max(n_conf_bins, 1)):
            seed[(m, b)] = dict(row)
    return seed
