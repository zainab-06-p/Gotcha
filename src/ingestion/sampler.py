"""
Gotcha — Stratified Sampling Module
Produces a representative subset of candidates, proportional by a stratification key.
"""

import logging
import random
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)


def stratified_sample(
    candidates: list[dict],
    n: int,
    stratify_by: str = "current_title",
    seed: int = 42,
) -> list[dict]:
    """Sample candidates proportionally by a stratification field.

    For example, if 20% of candidates are "Backend Engineer" and n=100,
    ~20 Backend Engineers will appear in the sample.

    Args:
        candidates: Full list of candidate dicts.
        n: Desired sample size.
        stratify_by: Field name inside candidate["profile"] to stratify on.
        seed: Random seed for reproducibility.

    Returns:
        List of n sampled candidates (or fewer if total < n).
    """
    if not candidates:
        logger.warning("No candidates to sample from")
        return []

    if n >= len(candidates):
        logger.info("Requested %d but only %d candidates available, returning all",
                     n, len(candidates))
        return list(candidates)

    rng = random.Random(seed)

    # Group candidates by stratification key
    strata: dict[str, list[dict]] = defaultdict(list)
    for cand in candidates:
        profile = cand.get("profile", {})
        key = profile.get(stratify_by, "unknown")
        if key is None:
            key = "unknown"
        strata[str(key).lower().strip()] = strata.get(str(key).lower().strip(), [])
        strata[str(key).lower().strip()].append(cand)

    # Compute proportional allocation
    total = len(candidates)
    sampled: list[dict] = []
    remainder_pool: list[tuple[str, list[dict]]] = []

    for key, group in strata.items():
        proportion = len(group) / total
        count = int(proportion * n)
        if count > len(group):
            count = len(group)
        chosen = rng.sample(group, count)
        sampled.extend(chosen)
        # Track remaining candidates in this stratum for filling gaps
        leftover = [c for c in group if c not in chosen]
        if leftover:
            remainder_pool.append((key, leftover))

    # Fill remaining slots due to rounding
    deficit = n - len(sampled)
    if deficit > 0:
        flat_remainder = []
        for _, leftover in remainder_pool:
            flat_remainder.extend(leftover)
        rng.shuffle(flat_remainder)
        sampled.extend(flat_remainder[:deficit])

    logger.info("Stratified sample: %d candidates across %d strata",
                len(sampled), len(strata))
    return sampled[:n]
