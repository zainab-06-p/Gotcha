"""
Gotcha — Behavioral + Availability Scoring Module (Redrob JD Edition)

Scores candidate market engagement and hiring feasibility using redrob_signals.

SENTINEL RULE (enforced here as a safety net even though normalizer already maps -1→None):
  github_activity_score == -1  →  treat as 0 bonus, NOT as penalty
  offer_acceptance_rate == -1  →  treat as neutral
  interview_completion_rate == -1 → treat as neutral
  Any None value → skip entirely (do not penalize missing data)

Signals scored:
  last_active_date         — recency of platform activity (highest weight)
  open_to_work_flag        — actively looking
  notice_period_days       — JD prefers sub-30 days
  recruiter_response_rate  — responsiveness signal
  github_activity_score    — only if linked (not -1/None)
  saved_by_recruiters_30d  — market demand signal
  search_appearance_30d    — discoverability signal
  interview_completion_rate — reliability signal
"""

import logging
from datetime import date, datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Reference date for recency calculations
_EVAL_DATE = date(2026, 6, 28)


def _safe_float(val, sentinel: float = -1.0) -> Optional[float]:
    """Parse a numeric value, returning None if it is missing, None, or the sentinel."""
    if val is None:
        return None
    try:
        f = float(val)
        if f == sentinel:
            return None
        return f
    except (TypeError, ValueError):
        return None


def _days_inactive(last_active_str: Optional[str]) -> Optional[int]:
    """Return days since last_active_date, or None if unparseable."""
    if not last_active_str:
        return None
    try:
        # Accept YYYY-MM-DD or ISO-8601 with time component
        last_active = datetime.fromisoformat(str(last_active_str).replace("Z", "")).date()
        return max(0, (_EVAL_DATE - last_active).days)
    except (ValueError, TypeError):
        return None


def score_behavioral(redrob_signals: Optional[dict]) -> float:
    """Score behavioral engagement and availability for the Senior AI Engineer role.

    Returns float in [0, 1]. Missing signals are skipped (sentinel-safe).
    """
    if not redrob_signals or not isinstance(redrob_signals, dict):
        return 0.40  # neutral baseline when no signals present

    score = 0.0
    weight_sum = 0.0

    # ── 1. Last active date recency (weight 0.30 — highest signal of intent) ─────
    days = _days_inactive(redrob_signals.get("last_active_date"))
    if days is not None:
        if days <= 7:
            recency = 1.0
        elif days <= 14:
            recency = 0.90
        elif days <= 30:
            recency = 0.80
        elif days <= 60:
            recency = 0.65
        elif days <= 90:
            recency = 0.50
        elif days <= 180:
            recency = 0.30
        else:
            recency = 0.05   # gone cold
        score += 0.30 * recency
        weight_sum += 0.30

    # ── 2. Open to work flag (weight 0.20) ────────────────────────────────────────
    otw = redrob_signals.get("open_to_work_flag")
    if otw is not None:
        score += 0.20 * (1.0 if otw else 0.2)
        weight_sum += 0.20

    # ── 3. Notice period (weight 0.15 — JD prefers sub-30 days) ──────────────────
    notice = _safe_float(redrob_signals.get("notice_period_days"))
    if notice is not None:
        if notice <= 0:
            np_score = 1.0   # immediate joiner
        elif notice <= 30:
            np_score = 1.0
        elif notice <= 60:
            np_score = 0.70
        elif notice <= 90:
            np_score = 0.40
        else:
            np_score = 0.10  # 90+ days
        score += 0.15 * np_score
        weight_sum += 0.15

    # ── 4. Recruiter response rate (weight 0.15) ──────────────────────────────────
    rrr = _safe_float(redrob_signals.get("recruiter_response_rate"))
    if rrr is not None:
        rrr = min(max(rrr, 0.0), 1.0)
        if rrr >= 0.70:
            rrr_score = 1.0
        elif rrr >= 0.40:
            rrr_score = 0.70
        elif rrr >= 0.20:
            rrr_score = 0.40
        else:
            rrr_score = 0.10
        score += 0.15 * rrr_score
        weight_sum += 0.15

    # ── 5. GitHub activity score (weight 0.10 — only if linked) ──────────────────
    github = _safe_float(redrob_signals.get("github_activity_score"))
    if github is not None:   # None = not linked = skip entirely
        github = max(github, 0.0)
        if github >= 70:
            gh_score = 1.0
        elif github >= 50:
            gh_score = 0.75
        elif github >= 30:
            gh_score = 0.50
        elif github >= 10:
            gh_score = 0.25
        else:
            gh_score = 0.10
        score += 0.10 * gh_score
        weight_sum += 0.10

    # ── 6. Saved by recruiters 30d (weight 0.05) ─────────────────────────────────
    saved = _safe_float(redrob_signals.get("saved_by_recruiters_30d"))
    if saved is not None:
        saved_score = min(max(saved, 0.0) / 20.0, 1.0)  # cap at 20 saves
        score += 0.05 * saved_score
        weight_sum += 0.05

    # ── 7. Interview completion rate (weight 0.05) ────────────────────────────────
    icr = _safe_float(redrob_signals.get("interview_completion_rate"))
    if icr is not None:
        icr = min(max(icr, 0.0), 1.0)
        icr_score = 1.0 if icr >= 0.80 else (0.65 if icr >= 0.50 else 0.20)
        score += 0.05 * icr_score
        weight_sum += 0.05

    # ── Normalise by available weight ─────────────────────────────────────────────
    if weight_sum == 0.0:
        return 0.40  # no signals → neutral

    final = score / weight_sum
    final = max(0.0, min(final, 1.0))
    logger.debug("Behavioral score: %.3f (weight_sum=%.2f)", final, weight_sum)
    return final
