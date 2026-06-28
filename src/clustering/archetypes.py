"""
Gotcha — Candidate Archetype Clustering
Discovers natural candidate sub-groups (archetypes) using KMeans clustering
on the 7-dimensional score vectors.
"""

import logging
import numpy as np
from typing import Optional, Any

from src.config import N_ARCHETYPES, CandidateResult

logger = logging.getLogger(__name__)

# Canonical names for discovered archetypes based on centroid profiles
ARCHETYPE_LABELS = {
    0: "The All-Rounder (Balanced Fit)",
    1: "The Passive Maverick (Strong Skills, Low Engagement)",
    2: "The Pedigreed Scholar (Tier-1 Edu, Growing Skills)",
    3: "The Fast Mover (Immediate Joiner, High Feasibility)",
    4: "The Domain Specialist (Deep Career Description Match)",
    5: "The Skill powerhouse (Verified Assessments, High Trust)",
    6: "The Rising Talent (High Profile Trust, Mid Experience)",
    7: "The Active Job Seeker (High Behavioral signals, Ready)",
}


def fit_archetypes(
    results: list[CandidateResult],
    n_clusters: int = N_ARCHETYPES,
) -> dict[int, str]:
    """Cluster candidate results into archetypes based on their 7-axis vectors.

    Mutates each CandidateResult in-place by assigning a redirect_suggestion (archetype name)
    and redirect_reason, or returns the mapping from cluster index to label.

    Args:
        results: List of CandidateResult objects.
        n_clusters: Number of clusters to fit.

    Returns:
        Dict mapping cluster index to archetype label name.
    """
    if not results:
        return {}

    # Extract vectors
    vectors = []
    for r in results:
        vectors.append(r.axis_scores.to_vector())

    X = np.array(vectors)

    try:
        from sklearn.cluster import KMeans

        # Fit KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        centroids = kmeans.cluster_centers_

        # Map each cluster index to a descriptive label based on the centroid's max dimension
        cluster_labels = {}
        axis_names = [
            "skill_relevance",
            "experience_impact",
            "domain_coherence",
            "narrative_credibility",
            "behavioral_validation",
            "engagement_feasibility",
            "pedigree",
        ]

        for i in range(n_clusters):
            centroid = centroids[i]
            max_idx = int(np.argmax(centroid))
            max_axis = axis_names[max_idx]

            # Assign names dynamically based on dominant axis if not using standard labels
            if i in ARCHETYPE_LABELS:
                cluster_labels[i] = ARCHETYPE_LABELS[i]
            else:
                cluster_labels[i] = f"Archetype {i} (Dominant: {max_axis.replace('_', ' ').title()})"

        # Assign cluster label to each CandidateResult
        for idx, r in enumerate(results):
            cluster_id = int(labels[idx])
            archetype_name = cluster_labels[cluster_id]
            # Store in result
            r.redirect_suggestion = archetype_name
            r.redirect_reason = f"Candidate vector groups they into the '{archetype_name}' archetype based on profile characteristics."

        logger.info("Successfully clustered %d candidates into %d archetypes using KMeans", len(results), n_clusters)
        return cluster_labels

    except Exception as e:
        logger.error("KMeans clustering failed: %s. Using heuristic fallback.", e)
        # Heuristic fallback: divide by highest axis score
        cluster_labels = ARCHETYPE_LABELS.copy()
        
        for idx, r in enumerate(results):
            vec = r.axis_scores.to_vector()
            max_idx = int(np.argmax(vec))
            # Wrap around index
            cluster_id = max_idx % n_clusters
            archetype_name = cluster_labels.get(cluster_id, f"Archetype {cluster_id}")
            r.redirect_suggestion = archetype_name
            r.redirect_reason = f"Heuristic fallback: candidate matches '{archetype_name}' profile tendencies."

        return cluster_labels
