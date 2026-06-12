# ─────────────────────────────────────────────────────────────
#   engine.py — Dual Recommendation Engine (CNN + Text)
#   Fashion Recommendation A/B Evaluation
# ─────────────────────────────────────────────────────────────
"""
Provides two independent recommenders sharing the same item catalog:
  - CNNRecommender  : VGG19 Exp3 feature vectors (512-D)
  - TextRecommender : One-Hot Filtered attribute vectors (1158-D)

Both use:
  - K-Means for cold-start item selection (on CNN features)
  - Mean + L2-normalize user profile
  - Cosine similarity for retrieval
"""

import os
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from typing import List, Tuple, Optional

from config import (
    MASTER_CSV, CNN_FEATURE_PATH, TEXT_FEATURE_PATH,
    COL_PATH, COL_ITEM_ID, COL_CATEGORY,
    COLD_START_K, REC_TOP_N, RANDOM_STATE,
)


class _BaseRecommender:
    """Shared logic for cosine-similarity based recommenders."""

    def __init__(self, feature_matrix: np.ndarray, df: pd.DataFrame,
                 top_n: int = REC_TOP_N):
        self.features = feature_matrix          # (N, D)
        self.df       = df                      # (N, ...)
        self.top_n    = top_n
        self._profile : Optional[np.ndarray] = None
        self._shown   : set = set()

    # ── User Profile ──────────────────────────────────────────
    def build_profile(self, liked_indices: List[int]) -> None:
        """Build user profile from liked item indices."""
        vecs = self.features[liked_indices]           # (K, D)
        mean = np.mean(vecs, axis=0)                  # (D,)
        norm = np.linalg.norm(mean)
        self._profile = mean / norm if norm > 0 else mean

    # ── Recommend ─────────────────────────────────────────────
    def recommend(self, liked_indices: List[int]) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Return top-N recommendations excluding liked + previously shown items.
        Returns (DataFrame with item info, similarity scores array).
        """
        if self._profile is None:
            raise RuntimeError("Build profile first via build_profile().")

        scores = self.features @ self._profile         # (N,) cosine sim

        # Exclude liked + shown
        exclude = set(liked_indices) | self._shown
        if exclude:
            exclude_arr = np.array(list(exclude), dtype=int)
            scores[exclude_arr] = -2.0

        ranked = np.argsort(scores)[::-1][:self.top_n]
        top_scores = scores[ranked]

        # Track shown items
        self._shown.update(ranked.tolist())

        recs = self.df.iloc[ranked].copy()
        recs['similarity_score'] = top_scores
        return recs, top_scores

    # ── Reset ─────────────────────────────────────────────────
    def reset(self) -> None:
        self._profile = None
        self._shown   = set()


class DualRecommenderSystem:
    """
    Orchestrates both CNN and Text recommenders.
    Provides cold-start selection, profile building, and A/B recommendation.
    """

    def __init__(self):
        # Load data once
        self._df = pd.read_csv(MASTER_CSV)
        self._cnn_feat  = np.load(CNN_FEATURE_PATH)    # (7975, 512)
        self._text_feat = np.load(TEXT_FEATURE_PATH)   # (7975, 1158)

        # L2-normalize both matrices for cosine similarity via dot product
        self._cnn_feat  = self._l2_normalize(self._cnn_feat)
        self._text_feat = self._l2_normalize(self._text_feat)

        # Initialize recommenders
        self.cnn  = _BaseRecommender(self._cnn_feat,  self._df, top_n=REC_TOP_N)
        self.text = _BaseRecommender(self._text_feat, self._df, top_n=REC_TOP_N)

        # Stratified K-Means: ensure gender diversity in cold start
        # Dataset is ~88% women / ~12% men, so allocate clusters proportionally
        gender_col = 'gender'
        self._women_mask = (self._df[gender_col] == 'WOMEN').values
        self._men_mask   = (self._df[gender_col] == 'MEN').values

        # Allocate: 5 clusters for women, 3 for men (total = COLD_START_K)
        self._women_clusters = 5
        self._men_clusters   = COLD_START_K - self._women_clusters  # 3

        self._women_km = MiniBatchKMeans(
            n_clusters=self._women_clusters, random_state=RANDOM_STATE, n_init=3)
        self._men_km = MiniBatchKMeans(
            n_clusters=self._men_clusters, random_state=RANDOM_STATE, n_init=3)

        w_idx = np.where(self._women_mask)[0]
        m_idx = np.where(self._men_mask)[0]
        self._women_labels = self._women_km.fit_predict(self._cnn_feat[w_idx])
        self._men_labels   = self._men_km.fit_predict(self._cnn_feat[m_idx])
        self._women_indices = w_idx
        self._men_indices   = m_idx

    @staticmethod
    def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
        """Row-wise L2 normalization."""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)   # avoid division by zero
        return matrix / norms

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    # ── Cold Start ────────────────────────────────────────────
    def get_cold_start_items(self) -> pd.DataFrame:
        """
        Stratified cold start: pick items from K-Means clusters within each gender.
        Returns 5 women's items + 3 men's items = COLD_START_K diverse items.
        """
        rng = np.random.default_rng()
        indices = []

        # Pick from women's clusters
        for cluster_id in range(self._women_clusters):
            cluster_members = np.where(self._women_labels == cluster_id)[0]
            if len(cluster_members) == 0:
                continue
            # Map local cluster index back to global dataset index
            local_chosen = rng.choice(cluster_members)
            global_idx = int(self._women_indices[local_chosen])
            indices.append(global_idx)

        # Pick from men's clusters
        for cluster_id in range(self._men_clusters):
            cluster_members = np.where(self._men_labels == cluster_id)[0]
            if len(cluster_members) == 0:
                continue
            local_chosen = rng.choice(cluster_members)
            global_idx = int(self._men_indices[local_chosen])
            indices.append(global_idx)

        # Mark cold start items as "shown" in both engines
        self.cnn._shown.update(indices)
        self.text._shown.update(indices)

        return self._df.iloc[indices].copy()

    # ── Build Profiles (both engines from same liked items) ──
    def build_profiles(self, liked_indices: List[int]) -> None:
        """Build user profiles for both CNN and Text from the same selections."""
        self.cnn.build_profile(liked_indices)
        self.text.build_profile(liked_indices)

    # ── Get Recommendations (both engines) ────────────────────
    def get_recommendations(self, liked_indices: List[int]
                            ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Return (cnn_recs, text_recs) — both DataFrames with top-N items.
        """
        cnn_recs, _  = self.cnn.recommend(liked_indices)
        text_recs, _ = self.text.recommend(liked_indices)
        return cnn_recs, text_recs

    # ── Reset Session ─────────────────────────────────────────
    def reset(self) -> None:
        self.cnn.reset()
        self.text.reset()
