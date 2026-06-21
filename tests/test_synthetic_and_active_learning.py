import numpy as np
import pandas as pd

from fbm_multimodal.active_learning import rank_unlabeled_for_review
from fbm_multimodal.synthetic import compose_fbm_images, cosine_similarity


def test_compose_fbm_images_supports_max_clipped_and_saturating_modes():
    first = np.array([[0, 4, 8]], dtype=np.float32)
    second = np.array([[8, 4, 1]], dtype=np.float32)

    np.testing.assert_array_equal(compose_fbm_images(first, second, mode="max"), [[8, 4, 8]])
    np.testing.assert_array_equal(compose_fbm_images(first, second, mode="clipped_sum"), [[8, 8, 8]])
    np.testing.assert_allclose(
        compose_fbm_images(first, second, mode="weighted_saturating_sum", alpha=0.5),
        [[4.0, 6.0, 8.0]],
    )


def test_cosine_similarity_scores_real_synthetic_candidate_similarity():
    assert cosine_similarity(np.array([1, 0, 0]), np.array([1, 0, 0])) == 1.0
    assert cosine_similarity(np.array([1, 0, 0]), np.array([0, 1, 0])) == 0.0


def test_active_learning_ranking_combines_confidence_uncertainty_and_disagreement():
    candidates = pd.DataFrame(
        {
            "chip_id": ["high-conf", "disagree", "uncertain", "ordinary"],
            "embedding_x": [0.0, 1.0, 2.0, 3.0],
            "prob_a": [0.99, 0.8, 0.52, 0.2],
            "prob_b": [0.01, 0.7, 0.48, 0.1],
            "image_prob_a": [0.98, 0.95, 0.52, 0.2],
            "tabular_prob_a": [0.97, 0.10, 0.51, 0.3],
        }
    )

    ranked = rank_unlabeled_for_review(
        candidates,
        label_columns=["a", "b"],
        target_labels=["a"],
        budget=3,
        embedding_columns=["embedding_x"],
    )

    assert list(ranked["chip_id"]) == ["disagree", "high-conf", "uncertain"]
    assert ranked.loc[ranked["chip_id"] == "disagree", "selection_reason"].iloc[0] == "image_tabular_disagreement"
    assert ranked.loc[ranked["chip_id"] == "high-conf", "selection_reason"].iloc[0] == "high_confidence_target"
