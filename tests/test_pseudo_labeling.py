import numpy as np

from fbm_multimodal.pseudo_labeling.pairwise_topk import select_pairwise_topk


def test_pairwise_topk_selects_per_pair_not_global_topk():
    probs = np.array(
        [
            [0.95, 0.94, 0.10],
            [0.90, 0.89, 0.88],
            [0.20, 0.91, 0.90],
            [0.80, 0.79, 0.99],
        ]
    )
    selected = select_pairwise_topk(
        probs,
        sample_ids=["a", "b", "c", "d"],
        class_pairs=[(0, 1), (1, 2)],
        top_k_per_pair=2,
        min_pair_score=0.5,
    )

    by_pair = selected.groupby(["class_a", "class_b"])["sample_id"].apply(list).to_dict()
    assert by_pair[(0, 1)] == ["a", "b"]
    assert by_pair[(1, 2)] == ["c", "b"]
    assert selected.groupby(["class_a", "class_b"]).size().max() == 2


def test_pairwise_topk_respects_threshold_and_exclusions():
    probs = np.array(
        [
            [0.95, 0.94],
            [0.90, 0.89],
            [0.70, 0.60],
        ]
    )
    selected = select_pairwise_topk(
        probs,
        sample_ids=["skip", "keep", "low"],
        class_pairs=[(0, 1)],
        top_k_per_pair=3,
        min_pair_score=0.8,
        exclude_sample_ids={"skip"},
    )

    assert list(selected["sample_id"]) == ["keep"]
    assert selected["rank_within_pair"].tolist() == [1]
    assert selected["pair_score"].tolist() == [0.89]
