import json

import pandas as pd

from fbm_multimodal.cli import main


def test_cli_validate_map_reports_coverage(tmp_path, capsys):
    manifest_path = tmp_path / "manifest.csv"
    map_path = tmp_path / "measurement_map.csv"
    pd.DataFrame(
        {
            "chip_id": ["c1"],
            "image_path": ["c1.png"],
            "MSR_000": [1.0],
            "MSR_001": [2.0],
            "defect_a": [1],
        }
    ).to_csv(manifest_path, index=False)
    pd.DataFrame(
        {
            "feature_name": ["MSR_000"],
            "measurement_condition": ["read"],
            "measurement_type": ["leakage"],
            "physical_region": ["top"],
        }
    ).to_csv(map_path, index=False)

    exit_code = main(["validate-map", "--manifest", str(manifest_path), "--measurement-map", str(map_path)])

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["coverage_ratio"] == 0.5
    assert output["missing_features"] == ["MSR_001"]


def test_cli_rank_unlabeled_writes_ranked_csv(tmp_path):
    candidates_path = tmp_path / "candidates.csv"
    output_path = tmp_path / "ranked.csv"
    pd.DataFrame(
        {
            "chip_id": ["a", "b"],
            "prob_defect": [0.99, 0.51],
            "image_prob_defect": [0.99, 0.8],
            "tabular_prob_defect": [0.98, 0.1],
        }
    ).to_csv(candidates_path, index=False)

    exit_code = main(
        [
            "rank-unlabeled",
            "--candidates",
            str(candidates_path),
            "--labels",
            "defect",
            "--target-labels",
            "defect",
            "--budget",
            "1",
            "--output",
            str(output_path),
        ]
    )

    ranked = pd.read_csv(output_path)
    assert exit_code == 0
    assert ranked.loc[0, "chip_id"] == "b"
    assert ranked.loc[0, "selection_reason"] == "image_tabular_disagreement"
