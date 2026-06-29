import numpy as np
import pandas as pd

from fbm_multimodal.synthetic_wl_map import compose_synthetic_wl_map
from fbm_multimodal.wl_residual_map import DEFAULT_WL_CHANNELS, WLResidualMapTensorizer, parse_wordline


def test_parse_wordline_accepts_common_formats():
    assert parse_wordline("WL000") == 0
    assert parse_wordline("WL001") == 1
    assert parse_wordline("WL200") == 200
    assert parse_wordline(17) == 17


def test_residual_map_uses_train_real_baseline_only_and_clips_high_side():
    train = pd.DataFrame(
        {
            "sample_id": ["tr1", "tr2", "tr3", "tr4", "val1", "syn1"],
            "split": ["train", "train", "train", "train", "valid", "train"],
            "eval_group": [
                "real_single",
                "real_single",
                "real_single",
                "real_single",
                "real_single",
                "synthetic_composite",
            ],
            "test_method": ["read"] * 6,
            "wordline": ["WL000"] * 6,
            "value": [10.0, 12.0, 14.0, 16.0, 1000.0, 1000.0],
        }
    )
    tensorizer = WLResidualMapTensorizer(num_wl_bins=20, clip_max=2.0)
    tensorizer.fit(train)

    transformed = tensorizer.transform(
        pd.DataFrame(
            {
                "sample_id": ["below", "above", "huge"],
                "test_method": ["read", "read", "read"],
                "wordline": ["WL000", "WL000", "WL000"],
                "value": [9.0, 16.0, 1000.0],
            }
        )
    )

    ch = tensorizer.channel_index
    assert tensorizer.fit_sample_ids_ == {"tr1", "tr2", "tr3", "tr4"}
    assert transformed["below"][ch["mean_residual"], 0, 0] == 0.0
    np.testing.assert_allclose(transformed["above"][ch["mean_residual"], 0, 0], 1.0, atol=1e-6)
    assert transformed["huge"][ch["mean_residual"], 0, 0] == 2.0


def test_residual_map_outputs_expected_channels_and_count_ratio(tmp_path):
    measurements = pd.DataFrame(
        {
            "sample_id": ["s1", "s1", "s2", "s2"],
            "split": ["train"] * 4,
            "eval_group": ["real_single"] * 4,
            "test_method": ["read", "read", "program", "program"],
            "wordline": [0, 1, 200, 200],
            "value": [1.0, 3.0, 5.0, 7.0],
        }
    )
    tensorizer = WLResidualMapTensorizer(num_wl_bins=20).fit(measurements)
    maps = tensorizer.transform(measurements)

    assert maps["s1"].shape == (len(DEFAULT_WL_CHANNELS), 20, 2)
    ch = tensorizer.channel_index
    assert maps["s1"][ch["observed_mask"], 0, tensorizer.test_method_index["read"]] == 1.0
    assert maps["s1"][ch["source_count_norm"], 0, tensorizer.test_method_index["read"]] == 1.0
    assert 0.0 <= maps["s1"][ch["count_ratio"], 0, tensorizer.test_method_index["read"]] <= 1.0
    assert maps["s1"][ch["observed_mask"], 19, tensorizer.test_method_index["program"]] == 0.0

    normalizer_path = tmp_path / "wl_tensorizer.json"
    cache_path = tmp_path / "wl_cache.npz"
    tensorizer.save(normalizer_path)
    loaded_tensorizer = WLResidualMapTensorizer.load(normalizer_path)
    loaded_tensorizer.save_tensor_cache(maps, cache_path)
    loaded_maps = WLResidualMapTensorizer.load_tensor_cache(cache_path)
    np.testing.assert_allclose(loaded_maps["s1"], maps["s1"])
    assert loaded_tensorizer.test_methods_ == tensorizer.test_methods_


def test_synthetic_wl_map_uses_union_mask_source_count_and_max_values():
    channels = DEFAULT_WL_CHANNELS
    ch = {name: i for i, name in enumerate(channels)}
    a = np.zeros((len(channels), 2, 1), dtype=float)
    b = np.zeros_like(a)
    a[ch["mean_residual"], 0, 0] = 1.5
    a[ch["max_residual"], 0, 0] = 2.0
    a[ch["observed_mask"], 0, 0] = 1.0
    a[ch["count_ratio"], 0, 0] = 0.4
    b[ch["mean_residual"], 0, 0] = 2.5
    b[ch["max_residual"], 0, 0] = 1.0
    b[ch["observed_mask"], 0, 0] = 1.0
    b[ch["count_ratio"], 0, 0] = 0.7
    b[ch["mean_residual"], 1, 0] = 3.0
    b[ch["observed_mask"], 1, 0] = 1.0

    synthetic = compose_synthetic_wl_map(a, b, channel_spec=channels)

    assert synthetic[ch["mean_residual"], 0, 0] == 2.5
    assert synthetic[ch["max_residual"], 0, 0] == 2.0
    assert synthetic[ch["observed_mask"], 0, 0] == 1.0
    assert synthetic[ch["source_count_norm"], 0, 0] == 1.0
    assert synthetic[ch["source_count_norm"], 1, 0] == 0.5
    assert synthetic[ch["observed_mask"], 1, 0] == 1.0
