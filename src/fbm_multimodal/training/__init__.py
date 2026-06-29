"""Training utilities for offline experiment artifacts."""

__all__ = ["CatBoostOOFResult", "train_catboost_oof_logits"]


def __getattr__(name: str):
    if name in __all__:
        from fbm_multimodal.training.train_catboost_oof import CatBoostOOFResult, train_catboost_oof_logits

        return {
            "CatBoostOOFResult": CatBoostOOFResult,
            "train_catboost_oof_logits": train_catboost_oof_logits,
        }[name]
    raise AttributeError(name)
