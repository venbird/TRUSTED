import os

from pidsmaker.featurization.trusted_utils import build_feature_state, save_feature_state
from pidsmaker.utils.utils import log, log_start


def main(cfg):
    log_start(__file__)
    model_save_path = cfg.featurization._model_dir
    os.makedirs(model_save_path, exist_ok=True)

    log("Building TRUSTED train-only feature profiles...")
    semantic_model, structural_scaler, trust_profile, all_stats, manifest = build_feature_state(cfg)
    save_feature_state(cfg, semantic_model, structural_scaler, trust_profile, all_stats, manifest)
    log(
        "TRUSTED feature dimensions: "
        f"semantic={manifest['semantic_dim']}, "
        f"structural={manifest['structural_dim']}, "
        f"trust={manifest['trust_dim']}, "
        f"total={manifest['total_dim']}"
    )
