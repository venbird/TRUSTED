from pidsmaker.featurization.featurization_methods import featurization_trusted
from pidsmaker.utils.utils import set_seed


def main(cfg):
    set_seed(cfg, seed=cfg.featurization.seed)
    method = cfg.featurization.used_method.strip()
    if method != "trusted":
        raise ValueError(f"Invalid node embedding method {method}")
    featurization_trusted.main(cfg)
