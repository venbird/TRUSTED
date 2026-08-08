from pidsmaker.featurization.trusted_utils import build_indexid2vec, ensure_feature_state
from pidsmaker.utils.utils import log_start


def main(cfg):
    log_start(__file__)
    ensure_feature_state(cfg)
    return build_indexid2vec(cfg)
