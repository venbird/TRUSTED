from pidsmaker.config import update_cfg_for_multi_dataset
from pidsmaker.utils.utils import copy_directory, get_multi_datasets, log_start, set_seed


def main_from_config(cfg):
    methods = [value.strip() for value in cfg.transformation.used_methods.split(",")]
    if methods != ["none"]:
        raise ValueError(f"TRUSTED only supports transformation.used_methods=none, got {methods}")
    copy_directory(cfg.construction._graphs_dir, cfg.transformation._graphs_dir)


def main(cfg):
    set_seed(cfg)
    log_start(__file__)
    datasets = get_multi_datasets(cfg)
    if "none" in datasets:
        main_from_config(cfg)
        return
    for dataset in datasets:
        updated_cfg, should_restart = update_cfg_for_multi_dataset(cfg, dataset)
        if should_restart["transformation"]:
            main_from_config(updated_cfg)
