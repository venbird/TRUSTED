"""Command-line entry point for the released TRUSTED pipeline."""

import argparse
import os
import shutil
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import wandb

from pidsmaker.config import (
    get_runtime_required_args,
    get_yml_cfg,
    set_task_to_done,
    update_task_paths_to_restart,
)
from pidsmaker.tasks import (
    batching,
    construction,
    evaluation,
    feat_inference,
    featurization,
    training,
    transformation,
    triage,
)
from pidsmaker.utils.utils import log, remove_underscore_keys, set_seed


def get_task_to_module(cfg):
    return {
        "construction": (construction, cfg.construction._task_path),
        "transformation": (transformation, cfg.transformation._task_path),
        "featurization": (featurization, cfg.featurization._task_path),
        "feat_inference": (feat_inference, cfg.feat_inference._task_path),
        "batching": (batching, cfg.batching._task_path),
        "training": (training, cfg.training._task_path),
        "evaluation": (evaluation, cfg.evaluation._task_path),
        "triage": (triage, cfg.triage._task_path),
    }


def clean_cfg_for_log(cfg):
    return remove_underscore_keys(
        dict(cfg), keys_to_keep=["_task_path", "_exp", "_tuning_file_path"]
    )


def main(cfg):
    if cfg._tuning_mode != "none" or cfg.experiment.used_method != "none":
        raise ValueError("The TRUSTED release supports standard deterministic runs only")

    set_seed(cfg)
    should_restart = update_task_paths_to_restart(cfg)
    task_results = {}
    for task, (module, task_path) in get_task_to_module(cfg).items():
        start = time.time()
        return_value = None
        if should_restart[task]:
            return_value = module.main(cfg)
            set_task_to_done(task_path)
        task_results[task] = {"time": time.time() - start, "return": return_value}

    metrics = task_results["evaluation"]["return"] or {}
    metrics["val_score"] = task_results["training"]["return"]
    times = {
        f"time_{task}": round(result["time"], 2)
        for task, result in task_results.items()
    }
    wandb.log(metrics)
    wandb.log(times)
    log("==" * 30)
    log("Run finished.")
    log("==" * 30)
    return metrics, times


if __name__ == "__main__":
    args, unknown_args = get_runtime_required_args(return_unknown_args=True)
    if unknown_args:
        raise argparse.ArgumentTypeError(f"Unknown args {unknown_args}")

    exp_name = args.exp.replace("dataset", args.dataset) if args.exp else f"{args.dataset}_{args.model}"
    tags = args.tags.split(",") if args.tags else [args.model]
    wandb.init(
        mode="online" if args.wandb else "disabled",
        project=args.project,
        name=exp_name,
        tags=tags,
    )
    cfg = get_yml_cfg(args)
    wandb.config.update(clean_cfg_for_log(cfg))
    main(cfg)
    wandb.finish()

    if cfg._restart_from_scratch:
        shutil.rmtree(cfg.construction._task_path, ignore_errors=True)
