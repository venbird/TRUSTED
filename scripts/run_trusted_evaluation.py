import argparse
import json
import numbers
import os
from datetime import datetime, timezone

import numpy as np

from pidsmaker.config import get_runtime_required_args, get_yml_cfg
from pidsmaker.detection.evaluation_methods import trusted_risk_evaluation
from pidsmaker.detection.evaluation_methods.evaluation_utils import compute_tw_labels


def coerce_value(value):
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def set_nested_attr(obj, dotted_key, value):
    current = obj
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = getattr(current, part)
    setattr(current, parts[-1], value)


def to_json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, numbers.Number):
        return value
    if isinstance(value, (str, bool)) or value is None:
        return value
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--edge-loss-dir", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()

    runtime_args, _ = get_runtime_required_args(
        True,
        args=[
            args.model,
            args.dataset,
            "--artifact_dir",
            "/home/artifacts",
            "--evaluation.best_model_selection",
            "last_epoch",
            "--exp",
            args.run_name,
        ],
    )
    cfg = get_yml_cfg(runtime_args)
    cfg.training._edge_losses_dir = args.edge_loss_dir
    cfg.evaluation.used_method = "temporal_risk_evaluation"

    for override in args.set:
        key, value = override.split("=", 1)
        set_nested_attr(cfg, key, coerce_value(value))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_path = os.path.join(
        "/home/artifacts/evaluation_research",
        f"{args.run_name}_{timestamp}",
        args.dataset,
    )
    cfg.evaluation._task_path = task_path
    cfg.evaluation._results_dir = os.path.join(task_path, "results")
    cfg.evaluation._precision_recall_dir = os.path.join(task_path, "precision_recall_dir")
    os.makedirs(cfg.evaluation._results_dir, exist_ok=True)
    os.makedirs(cfg.evaluation._precision_recall_dir, exist_ok=True)

    epoch_dir = f"model_epoch_{args.epoch}"
    val_tw_path = os.path.join(args.edge_loss_dir, "val", epoch_dir)
    test_tw_path = os.path.join(args.edge_loss_dir, "test", epoch_dir)
    tw_to_malicious_nodes = compute_tw_labels(cfg)
    stats = trusted_risk_evaluation.main(
        val_tw_path,
        test_tw_path,
        epoch_dir,
        cfg,
        tw_to_malicious_nodes=tw_to_malicious_nodes,
    )
    stats["epoch"] = args.epoch
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "dataset": args.dataset,
        "edge_loss_dir": args.edge_loss_dir,
        "epoch": args.epoch,
        "task_path": task_path,
        "overrides": args.set,
        "stats": {
            key: scalar
            for key, value in stats.items()
            for scalar in [to_json_scalar(value)]
            if scalar is not None
        },
    }
    with open(os.path.join(task_path, "metadata.json"), "w") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
