"""Edge-loss inference used by TRUSTED."""

import os
import time
import tracemalloc

import numpy as np
import pandas as pd
import torch

from pidsmaker.utils.utils import (
    get_device,
    log,
    log_tqdm,
    ns_time_to_datetime_US,
    set_seed,
)


@torch.no_grad()
def test_edge_level(data, model, split, model_epoch_file, cfg, device):
    model.eval()
    start_time = data.t[0]
    results = model(data, inference=True, validation=split == "val")
    edge_losses = results["loss"]
    original_edges = data.original_edge_index
    edge_df = pd.DataFrame(
        {
            "loss": edge_losses.cpu().numpy().astype(float),
            "srcnode": original_edges[0, :].cpu().numpy().astype(int),
            "dstnode": original_edges[1, :].cpu().numpy().astype(int),
            "time": data.t.cpu().numpy().astype(int),
            "edge_type": (data.edge_type.max(dim=1).indices + 1).cpu().numpy().astype(int),
        }
    )
    interval = (
        ns_time_to_datetime_US(start_time)
        + "~"
        + ns_time_to_datetime_US(edge_df["time"].max())
    )
    logs_dir = os.path.join(cfg.training._edge_losses_dir, split, model_epoch_file)
    os.makedirs(logs_dir, exist_ok=True)
    edge_df.to_csv(
        os.path.join(logs_dir, interval + ".csv"),
        sep=",",
        header=True,
        index=False,
        encoding="utf-8",
    )
    return edge_losses.cpu().numpy().tolist()


def main(cfg, model, val_data, test_data, epoch, split, logging=True):
    set_seed(cfg)
    if cfg._is_node_level:
        raise ValueError("TRUSTED requires the released edge-prediction objective")
    split_map = {
        "all": [(val_data, "val"), (test_data, "test")],
        "val": [(val_data, "val")],
        "test": [(test_data, "test")],
    }
    if split not in split_map:
        raise ValueError(f"Invalid split {split}")

    inference_device = cfg.training.inference_device
    if inference_device is not None:
        if inference_device not in ["cpu", "cuda"]:
            raise ValueError(f"Invalid inference device {inference_device}")
        device = torch.device(inference_device)
    else:
        device = get_device(cfg)
    use_cuda = device == torch.device("cuda")
    model.to_device(device)
    model_epoch_file = f"model_epoch_{epoch}"
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device=device)

    val_score = 0.0
    peak_cpu_memory = 0.0
    peak_gpu_memory = 0.0
    times_per_batch = []
    losses_by_split = {}
    for dataset, split_name in split_map[split]:
        description = "Validation" if split_name == "val" else "Testing"
        tracemalloc.start()
        all_losses = []
        for graphs in dataset:
            for graph in log_tqdm(graphs, desc=description, logging=logging):
                graph.to(device=device)
                start = time.time()
                losses = test_edge_level(
                    data=graph,
                    model=model,
                    split=split_name,
                    model_epoch_file=model_epoch_file,
                    cfg=cfg,
                    device=device,
                )
                all_losses.extend(losses)
                times_per_batch.append(time.time() - start)
                graph.to("cpu")
                if use_cuda:
                    torch.cuda.empty_cache()

        _, peak = tracemalloc.get_traced_memory()
        peak_cpu_memory = max(peak_cpu_memory, peak / (1024**3))
        tracemalloc.stop()
        if use_cuda:
            peak = torch.cuda.max_memory_allocated(device=device) / (1024**3)
            peak_gpu_memory = max(peak_gpu_memory, peak)
            torch.cuda.reset_peak_memory_stats(device=device)

        mean_loss = np.mean(all_losses)
        losses_by_split[split_name] = mean_loss
        if split_name == "val":
            val_score = model.get_val_ap()
        if logging:
            log(
                f"[@epoch{epoch:02d}] {description} finished - "
                f"Loss: {mean_loss:.4f}",
                return_line=True,
            )

    del model
    return {
        "val_score": val_score,
        "val_loss": losses_by_split.get("val"),
        "test_loss": losses_by_split.get("test"),
        "peak_inference_cpu_memory": peak_cpu_memory,
        "peak_inference_gpu_memory": peak_gpu_memory,
        "time_per_batch_inference": np.mean(times_per_batch),
    }
