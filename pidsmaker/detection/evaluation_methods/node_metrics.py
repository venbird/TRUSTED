"""Reporting helpers for TRUSTED node-risk results."""

import os
from collections import defaultdict

import torch

from pidsmaker.detection.evaluation_methods.evaluation_utils import (
    classifier_evaluation,
    compute_discrimination_score,
    compute_discrimination_tp,
    get_detected_tps_node_level,
    get_metrics_if_all_attacks_detected,
    plot_detected_attacks_vs_precision,
    plot_discrimination_metric,
    plot_scores_neat,
    plot_scores_with_paths_node_level,
    transform_attack2nodes_to_node2attacks,
)
from pidsmaker.utils.labelling import get_GP_of_each_attack
from pidsmaker.utils.utils import get_node_to_path_and_type, log


def analyze_false_positives(y_truth, y_preds, max_loss_windows, tw_to_malicious_nodes):
    fp_indices = [
        index
        for index, (truth, prediction) in enumerate(zip(y_truth, y_preds))
        if prediction and not truth
    ]
    malicious_windows = set(tw_to_malicious_nodes)
    count = sum(max_loss_windows[index] in malicious_windows for index in fp_indices)
    return count / len(fp_indices) if fp_indices else float("nan")


def evaluate_node_results(results, threshold, model_epoch_dir, cfg, tw_to_malicious_nodes):
    os.makedirs(cfg.evaluation._results_dir, exist_ok=True)
    results_save_path = os.path.join(cfg.evaluation._results_dir, "results.pth")
    torch.save(results, results_save_path)
    log(f"Results saved to {results_save_path}")

    node_to_path = get_node_to_path_and_type(cfg)
    out_dir = cfg.evaluation._precision_recall_dir
    os.makedirs(out_dir, exist_ok=True)
    adp_image = os.path.join(out_dir, f"adp_curve_{model_epoch_dir}.png")
    scores_image = os.path.join(out_dir, f"scores_{model_epoch_dir}.png")
    neat_scores_image = os.path.join(out_dir, f"neat_scores_{model_epoch_dir}.svg")
    discrimination_image = os.path.join(out_dir, f"discrim_curve_{model_epoch_dir}.png")

    attack_to_ground_positive = get_GP_of_each_attack(cfg)
    attack_to_true_positive = defaultdict(int)
    nodes, y_truth, y_preds, pred_scores, max_loss_windows = [], [], [], [], []
    for node_id, result in results.items():
        nodes.append(node_id)
        y_truth.append(result["y_true"])
        y_preds.append(result["y_hat"])
        pred_scores.append(result["score"])
        max_loss_windows.append(result["tw_with_max_loss"])
        if result["y_true"] == 1:
            log(
                f"-> Malicious node {node_id:<7}: loss={result['score']:.3f} | is TP:"
                + (" yes " if result["y_hat"] else " no ")
                + node_to_path[node_id]["path"]
            )
            if result["y_hat"]:
                for attack, values in attack_to_ground_positive.items():
                    if node_id in values["nids"]:
                        attack_to_true_positive[attack] += 1

    attack_to_nodes = {
        attack: values["nids"] for attack, values in attack_to_ground_positive.items()
    }
    node_to_attacks = transform_attack2nodes_to_node2attacks(attack_to_nodes)
    adp_score = plot_detected_attacks_vs_precision(
        pred_scores, nodes, node_to_attacks, y_truth, adp_image
    )
    discrimination_scores = compute_discrimination_score(
        pred_scores, nodes, node_to_attacks, y_truth
    )
    plot_discrimination_metric(pred_scores, y_truth, discrimination_image)
    discrimination_tp = compute_discrimination_tp(
        pred_scores, nodes, node_to_attacks, y_truth
    )
    plot_scores_with_paths_node_level(
        pred_scores,
        y_truth,
        nodes,
        max_loss_windows,
        tw_to_malicious_nodes,
        node_to_attacks,
        scores_image,
        cfg,
        threshold,
    )
    plot_scores_neat(
        pred_scores, y_truth, nodes, node_to_attacks, neat_scores_image, threshold
    )

    stats = classifier_evaluation(y_truth, y_preds, pred_scores)
    fp_ratio = analyze_false_positives(
        y_truth, y_preds, max_loss_windows, tw_to_malicious_nodes
    )
    stats["fp_in_malicious_tw_ratio"] = round(fp_ratio, 3)
    stats["percent_detected_attacks"] = (
        round(len(attack_to_true_positive) / len(attack_to_ground_positive), 2)
        if attack_to_ground_positive
        else 0
    )
    fps, tps, precision, recall = get_metrics_if_all_attacks_detected(
        pred_scores, nodes, attack_to_ground_positive
    )
    stats.update(
        {
            "fps_if_all_attacks_detected": fps,
            "tps_if_all_attacks_detected": tps,
            "precision_if_all_attacks_detected": precision,
            "recall_if_all_attacks_detected": recall,
            "adp_score": round(adp_score, 3),
            "threshold": float(threshold),
        }
    )
    stats.update({key: round(value, 4) for key, value in discrimination_scores.items()})
    attack_to_tps = get_detected_tps_node_level(
        pred_scores, nodes, node_to_attacks, y_truth, cfg
    )
    for attack, detected_tps in attack_to_tps.items():
        stats[f"tps_{attack}"] = str(detected_tps)
    stats.update(discrimination_tp)

    results_file = os.path.join(out_dir, f"result_{model_epoch_dir}.pth")
    stats_file = os.path.join(out_dir, f"stats_{model_epoch_dir}.pth")
    scores_file = os.path.join(out_dir, f"scores_{model_epoch_dir}.pkl")
    torch.save(results, results_file)
    torch.save(stats, stats_file)
    torch.save(
        {
            "pred_scores": pred_scores,
            "y_preds": y_preds,
            "y_truth": y_truth,
            "nodes": nodes,
            "node2attacks": node_to_attacks,
        },
        scores_file,
    )
    stats["scores_file"] = scores_file
    stats["neat_scores_img_file"] = neat_scores_image
    return stats
