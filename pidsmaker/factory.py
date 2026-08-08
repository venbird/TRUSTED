"""Factories for the released TRUSTED model components."""

import torch
import torch.nn as nn

from pidsmaker.decoders import CustomEdgeMLP
from pidsmaker.encoders import GraphAttentionEmbedding, LinearEncoder, TGNEncoder
from pidsmaker.losses import cross_entropy
from pidsmaker.model import Model
from pidsmaker.objectives import EdgeTypePrediction, ValidationWrapper
from pidsmaker.tgn import IdentityMessage, LastAggregator, TGNMemory, TimeEncodingMemory
from pidsmaker.utils.data_utils import GraphReindexer
from pidsmaker.utils.dataset_utils import get_node_map, get_num_edge_type, get_rel2id


def build_model(data_sample, device, cfg, max_node_num):
    msg_dim, _, in_dim = get_dimensions_from_data_sample(data_sample)
    graph_reindexer = GraphReindexer(
        device=device,
        num_nodes=max_node_num,
        fix_buggy_graph_reindexer=cfg.batching.fix_buggy_graph_reindexer,
    )
    encoder = encoder_factory(
        cfg,
        msg_dim=msg_dim,
        in_dim=in_dim,
        device=device,
        max_node_num=max_node_num,
        graph_reindexer=graph_reindexer,
    )
    objectives = objective_factory(cfg, in_dim=in_dim, graph_reindexer=graph_reindexer)
    return Model(
        encoder=encoder,
        objectives=objectives,
        objective_few_shot=None,
        device=device,
        is_running_mc_dropout=False,
        use_few_shot=False,
        freeze_encoder=False,
    ).to(device)


def encoder_factory(cfg, msg_dim, in_dim, device, max_node_num, graph_reindexer):
    methods = [value.strip() for value in cfg.training.encoder.used_methods.split(",")]
    use_tgn = "tgn" in methods
    graph_methods = [value for value in methods if value != "tgn"]
    if len(graph_methods) != 1:
        raise ValueError(f"TRUSTED expects exactly one graph encoder, got {methods}")

    original_in_dim = in_dim
    if use_tgn:
        in_dim = cfg.training.encoder.tgn.tgn_memory_dim

    method = graph_methods[0]
    if method == "graph_attention":
        params = cfg.training.encoder.graph_attention
        encoder = GraphAttentionEmbedding(
            in_dim=in_dim,
            hid_dim=cfg.training.node_hid_dim,
            out_dim=cfg.training.node_out_dim,
            edge_dim=get_edge_dim(cfg, msg_dim) or None,
            activation=activation_fn_factory(params.activation),
            dropout=cfg.training.encoder.dropout,
            num_heads=params.num_heads,
            concat=params.concat,
            flow=params.flow,
            num_layers=params.num_layers,
        )
    elif method == "none":
        encoder = LinearEncoder(in_dim, cfg.training.node_out_dim, cfg.training.encoder.dropout)
    else:
        raise ValueError(f"Invalid TRUSTED encoder {method}")

    if not use_tgn:
        return encoder

    tgn_cfg = cfg.training.encoder.tgn
    edge_features = [value.strip() for value in cfg.batching.edge_features.split(",")]
    use_time_encoding = "time_encoding" in edge_features
    if tgn_cfg.use_memory:
        memory = TGNMemory(
            max_node_num,
            msg_dim,
            tgn_cfg.tgn_memory_dim,
            tgn_cfg.tgn_time_dim,
            message_module=IdentityMessage(
                msg_dim, tgn_cfg.tgn_memory_dim, tgn_cfg.tgn_time_dim
            ),
            aggregator_module=LastAggregator(),
            device=device,
        )
    elif use_time_encoding:
        memory = TimeEncodingMemory(max_node_num, tgn_cfg.tgn_time_dim, device=device)
    else:
        memory = None

    return TGNEncoder(
        encoder=encoder,
        memory=memory,
        time_encoder=memory.time_enc if memory else None,
        in_dim=original_in_dim,
        memory_dim=tgn_cfg.tgn_memory_dim,
        use_node_feats_in_gnn=tgn_cfg.use_node_feats_in_gnn,
        edge_features=edge_features,
        device=device,
        use_memory=tgn_cfg.use_memory,
        use_time_enc=use_time_encoding,
        edge_dim=get_edge_dim(cfg, msg_dim),
        use_time_order_encoding=tgn_cfg.use_time_order_encoding,
        project_src_dst=tgn_cfg.project_src_dst,
        node_map=get_node_map(from_zero=True),
        edge_map=get_rel2id(cfg, from_zero=True),
    )


def objective_factory(cfg, in_dim, graph_reindexer):
    methods = [value.strip() for value in cfg.training.decoder.used_methods.split(",")]
    if methods != ["predict_edge_type"]:
        raise ValueError(f"TRUSTED only supports predict_edge_type, got {methods}")
    params = cfg.training.decoder.predict_edge_type
    decoder = CustomEdgeMLP(
        in_dim=cfg.training.node_out_dim,
        out_dim=get_num_edge_type(cfg),
        architecture=params.edge_mlp.architecture_str,
        dropout=cfg.training.encoder.dropout,
        src_dst_projection_coef=params.edge_mlp.src_dst_projection_coef,
    )
    objective = EdgeTypePrediction(
        decoder=decoder,
        loss_fn=cross_entropy,
        balanced_loss=params.balanced_loss,
        edge_type_dim=get_num_edge_type(cfg),
    )
    return [
        ValidationWrapper(
            objective,
            graph_reindexer,
            is_edge_type_prediction=True,
            use_few_shot=False,
        )
    ]


def activation_fn_factory(activation):
    functions = {
        "sigmoid": nn.Sigmoid,
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "prelu": nn.PReLU,
        "none": nn.Identity,
    }
    try:
        return functions[activation]()
    except KeyError as exc:
        raise ValueError(f"Invalid activation function {activation}") from exc


def optimizer_factory(cfg, parameters):
    return torch.optim.Adam(
        parameters,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )


def optimizer_few_shot_factory(cfg, parameters):
    raise RuntimeError("Few-shot training is not part of the released TRUSTED method")


def get_dimensions_from_data_sample(data):
    edge_dim = data.edge_feats.shape[1] if hasattr(data, "edge_feats") else None
    msg_dim = data.msg.shape[1] if hasattr(data, "msg") else edge_dim
    in_dim = data.x_src.shape[1] if hasattr(data, "x_src") else data.x.shape[1]
    return msg_dim, edge_dim, in_dim


def get_edge_dim(cfg, msg_dim):
    edge_dim = 0
    edge_features = [value.strip() for value in cfg.batching.edge_features.split(",")]
    for feature in edge_features:
        if feature in {"edge_type", "edge_type_triplet"}:
            edge_dim += get_num_edge_type(cfg)
        elif feature == "msg":
            edge_dim += msg_dim
        elif feature == "time_encoding":
            if "tgn" not in cfg.training.encoder.used_methods:
                raise TypeError("time_encoding requires TGN")
            edge_dim += cfg.training.encoder.tgn.tgn_memory_dim
        elif feature != "none":
            raise ValueError(f"Invalid edge feature {feature}")
    return edge_dim
