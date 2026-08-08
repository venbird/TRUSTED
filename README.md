# TRUSTED

TRUSTED is a provenance-based APT detection and evidence-reconstruction method.
It combines training-derived semantic, structural, and behavioral trust features
with temporal risk aggregation, then uses TrustTrace-v3 to reconstruct a compact,
connected, time-consistent evidence graph from real audit events.

This repository contains only the code required by TRUSTED. It does not include
other detection systems from PIDSMaker, trained models, raw datasets, Ground
Truth, result tables, logs, or experiment artifacts.

## Components

- `pidsmaker/featurization/trusted_utils.py`: TRUSTED semantic, structural, and
  trust feature construction.
- `pidsmaker/detection/evaluation_methods/trusted_risk_evaluation.py`: temporal
  direct-risk, propagation, state, and bounded trust-context aggregation.
- `pidsmaker/triage/tracing_methods/trust_trace_v3.py`: evidence-only provenance
  reconstruction. This module never reads Ground Truth.
- `config/trusted_main.yml`: canonical paper configuration.
- `config/trusted_*.yml`: TRUSTED-only ablation configurations.

The `pidsmaker` package name is retained because TRUSTED reuses a minimal subset
of the PIDSMaker pipeline. The method and all public configuration identifiers
use the name TRUSTED.

## Environment

The reproducible environment is defined in `Dockerfile`. It uses Python 3.9,
PyTorch 1.13.1 with CUDA 11.7, and PyTorch Geometric 2.5.3.

```bash
cp .env.example .env
docker network create shared_network 2>/dev/null || true
docker compose -f compose-postgres.yml up -d
docker compose -f compose-pidsmaker.yml up -d --build
```

Dataset acquisition and PostgreSQL preparation are described in
[`docs/data.md`](docs/data.md). Raw data and labels must be obtained from their
original providers and are not redistributed here.

## Detection

Run the canonical configuration inside the container:

```bash
docker exec trusted-pids bash -lc \
  'cd /home/pids && python pidsmaker/main.py trusted_main CADETS_E3 \
   --artifact_dir /home/artifacts --restart_from_scratch'
```

`trusted_main` uses `tgn,graph_attention`, TRUSTED featurization,
`temporal_risk_evaluation`, `best_adp`, `score_fusion_mode=standard`, and a
validation-derived `max_val_loss` threshold.

For evaluation from frozen edge losses:

```bash
docker exec trusted-pids bash -lc \
  'cd /home/pids && PYTHONPATH=/home/pids \
   python scripts/run_trusted_evaluation.py \
   --model trusted_main --dataset CADETS_E3 \
   --edge-loss-dir /home/artifacts/<edge-loss-run> \
   --epoch <epoch> --run-name <name> \
   --set evaluation.temporal_risk_evaluation.score_fusion_mode=standard \
   --set evaluation.temporal_risk_evaluation.threshold_method=max_val_loss'
```

## TrustTrace-v3

List incidents from a frozen detector result:

```bash
python pidsmaker/triage/tracing_methods/trust_trace_v3.py \
  --result <result.pth> --edge-loss-dir <epoch-csv-directory> \
  --list-incidents
```

Reconstruct one incident:

```bash
python pidsmaker/triage/tracing_methods/trust_trace_v3.py \
  --result <result.pth> \
  --graphs-dir <transformed-graph-root> \
  --edge-loss-dir <epoch-csv-directory> \
  --incident-id incident_0 \
  --output-dir <new-output-directory>
```

TrustTrace-v3 refuses an existing output directory. Its manifest records source
paths and SHA-256 hashes. Ground Truth must be used only for a separate posterior
evaluation and never for incident formation, candidate ranking, or graph search.

See [`docs/reproduction.md`](docs/reproduction.md) for the complete workflow and
[`docs/method.md`](docs/method.md) for the implementation map.

## License and attribution

TRUSTED is released under Apache-2.0. The common pipeline code is adapted from
[PIDSMaker](https://github.com/ubc-provenance/PIDSMaker); see `NOTICE`.
