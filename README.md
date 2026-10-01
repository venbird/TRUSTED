# TRUSTED

Code for APT detection and attack tracing on provenance graphs.

## Setup

Requires Docker Compose and NVIDIA Container Toolkit.

```bash
git clone https://github.com/venbird/TRUSTED.git
cd TRUSTED
cp .env.example .env
```

Set data and output directories in `.env`, then start the containers:

```bash
docker network create shared_network 2>/dev/null || true
docker compose -f compose-postgres.yml up -d
docker compose -f compose-pidsmaker.yml up -d --build
```

Prepare the database and label files as described in [Data preparation](docs/data.md).

## Detection

```bash
docker exec trusted-pids bash -lc \
  'cd /home/pids && python pidsmaker/main.py trusted_main CADETS_E3 \
   --artifact_dir /home/artifacts --restart_from_scratch'
```

Replace `CADETS_E3` with the dataset name. Settings are in
`config/trusted_main.yml`. Outputs are saved under `ARTIFACTS_DIR`.

## Evaluation

To evaluate saved edge losses:

```bash
docker exec trusted-pids bash -lc \
  'cd /home/pids && PYTHONPATH=/home/pids \
   python scripts/run_trusted_evaluation.py \
   --model trusted_main --dataset CADETS_E3 \
   --edge-loss-dir "/home/artifacts/<edge-loss-run>" \
   --epoch "<epoch>" --run-name "<name>"'
```

The edge-loss root must contain `val/` and `test/` directories.

## Tracing

List incidents:

```bash
docker exec trusted-pids bash -lc \
  'cd /home/pids && python pidsmaker/triage/tracing_methods/trust_trace_v3.py \
   --result "<result.pth>" --edge-loss-dir "<epoch-csv-directory>" \
   --list-incidents'
```

Trace an incident:

```bash
docker exec trusted-pids bash -lc \
  'cd /home/pids && python pidsmaker/triage/tracing_methods/trust_trace_v3.py \
   --result "<result.pth>" --graphs-dir "<transformed-graph-root>" \
   --edge-loss-dir "<epoch-csv-directory>" --incident-id incident_0 \
   --output-dir "<new-output-directory>"'
```

Replace `<...>` with paths and values from the same run and epoch. Paths must
be accessible inside the container. Use an incident ID from the list above and
a new output directory. If you changed `TRAINING_CONTAINER_NAME` in `.env`,
replace `trusted-pids` in the commands.

## License

Apache-2.0. Includes code adapted from
[PIDSMaker](https://github.com/ubc-provenance/PIDSMaker); see `NOTICE`.
