# Data preparation

TRUSTED does not redistribute raw DARPA TC, OpTC, Ground Truth, database dumps,
or processed experiment artifacts.

The retained scripts under `dataset_preprocessing/` are adapted from PIDSMaker:

- `dataset_preprocessing/darpa_tc/` prepares DARPA Transparent Computing data.
- `dataset_preprocessing/optc/` prepares OpTC data.
- `dataset_preprocessing/create_database.sh` creates the PostgreSQL schema.
- `scripts/load_dumps.sh` restores user-provided database dumps mounted at
  `/data`.

Follow the access terms of each original dataset provider. Dataset splits and
labels used for evaluation must remain external to TrustTrace-v3; Ground Truth
must never be supplied to its incident selection or evidence-ranking inputs.

When using `download_datasets.sh`, inject the provider token through the
`DATASET_ACCESS_TOKEN` environment variable. The repository does not contain or
persist a token.
