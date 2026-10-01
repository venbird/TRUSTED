# Data preparation

## Database

Place PostgreSQL dumps in the directory set by `INPUT_DIR` in `.env`.
Use the database name as the filename, for example `cadets_e3.dump`.

```bash
docker exec postgres bash /scripts/load_dumps.sh
```

The restore script skips databases that already contain tables.

To download a supported dataset:

```bash
export DATASET_ACCESS_TOKEN='<access-token>'
bash download_datasets.sh cadets_e3
```

Downloads are saved to `./data/`. Use `INPUT_DIR=./data` before starting
PostgreSQL. For raw audit logs, see
[Dataset preprocessing](../dataset_preprocessing/README.md).

## Labels

Place evaluation label CSV files under `Ground_Truth/orthrus/`, or mount the
label directory at `/home/pids/Ground_Truth/orthrus` in `compose-pidsmaker.yml`.

Database names, label filenames, and train/validation/test dates are listed in
`DATASET_DEFAULT_CONFIG` in `pidsmaker/config/config.py`.
