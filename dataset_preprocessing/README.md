# Dataset preprocessing

Mount the unpacked audit logs at `/data` in `compose-pidsmaker.yml`.

Create the database schema, then import the logs. For CADETS E3:

```bash
docker exec postgres bash /dataset_preprocessing/create_database.sh CADETS_E3

docker exec trusted-pids bash -lc \
  'cd /home/pids && PYTHONPATH=/home/pids \
   python dataset_preprocessing/darpa_tc/create_database_e3.py \
   trusted_main CADETS_E3'
```

Choose the import script for the dataset:

| Data | Script |
| --- | --- |
| DARPA TC E3 | `darpa_tc/create_database_e3.py` |
| DARPA TC E5 | `darpa_tc/create_database_e5.py` |
| OpTC | `optc/create_database_optc.py` |

Replace the script path and dataset name in the commands above. Expected
database names and raw filenames are defined in `pidsmaker/config/config.py`
and the corresponding preprocessing scripts.
