# Reproduction workflow

1. Build the container environment from the supplied Dockerfile.
2. Obtain and preprocess an evaluated dataset as described in `data.md`.
3. Run `trusted_main` from scratch. Keep the generated configuration and edge
   losses associated with the same run.
4. Select model epochs using validation-derived policy only. The canonical
   configuration uses `best_adp`, standard score fusion, and `max_val_loss`.
5. Run TrustTrace-v3 from frozen detector results, transformed graphs, and the
   matching epoch edge-loss directory.
6. If Ground Truth evaluation is required, perform it only after the detector,
   trace configuration, code, and output hashes are frozen.

The repository intentionally contains no model checkpoints or result files.
Commands therefore use placeholders for run-specific artifact paths and epochs.

TRUSTED ablations are provided as `config/trusted_*.yml`. They are diagnostic or
component-level configurations and must not be presented as the canonical main
configuration.
