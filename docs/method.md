# Implementation map

## Detection

TRUSTED first builds training-only profiles from provenance graphs. Semantic
hashing represents normalized entity labels; structural statistics describe
degree and relation distributions; behavioral trust measures consistency with
the training history. The feature implementation is in
`pidsmaker/featurization/trusted_utils.py`.

The canonical encoder uses a Temporal Graph Network wrapper and graph attention
to predict edge types. Per-edge prediction loss supplies direct anomaly evidence.
`trusted_risk_evaluation.py` aggregates this evidence into node risk using direct
risk, neighborhood propagation, temporal state, and bounded trust context. The
canonical configuration retains direct evidence and selects its threshold from
validation data.

## Evidence reconstruction

TrustTrace-v3 consumes frozen node results, transformed provenance graphs, and
edge-loss CSV files. It groups alert windows into incidents, resolves entities,
deduplicates real event UUIDs, assigns conservative event semantics, aggregates
events, and selects connected evidence under fixed node and edge budgets.

The trace module does not import the detector, old TrustTrace versions, report
providers, or Ground Truth. Missing loss and identity evidence remains explicitly
unknown rather than being synthesized.
