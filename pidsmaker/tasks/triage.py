def main(cfg):
    method = cfg.triage.used_method
    if method is None or str(method).strip().lower() == "none":
        return None
    raise ValueError(
        "Integrated triage is disabled. Run "
        "pidsmaker/triage/tracing_methods/trust_trace_v3.py explicitly with frozen inputs."
    )
