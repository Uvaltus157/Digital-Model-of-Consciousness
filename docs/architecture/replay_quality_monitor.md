# Replay Quality Monitor

Shows whether sleep/replay is actually integrating experience, not just active.

It tracks:

```text
selected_episode_summary
selected_identity_token
replay_source
dream_pressure and delta
stress / relief / coherence deltas
expected_affect_delta
M13 relevance
M4 identity gate/stability/novelty
M5 replay seed gate/norm
quality_score / quality_ema
verdict: idle | weak | replaying | integrating
```

This is a read-only diagnostic layer based on `latest_out`.
