# M11 affect visibility in Sleep Replay Monitor

M11 affect can change very slowly because `EmotionalDriveConfig.ema_decay` is high
and live values are rounded in the UI. This does not mean M11 is inactive.

This patch adds monitor-only diagnostics:

```text
m11_delta:
    valence_delta
    arousal_delta
    stress_delta
    panic_delta
    curiosity_delta

m11_range:
    stress_min / stress_max
    panic_min / panic_max
    curiosity_min / curiosity_max

m11_activity:
    change_score
    trend
    samples
```

It does not change M11 emotional dynamics. It only makes small changes visible.
