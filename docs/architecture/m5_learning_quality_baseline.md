# M5 Learning Quality Baseline

This monitor separates two questions:

```text
1. Is the sleep/replay circuit wired correctly?
2. Is M5 actually learning a useful world model?
```

The first question was covered by Sleep Replay Monitor and Replay Quality Monitor.

This baseline monitor focuses on M5 learning proxies:

```text
train_loss and delta
prediction_error and delta
reconstruction_error and delta
latent_coherence and delta
seed_gate / seed_norm / feedback_gate
seed_response
object reconstruction proxy
identity stability / novelty proxies
```

Important:

```text
Before M5 is trained, this monitor can only show signal response and metric availability.
It cannot prove semantic understanding.
```

Verdicts:

```text
untrained_or_no_data
idle
seed_reactive_untrained
tracking
improving
training_error
```
