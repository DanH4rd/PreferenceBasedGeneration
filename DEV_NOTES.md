# Dev Notes

Informal log of hypotheses, observations, and ideas worth revisiting. Not documentation — just working notes.

## 2026-09-11 — Slight overtraining per round may be desirable

Hypothesis: mild overtraining of the reward model (and/or latent) within a round is not
purely a downside. It could push the destination action into more varied/extreme regions
of the noise space, producing more diverse examples for `RoundsMemory`. Since each round
retrains from scratch on fresh + discounted-past data, any overtraining artifact from round
N gets corrected in round N+1 rather than compounding.

If true, this argues against being too conservative with `training_epochs_reward` /
`training_epochs_latent` — some overshoot per round trades short-term fit for long-term
diversity in the preference memory.

Untested. Worth checking by comparing action/image diversity across rounds at a couple of
epoch settings.
