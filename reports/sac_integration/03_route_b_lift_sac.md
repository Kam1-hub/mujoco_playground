# Route B: LIFT-Style Asymmetric SAC

Status: TODO for next agent.

## Goal

Implement the main SAC baseline:

- actor obs key: `state`
- critic obs key: `privileged_state`
- tanh Gaussian actor
- twin Q
- target Q
- trainable alpha
- uniform replay
- UTD via `grad_updates_per_step`
- deterministic eval
- checkpoint save

## Required Files

```text
learning/sac_lift/
learning/train_jax_sac_lift.py
g1_env/config/sac_params.py
```

## Key Constraints

- Do not copy LIFT forks.
- Do not double-scale G1 actions.
- Do not inherit LIFT CLI bugs.
- Add warnings for fallback privileged obs, fallback truncation, or fallback action scale.

Runtime status: NOT VALIDATED.

