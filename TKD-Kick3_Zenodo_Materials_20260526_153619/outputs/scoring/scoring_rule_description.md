# Scoring Rule Audit

The active scoring implementation is `scripts/scoring_api.py`. It implements a 10-point score composed of a 4-point accuracy component and a 6-point expressiveness component.

Important terminology correction: the variable named `power_raw` in the active implementation is not a biomechanical power measurement. It is derived from two-dimensional pose kinematics and should be described as a kinematic vigor index or impact-speed proxy. The manuscript should avoid the term mechanical power unless force or torque data are collected.

| Component | Points | Implementation Status |
|---|---:|---|
| Accuracy | 4.0 | Implemented with kick-specific items FR/RH/AX_A1-A4 |
| Expressiveness | 6.0 | Implemented with common speed, kinematic vigor, height, and smoothness items |
| Total | 10.0 | `total10 = accuracy4 + expression6`, clipped to [0, 10] |

See `scoring_rule_configuration.csv` for item-level weights and code-derived thresholds.

Known limitation: `scripts/scoring_rules.py` is a compact legacy scoring module and does not exactly match the active `scoring_api.py`. The manuscript should cite the active API rules as the source of truth.
