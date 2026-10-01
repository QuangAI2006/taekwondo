# Recognition Summary

Protocol: current cleaned sample-level validation split. Participant identifiers are unavailable, so these results are not subject-independent.

| Model | Setting | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | Balanced accuracy (%) |
|---|---|---:|---:|---:|---:|
| GCN | PA-Sweep | 51.0 +/- 1.7 | 49.9 +/- 1.9 | 50.4 +/- 1.5 | 51.2 +/- 2.9 |
| GCN | UNI-Argmax | 50.6 +/- 1.1 | 50.1 +/- 1.7 | 50.4 +/- 2.0 | 51.3 +/- 1.8 |
| LSTM | PA-Sweep | 71.2 +/- 1.0 | 69.9 +/- 1.9 | 71.1 +/- 1.1 | 70.3 +/- 2.5 |
| LSTM | UNI-Argmax | 75.3 +/- 2.3 | 74.3 +/- 2.3 | 75.3 +/- 2.3 | 74.5 +/- 2.3 |
| Transformer | PA-Oracle | 69.2 +/- 2.5 | 68.1 +/- 2.0 | 69.7 +/- 2.2 | 68.5 +/- 1.7 |
| Transformer | PA-SingleHypothesis | 65.4 +/- 2.6 | 64.6 +/- 2.1 | 65.9 +/- 2.4 | 65.2 +/- 1.8 |
| Transformer | PA-Sweep | 67.3 +/- 1.7 | 66.5 +/- 1.3 | 67.9 +/- 1.7 | 67.0 +/- 1.1 |
| Transformer | UNI-Argmax | 62.2 +/- 2.4 | 60.5 +/- 2.1 | 61.8 +/- 1.8 | 60.8 +/- 1.9 |
