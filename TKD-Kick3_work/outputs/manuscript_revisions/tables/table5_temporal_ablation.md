Table 5. Ablation of temporal normalization and inference strategy for the Transformer model

| Configuration | Deployable | Accuracy (%) | Macro-F1 (%) | Balanced Acc. (%) | Interpretation |
| --- | --- | --- | --- | --- | --- |
| UNI-Argmax | Yes | 62.2 +/- 2.4 | 60.5 +/- 2.1 | 60.8 +/- 1.9 | Controlled comparison |
| PA-SingleHyp. | Yes | 65.4 +/- 2.6 | 64.6 +/- 2.1 | 65.2 +/- 1.8 | Controlled comparison |
| PA-Sweep | Yes | 67.3 +/- 1.7 | 66.5 +/- 1.3 | 67.0 +/- 1.1 | Combined PA+sweep |
| PA-Oracle | No, analysis upper bound | 69.2 +/- 2.5 | 68.1 +/- 2.0 | 68.5 +/- 1.7 | Controlled comparison |
