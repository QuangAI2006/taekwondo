# Selected Final Recognition Configuration

Protocol: current cleaned sample-level validation split; participant identifiers are unavailable, so this is not a subject-independent result.

Selected by validation accuracy among the completed experiment matrix:

- Final recognition configuration: LSTM-UNI (BiLSTM, uniform temporal resampling, argmax inference).
- Accuracy: 75.3 +/- 2.3%.
- Macro-F1: 74.3 +/- 2.3%.
- Weighted-F1: 75.3 +/- 2.3%.
- Balanced accuracy: 74.5 +/- 2.3%.

Important interpretation:

- LSTM-PA-Sweep was successfully added and reached 71.2 +/- 1.0% accuracy.
- LSTM-PA-Sweep outperformed Transformer-PA-Sweep (67.3 +/- 1.7%) but did not outperform LSTM-UNI.
- Therefore, the current data support LSTM as the stronger recognition backbone, and do not support retaining Transformer as the primary classifier on recognition accuracy alone.
- Because subject identifiers are unavailable, these conclusions are limited to sample-level validation.
