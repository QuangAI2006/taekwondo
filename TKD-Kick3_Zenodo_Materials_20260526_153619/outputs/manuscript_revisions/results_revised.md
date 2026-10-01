# Revised Results

## Dataset Summary

The active TKD-Kick3 repository contained 765 retained three-class pose sequences from a protocol-level collection of 840 smartphone videos. The active file-level training split contained 648 sequences, and the validation split contained 104 sequences. Because participant identifiers were unavailable, subject leakage could not be excluded and the validation results are reported as sample-level validation.

## Backbone Comparison

The best recognition result was obtained by the BiLSTM with uniform temporal resampling and argmax inference. Across three seeds, this configuration achieved 75.3 +/- 2.3% accuracy, 74.3 +/- 2.3% macro-F1, 75.3 +/- 2.3% weighted-F1, and 74.5 +/- 2.3% balanced accuracy. The added LSTM-PA-Sweep experiment reached 71.2 +/- 1.0% accuracy, confirming that the phase-aligned sweep configuration did not improve the LSTM under the current split.

The Transformer with combined phase-aligned resampling and sweep inference reached 67.3 +/- 1.7% accuracy, compared with 62.2 +/- 2.4% for Transformer-UNI-Argmax. The GCN configurations were weaker, with approximately 51% accuracy. These results do not support retaining the Transformer as the primary recognition backbone on accuracy alone; the current evidence favors LSTM-UNI as the selected recognition module.

## Ablation Study

For the Transformer, UNI-Argmax achieved 62.2 +/- 2.4% accuracy. The deployable PA-Sweep configuration achieved 67.3 +/- 1.7%, while PA-SingleHypothesis reached 65.4 +/- 2.6%. The PA-Oracle analysis upper bound achieved 69.2 +/- 2.5%. Therefore, the observed improvement should be described as the effect of the combined phase-aligned resampling and sweep-inference configuration, not as the independent effect of temporal alignment alone.

## Runtime Analysis

Runtime was measured on a desktop CPU prototype using 30 validation videos, 10 per class. Pose extraction required 7.51 +/- 4.52 s per video, whereas recognition inference required only 0.0030 +/- 0.0014 s per video. The total processing time excluding feedback rendering was 7.53 +/- 4.53 s per video. The pipeline should therefore be described as a desktop-CPU prototype using smartphone-captured videos, not as a validated on-device mobile deployment.

## Human-AI Agreement

The expert workbook contained 30 videos scored independently by three taekwondo experts. Expert score reliability was high, with ICC(A,3) = 0.915. The system classification matched the expert majority class in 83.3% of samples, with Cohen's kappa = 0.750. For total scores, Spearman rho was 0.627, Pearson r was 0.606, MAE was 0.601 points, and Bland-Altman bias was +0.341 points. These findings provide preliminary evidence of agreement but should not be described as expert-equivalent scoring.
