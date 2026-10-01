# Revised Conclusion

This work developed and audited a smartphone-video-based pipeline for recognizing and scoring front, roundhouse, and axe kicks from monocular videos. Using 13 MediaPipe pose keypoints and 52-dimensional temporal features, the best current recognition result was achieved by the LSTM-UNI configuration, with 75.3 +/- 2.3% sample-level validation accuracy across three seeds. The expert-annotated subset showed preliminary agreement between system outputs and human ratings, but the evidence does not support claims of expert-equivalent scoring or real-time on-device deployment.

The main next step is to recover or reconstruct participant-level metadata so that leakage-free subject-independent validation can be performed. Future work should also evaluate the pipeline on additional recording environments, camera viewpoints, skill levels, and true mobile hardware.
