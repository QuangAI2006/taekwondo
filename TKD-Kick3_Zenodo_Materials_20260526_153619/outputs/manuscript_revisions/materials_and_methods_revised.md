# Revised Materials and Methods

## Participants and Smartphone-Video Acquisition

The TKD-Kick3 dataset was collected in routine first-year university physical-education classes. A total of 140 students from four teaching classes participated. Most participants were beginners or had only initial exposure to taekwondo. The acquisition protocol included three fundamental kicks: front kick, roundhouse kick, and axe kick. Each participant was asked to perform two videos per kick type, giving 840 planned smartphone videos at the protocol level. No wearable or force-measurement sensors were used.

The present analysis used the cleaned pose-sequence repository available in the project directory. Side-kick recordings were excluded from the three-class study. Ethical approval and class-level notification documents were available, but the current data files did not include participant identifiers.

## Dataset Construction and Exclusion Process

From the protocol-level total of 840 recordings, 765 active three-class pose sequences were retained. The active dataset contained 288 front-kick sequences, 190 roundhouse-kick sequences, and 287 axe-kick sequences. The difference between the protocol-level total and the retained pose sequences was 75 recordings. The current files identify one front-kick video with no usable active sequence and 57 disabled side-kick videos; however, the complete per-reason exclusion counts for failed pose extraction, unstable tracking, severe occlusion, missing action phases, and category mismatch could not be reconstructed from the available files.

## Data Partition and Leakage Audit

The current repository contained a file-level split with 648 training sequences and 104 validation sequences, plus 13 additional release/example sequences. The validation set contained 38 front kicks, 25 roundhouse kicks, and 41 axe kicks. Participant identifiers were not present in the active `.npz` files or in a separate metadata table. Therefore, a participant-independent split could not be generated, and the current recognition results should be interpreted as sample-level validation results. The 13 release/example sequences were not treated as a formal independent test set.

## Pose Extraction and Feature Construction

Pose extraction used MediaPipe PoseLandmarker with the `pose_landmarker_full.task` model. The extraction configuration used up to four candidate poses per frame and selected the likely performer by lower-limb motion and tracking coverage. The minimum pose detection, presence, and tracking confidences were 0.5. The pipeline reduced the MediaPipe pose output to 13 two-dimensional keypoints: nose, bilateral shoulders, elbows, wrists, hips, knees, and ankles.

For each frame, keypoints were spatially normalized by subtracting the hip midpoint when both hips were visible, falling back to the shoulder midpoint when the hips were unavailable, and finally to the mean of available joints. The scale factor was the maximum of shoulder width, hip width, and 1.0. Each frame was represented by 26 normalized position values and 26 first-order velocity values, resulting in a 52-dimensional input feature vector. Sequences were resampled to 96 frames.

## Temporal Alignment and Recognition Models

Uniform resampling linearly resized each sequence to 96 frames. Phase-aligned resampling used a pivot-centered asymmetric temporal window with pre = 0.45 and post = 0.55. For front and roundhouse kicks, the pivot was based on the maximum attacking-ankle to ipsilateral-hip distance. For axe kicks, the pivot was based on the highest attacking ankle. At inference, sweep inference evaluated candidate class-specific alignments and selected the class with the highest self-consistent probability.

Three recognition backbones were evaluated: a bidirectional LSTM, a spatial GCN baseline, and a Transformer encoder. All models used 52-dimensional input features and 96-frame sequences. Training used Adam with learning rate 1e-3, batch size 8, dropout 0.2, and 40 epochs. Experiments were repeated with seeds 0, 1, and 2. The Transformer used label smoothing of 0.05; the baseline scripts did not apply label smoothing.

## Explainable Scoring Rules

The scoring module produced a 10-point score consisting of 4 points for accuracy and 6 points for expressiveness. Accuracy used kick-specific items for chambering, path/alignment, height, knee extension, recovery, stability, and landing or hand-guard control. Expressiveness used common items for speed, a kinematic vigor index, height, and smoothness. The variable previously named `power_raw` in code should not be interpreted as mechanical power; it is a two-dimensional pose-derived kinematic proxy.

## Statistical Analysis

Recognition performance was reported as accuracy, macro-F1, weighted-F1, balanced accuracy, class-wise precision, recall, F1, and confusion matrices. For three-seed experiments, results are reported as mean +/- standard deviation. Expert agreement was recalculated from the source workbook for 30 videos using expert majority class labels and mean expert scores. Metrics included ICC(A,3), Cohen's kappa, Spearman and Pearson correlations, MAE, MSE, and Bland-Altman bias and limits of agreement.
