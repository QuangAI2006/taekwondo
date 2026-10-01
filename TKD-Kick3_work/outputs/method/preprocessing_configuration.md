# Preprocessing Configuration

| Item | Value | Code source |
| --- | --- | --- |
| Pose estimator | MediaPipe PoseLandmarker, pose_landmarker_full.task | scripts/extract_pose_mediapipe.py |
| MediaPipe version | 0.10.14 in the current environment | outputs/runtime/runtime_summary.md |
| Number of poses tracked | 4 candidate poses, then kick-performer selection by lower-limb motion and coverage | scripts/extract_pose_mediapipe.py |
| Minimum detection confidence | 0.5 | scripts/extract_pose_mediapipe.py |
| Minimum presence confidence | 0.5 | scripts/extract_pose_mediapipe.py |
| Minimum tracking confidence | 0.5 | scripts/extract_pose_mediapipe.py |
| Coordinate type | 2D image pixel coordinates reduced to 13 landmarks | scripts/extract_pose_mediapipe.py |
| Visibility threshold in LSTM/GCN baseline normalization | tau = 0.5 | scripts/train_kick3_baselines.py |
| Visibility threshold in Transformer feature code | VTH = 0.2 | scripts/train_kick3_transformer_ablation.py |
| Spatial center | hip midpoint if both hips are visible; shoulder midpoint fallback; otherwise joint mean | model training and evaluation scripts |
| Scale normalization | max(shoulder width, hip width, 1.0) | model training and evaluation scripts |
| Input feature dimension | 52 per frame: 26 position values plus 26 first-order velocity values | model training and evaluation scripts |
| Sequence length | T = 96 | training checkpoints and scripts |
| Uniform resampling | linear temporal resize to 96 frames | model training and evaluation scripts |
| Phase-aligned resampling | pivot-centered asymmetric window, pre = 0.45, post = 0.55 | model training and evaluation scripts |
| Pivot rule | front/roundhouse: maximum attacking-ankle to ipsilateral-hip distance; axe: highest attacking ankle / minimum y | baseline code; Transformer code uses related candidate-class pivoting |
| Attacking leg handling | heuristic from lower-limb motion or ankle-hip excursion; no verified participant-level side metadata | scripts/train_kick3_baselines.py; scripts/train_kick3_transformer_ablation.py |

Important manuscript note: the current codebase uses different visibility thresholds across model families. The manuscript should not state a single universal threshold unless the code is harmonized in a later run.
