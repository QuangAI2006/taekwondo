# Runtime Summary

Protocol: fixed validation runtime subset with 30 videos (10 per class) from the current cleaned sample-level split. Because participant identifiers are unavailable, this is not a subject-independent runtime subset.

Recognition model: LSTM-UNI seed 0 checkpoint `<PROJECT_ROOT>\models\checkpoints\sample_level\sample_lstm_uniform_seed0.pt`.
Trainable parameters: 582,403.

Hardware and software:

- OS: Windows-10-10.0.26200-SP0
- CPU: AMD Ryzen 7 7730U with Radeon Graphics
- RAM: 21.8 GB
- Python: 3.11.9
- PyTorch: 2.8.0+cpu
- MediaPipe: 0.10.14
- OpenCV: 4.12.0
- CUDA available: False

| Stage | N | Mean +/- SD (s/video) |
|---|---:|---:|
| Pose extraction | 30 | 7.5099 +/- 4.5228 s |
| Feature preprocessing | 30 | 0.0065 +/- 0.0045 s |
| Recognition inference | 30 | 0.0030 +/- 0.0014 s |
| Rule-based scoring | 30 | 0.0100 +/- 0.0040 s |
| Total excluding feedback rendering | 30 | 7.5294 +/- 4.5303 s |

Feedback rendering was not measured because the current repository does not expose a standalone production feedback-rendering function. The available video writer is a quality-inspection utility rather than the final scoring-feedback renderer.

Manuscript wording should state `desktop-CPU prototype using smartphone-captured videos` or `smartphone-video-based assessment pipeline`, not `on-device`, `real-time mobile deployment`, or `fully deployed mobile system`.
