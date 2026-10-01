# Runtime Summary

Protocol: fixed validation runtime subset with 30 videos (10 per class) from the current cleaned sample-level split. Because participant identifiers are unavailable, this is not a subject-independent runtime subset.

Recognition model: LSTM-UNI seed 0 checkpoint `E:\martial arts\tkd\TKD-Kick3_work\models\checkpoints\sample_level\sample_lstm_uniform_seed0.pt`.
Trainable parameters: 582,403.

Hardware and software:

- OS: Windows-11-10.0.26200-SP0
- CPU: AMD Ryzen 7 7735HS with Radeon Graphics
- RAM: 15.2 GB
- Python: 3.12.8
- PyTorch: 2.14.0+cpu
- MediaPipe: 0.10.14
- OpenCV: 5.0.0
- CUDA available: False

| Stage | N | Mean +/- SD (s/video) |
|---|---:|---:|
| Pose extraction | 0 | not measured |
| Feature preprocessing | 30 | 0.0045 +/- 0.0023 s |
| Recognition inference | 30 | 0.0025 +/- 0.0008 s |
| Rule-based scoring | 30 | 0.0081 +/- 0.0026 s |
| Total excluding feedback rendering | 0 | not measured |

Feedback rendering was not measured because the current repository does not expose a standalone production feedback-rendering function. The available video writer is a quality-inspection utility rather than the final scoring-feedback renderer.

Manuscript wording should state `desktop-CPU prototype using smartphone-captured videos` or `smartphone-video-based assessment pipeline`, not `on-device`, `real-time mobile deployment`, or `fully deployed mobile system`.

Pose extraction timing note: MediaPipe task model not found; pose extraction not timed.
