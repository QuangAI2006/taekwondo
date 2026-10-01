# Data and Code Audit

Audit date: 2026-05-24

Latest manuscript inspected:

- `<DESKTOP>/Sensors_Submission_Package_20260524_081544/Sensors_revised_manuscript.docx`

Project root:

- `<PROJECT_ROOT>`

## Current Available Materials

### Manuscript and submission package

- Main manuscript: `<DESKTOP>/Sensors_Submission_Package_20260524_081544/Sensors_revised_manuscript.docx`
- QA PDF: `<DESKTOP>/Sensors_Submission_Package_20260524_081544/Sensors_revised_manuscript_QA.pdf`
- Submission package tables/statistics: `<DESKTOP>/Sensors_Submission_Package_20260524_081544/tables_and_statistics/`
- Supplementary code/data package: `<DESKTOP>/Sensors_Submission_Package_20260524_081544/supplementary_code_data/`
- Source backup expert workbook: `<DESKTOP>/Sensors_Submission_Package_20260524_081544/source_backups/human_ai_agreement.xlsx`

### Data

- Raw/private videos exist under `<PROJECT_ROOT>/data/videos/`.
- Active three-class pose sequences exist under `<PROJECT_ROOT>/data/sequences/`.
- A synchronized duplicate of sequence data also exists under `<PROJECT_ROOT>/scripts/data/sequences/`; active file names and hashes match `data/sequences/`.
- Active retained three-class pose sequences: 765.
- Active sequence counts:

| Split | Front | Roundhouse | Axe | Total |
|---|---:|---:|---:|---:|
| train | 245 | 159 | 244 | 648 |
| val | 38 | 25 | 41 | 104 |
| test | 5 | 6 | 2 | 13 |
| all | 288 | 190 | 287 | 765 |

### Non-active or excluded materials

- `data/videos/_disabled/`: 57 files, all side-kick videos.
- `data/sequences/_disabled/`: 58 files, including 57 side-kick sequences.
- `data/videos/_excluded_missing_sequence/`: 1 front-kick video with no usable sequence.
- `data/sequences/_cleaning_backup_20260524_081732/`: 1 misplaced sequence backup from prior cleaning.
- `data/sequences/scores_test.csv` still includes side-kick rows; side is excluded from the active three-class study.

### Code

Key scripts found:

- Pose extraction: `scripts/extract_pose_mediapipe.py`
- Transformer training: `scripts/train_kick3_transformer_ablation.py`
- LSTM/GCN training: `scripts/train_kick3_baselines.py`
- Transformer evaluation: `scripts/eval_val_confusion_v3.py`
- LSTM/GCN evaluation: `scripts/eval_kick3_baselines.py`
- Scoring modules: `scripts/scoring_api.py`, `scripts/scoring_rules.py`
- Existing manuscript-support tools: `paper_revision/tools/`

### Models and prior results

- Existing checkpoints found under `models/checkpoints/`.
- Available checkpoints include Transformer uniform/phase seeds 0-2, LSTM uniform seeds 0-2, and GCN uniform seeds 0-2.
- No LSTM phase-aligned checkpoint was found.
- No GCN phase-aligned checkpoint was found.
- Existing validation-set summaries are available under `paper_revision/outputs/`, but they are not subject-independent results.

## Data Field Audit

Active `.npz` files contain:

- `kpts`: 765/765, shape `(T, 13, 2)`
- `vis`: 765/765, shape `(T, 13)`
- `fps`: 765/765
- extraction/quality metadata such as `src_video`, `bbox_xyxy`, `invalid_for_scoring`, `win_start`, `win_end`, and `win_reason`: 752/765
- the 13 test/release examples have only `kpts`, `vis`, `fps`, `size`, and `coord_mode`

No participant-level key was found in active `.npz` files. Scanned for participant-like fields including `participant_id`, `subject_id`, `student_id`, `person_id`, `pid`, and `sid`: no hits.

## Participant Identifier Status

Participant identifiers required for leakage-free evaluation are unavailable.

Because no reliable participant identifier is available in the current data or metadata, the project cannot yet support a formal claim of subject-independent train/validation/test evaluation. File names such as `front_0001.npz` are not sufficient to reconstruct student identity without an external mapping table.

## Split and Leakage Audit

Current split membership can be checked at the file level but not at the participant level.

File-stem overlaps were detected:

- Train/test or val/test overlap among test stems, e.g. `front_0001`, `front_0002`, `roundhouse_0001`, `axe_0002`.
- Train/val overlap among stems `roundhouse_0061`, `roundhouse_0062`, and `roundhouse_0063`.
- No identical `.npz` file hashes were detected among active duplicate stems.
- No duplicated `src_video` values were detected among the 752 active sequences that contain `src_video`.

Interpretation: the active files are not byte-level duplicates, but participant leakage cannot be ruled out. The 13-file `test` directory should be treated as an additional release/example set, not as an independent formal test set.

## Preprocessing and Feature Audit

Confirmed in code:

- 13 keypoints: nose/head proxy, shoulders, elbows, wrists, hips, knees, ankles.
- Feature dimension: normalized 2D position `13 x 2 = 26` plus first-order velocity `13 x 2 = 26`, total `D = 52`.
- Sequence length: `T = 96`.
- Spatial normalization in model code uses hip midpoint first, shoulder midpoint second, and visible-joint mean fallback.
- Scale uses shoulder width and hip width with a lower bound of 1.0 in Transformer code.
- Model training/evaluation uses visibility threshold `VTH = 0.2` in several scripts; LSTM baseline normalization uses tau `0.5`. This inconsistency must be resolved before the manuscript states a single threshold.
- Phase-aligned resize exists with asymmetric window parameters `pre = 0.45`, `post = 0.55`.
- Transformer attacking side uses ankle path length; LSTM/GCN baseline code uses maximum ankle-hip distance. This difference must be documented or harmonized.
- MediaPipe extraction uses `PoseLandmarker` with `pose_landmarker_full.task`, `num_poses = 4`, `min_pose_detection_confidence = 0.5`, `min_pose_presence_confidence = 0.5`, and `min_tracking_confidence = 0.5`.

## Recognition Experiment Capability

Already supported:

- Transformer uniform + argmax.
- Transformer phase + sweep.
- Transformer phase + oracle/ground-truth alignment.
- Transformer two-pass style inference variants in `eval_val_confusion_v3.py`.
- LSTM/GCN uniform + argmax.
- LSTM/GCN phase training is partially supported by `scripts/train_kick3_baselines.py --train_align phase`.
- LSTM/GCN phase + sweep inference is partially supported by `scripts/eval_kick3_baselines.py --align phase --decision sweep`.

Missing for the requested formal study:

- Participant-independent split cannot be generated without participant metadata.
- LSTM-PA-Sweep has not been trained/evaluated from existing checkpoints.
- GCN-PA-Sweep has not been trained/evaluated from existing checkpoints.
- Runtime measurement outputs do not yet exist in the requested `outputs/runtime/` structure.
- Current result files are validation-set based, not independent-test based.

## Expert Agreement Audit

Available:

- Expert workbook exists in the submission package source backup.
- Existing output files include per-sample and summary human-AI agreement results.
- The workbook contains 30 samples with three expert class rows, three expert total-score rows, and machine class/score rows.

Limitations:

- The workbook indexes samples as 1-30 but does not contain file names or `participant_id`.
- The manuscript currently says the expert subset was sampled from validation. Without participant IDs and sample-file mapping, it cannot be called an independent subject-level test subset.

## Main Research Risks

1. No participant IDs are currently available, so subject-independent evaluation cannot be claimed.
2. The existing 13-file test directory is too small and lacks extraction metadata; it should not be treated as a formal independent test set.
3. Current old results are validation-set results and may include subject leakage if the original split was file-level.
4. The paper still frames the pipeline as mobile-oriented; actual processing was desktop CPU on smartphone-captured videos.
5. The term `power` in the scoring module is a 2D kinematic proxy, not true biomechanical power.
6. Visibility thresholds are inconsistent across scripts and manuscript text.
7. LSTM currently outperforms Transformer in old validation results; LSTM-PA-Sweep must be run before choosing the final recognition backbone.
8. Expert agreement results are reproducible from the workbook but cannot yet be linked to an independent test subset.

## Critical Missing Materials

Formal experiments should not begin until these are provided or explicitly waived:

1. A mapping table with at least:
   - `sequence_path` or `video_file`
   - `participant_id`
   - `kick_label`
   - `attempt_id`
   - optional `class_id` or `teaching_class`
2. If available, a cleaning log with exclusion reason per original video.
3. A mapping from the 30 expert samples to source sequence/video file names.
4. Confirmation whether the 13 `test` sequences are only release examples or should be part of a designed test set.

## Next-Step Decision

Do not claim subject-independent evaluation unless participant metadata is recovered.

Author update on 2026-05-24: the participant mapping table cannot currently be found. Therefore, the executable route is a conservative sample-level validation study using the current cleaned split. The manuscript must explicitly state that participant-level leakage cannot be excluded and that subject-independent validation remains future work or requires recovery of participant metadata.

If the participant table is later recovered, the experiment plan can be upgraded to subject-independent splitting before final model training.
