# Experiment Plan

This plan records the next steps for making the Sensors manuscript scientifically auditable.

Author update on 2026-05-24: the participant metadata table cannot currently be found. Therefore, the default executable route is now the conservative fallback route: sample-level validation on the current cleaned split, with no claim of subject-independent evaluation.

## Gate 1: Metadata Completion

Input required:

- A mapping from each video or sequence to `participant_id`.
- The corresponding `kick_label` and `attempt_id`.
- If possible, teaching class and cleaning/exclusion reason.
- A mapping from the 30 expert-agreement samples to file names.

Completion standard:

- Every active sequence used for recognition has a valid participant identifier.
- No active sample has an unknown or ambiguous class label.

Current status:

- Blocked. Participant identifiers are unavailable.
- Do not write or imply subject-independent train/validation/test evaluation.

Fallback if metadata remains unavailable:

- Keep the current cleaned `train` and `val` split as a sample-level development/validation protocol.
- Treat the 13 `test` sequences as release/example files only, unless the author provides evidence that they form a designed independent test set.
- Run missing experiments on the same current validation protocol for internal consistency.
- Add a clear limitation: participant-level overlap cannot be ruled out.

## Gate 2: Subject-Independent Split

Action:

- Create a participant-level split with seed 2025.
- Target ratios: 70% train, 15% validation, 15% test.
- Keep all sequences from the same participant in one split.
- Check class distribution after splitting.

Outputs:

- `outputs/splits/subject_independent_split_seed2025.csv`
- `outputs/splits/split_summary.csv`
- `outputs/splits/split_summary.md`
- Updated leakage report.

Completion standard:

- Train, validation, and test contain disjoint participant sets.
- Each split has front, roundhouse, and axe samples.

Fallback output when metadata is unavailable:

- `outputs/splits/current_sample_level_split_summary.csv`
- `outputs/splits/current_sample_level_split_summary.md`
- `outputs/audit/subject_leakage_report.md` remains the formal caveat.

## Gate 3: Preprocessing and Feature Harmonization

Action:

- Confirm or harmonize visibility threshold usage.
- Confirm 13-keypoint mapping.
- Confirm `D = 52` feature construction.
- Confirm phase-aligned resize parameters.
- Decide whether attacking-leg inference should be unified across Transformer and baseline code.

Outputs:

- `outputs/method/keypoint_mapping.md`
- `outputs/method/preprocessing_configuration.md`
- `outputs/method/preprocessing_configuration.csv`

Completion standard:

- Manuscript formulas and text match code behavior.

## Gate 4: Formal Recognition Experiments

Preferred route: run only after Gates 1-3 pass.

Fallback route: if participant metadata remains unavailable, run the same matrix on the current cleaned sample-level validation split and label every table as validation-set/sample-level results.

Required experiments:

- LSTM-UNI
- GCN-UNI
- Transformer-UNI
- Transformer-PA-Sweep
- LSTM-PA-Sweep

Optional if stable and affordable:

- GCN-PA-Sweep

Transformer ablation:

- UNI-Argmax
- PA-Sweep
- PA-Oracle
- PA-SingleHypothesis or two-pass variant if code behavior is reliable.

Seeds:

- 0, 1, 2.

Outputs:

- `outputs/results/recognition_overall_by_seed.csv`
- `outputs/results/recognition_summary_mean_std.csv`
- `outputs/results/classwise_metrics_by_seed.csv`
- `outputs/results/classwise_metrics_summary.csv`
- `outputs/results/confusion_matrices/`
- `outputs/figures/confusion_matrix_final_model.png`
- `outputs/figures/confusion_matrix_final_model.pdf`

Completion standard:

- Metrics are produced from code and test-set predictions, not manually edited.

## Gate 5: Runtime and Scoring Audit

Action:

- Measure runtime on the current desktop CPU.
- Audit scoring weights and thresholds.
- Rename or describe `power` as a kinematic proxy, not biomechanical power.

Outputs:

- `outputs/runtime/runtime_per_video.csv`
- `outputs/runtime/runtime_summary.csv`
- `outputs/runtime/runtime_summary.md`
- `outputs/scoring/scoring_rule_configuration.csv`
- `outputs/scoring/scoring_rule_description.md`

Completion standard:

- Manuscript avoids unsupported claims about smartphone on-device or real-time mobile deployment.

## Gate 6: Expert Agreement

Action:

- If expert sample file mapping is provided, verify whether samples are from the independent test set.
- Recompute ICC(A,3), kappa, correlations, MAE, MSE, and Bland-Altman statistics.

Outputs:

- `outputs/expert_agreement/expert_raw_data_checked.csv`
- `outputs/expert_agreement/expert_agreement_results.csv`
- `outputs/expert_agreement/bland_altman_plot.png`
- `outputs/expert_agreement/bland_altman_plot.pdf`
- `outputs/expert_agreement/expert_agreement_report.md`

Completion standard:

- No expert statistic is reported unless it is reproducible from available raw data.

## Gate 7: Manuscript Revision Text

Action:

- Generate manuscript replacement text after final results are available.
- Do not overwrite the original Word manuscript.

Outputs:

- `outputs/manuscript_revisions/materials_and_methods_revised.md`
- `outputs/manuscript_revisions/results_revised.md`
- `outputs/manuscript_revisions/discussion_revised.md`
- `outputs/manuscript_revisions/conclusion_revised.md`
- `outputs/manuscript_revisions/abstract_updated.md`
- `outputs/manuscript_revisions/replacement_map.md`

Completion standard:

- Every number in the text traces back to an output CSV or log.
