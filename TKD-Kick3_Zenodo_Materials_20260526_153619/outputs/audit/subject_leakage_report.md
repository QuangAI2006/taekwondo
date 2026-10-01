# Subject Leakage Report

Participant identifiers required for leakage-free evaluation are unavailable.

The current repository contains class labels through directory structure and file names, but it does not contain `participant_id`, `subject_id`, `student_id`, or an equivalent field in the active `.npz` files. No separate participant metadata table was found during the local scan.

## What Could Be Checked

- File-level active counts by split and class.
- Duplicate file stems across splits.
- Duplicate `src_video` values where `src_video` exists.
- Identical `.npz` hashes among active files.

## Findings

- Same file stems occur across splits. Examples include several `test` examples and `roundhouse_0061` to `roundhouse_0063` between train and validation.
- No identical `.npz` hashes were detected among duplicate stems.
- No duplicated `src_video` values were detected among the 752 active sequences that contain `src_video`.
- The 13 `test` examples lack `src_video` and quality metadata.

## Interpretation

These checks do not prove file-level duplication, but they also cannot rule out subject leakage. A subject-independent split cannot be created from the current files alone.

## Required Author File

Please provide a metadata table with columns such as:

| video_file | sequence_path | participant_id | kick_label | attempt_id | teaching_class |
|---|---|---|---|---|

Once this table is available, the next step is to generate `outputs/splits/subject_independent_split_seed2025.csv` and then rerun the planned experiment matrix.

