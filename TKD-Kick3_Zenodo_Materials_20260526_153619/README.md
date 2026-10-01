# TKD-Kick3 Zenodo Materials

This package contains public, de-identified derived materials for the study:

**Recognition and Explainable Quantitative Evaluation of Fundamental Taekwondo Kicks from Smartphone Videos**

## Contents

- `data/processed_pose_sequences/`: sanitized MediaPipe-derived pose sequences for front kick, roundhouse kick, and axe kick.
- `data/metadata/`: keypoint mapping, split summary, inclusion/exclusion audit, and leakage-risk notes.
- `code/`: Python scripts and reproducibility notes used for preprocessing, recognition, scoring, evaluation, and manuscript-result generation.
- `models/checkpoints/sample_level/`: trained recognition checkpoints from the reproduced sample-level experiment matrix.
- `outputs/`: code-generated audit reports, recognition metrics, confusion matrices, scoring-rule audit, expert-agreement analysis, runtime results, and manuscript revision tables.
- `figures/`: non-identifying result figures suitable for reuse in the manuscript or supplement.

## What Is Not Included

This public package intentionally excludes:

- raw smartphone videos;
- unblurred or identifiable participant images;
- Word/PDF manuscript files containing visual feedback screenshots;
- signed/stamped ethics approval scans;
- local absolute source video paths inside `.npz` files.

The `.npz` pose files in this package were regenerated from the active processed sequences with local `src_video` paths removed.

## Dataset Summary

Protocol-level acquisition planned 840 videos from 140 first-year students in four teaching classes. The current public derived package contains 765 cleaned three-class pose sequences:

| Split | Front | Roundhouse | Axe | Total |
|---|---:|---:|---:|---:|
| train | 245 | 159 | 244 | 648 |
| val | 38 | 25 | 41 | 104 |
| test/release examples | 5 | 6 | 2 | 13 |
| all | 288 | 190 | 287 | 765 |

Participant identifiers were not available in the current files. Therefore, the included split is a file-level/sample-level split and must not be described as participant-independent.

## Pose Sequence Format

Each `.npz` file contains:

- `kpts` or `keypoints`: shape `(T, 13, 2)`, 2D image-coordinate keypoints;
- `vis` or `visibility`: shape `(T, 13)`, landmark visibility;
- `fps`: video frame rate used for timing;
- optional quality/window metadata from the extraction pipeline;
- `split`, `class_label`, `deidentified`, and `source_video_path_removed`.

The 13 keypoints are listed in `data/metadata/keypoint_mapping.csv`.

## Reproduction

Create a Python environment with the packages in `code/requirements.txt`, then run from the package root:

```powershell
python code/scripts/run_sample_level_experiments.py
python code/scripts/run_post_experiment_audits.py
python code/scripts/generate_manuscript_revision_materials.py
```

The scripts in the original project expected paths relative to `<PROJECT_ROOT>`; if running outside that workspace, update path constants or copy the package contents into an equivalent project root before executing.

## Main Reproduced Finding

Under the current cleaned sample-level validation protocol, the strongest recognition configuration was LSTM-UNI:

- Accuracy: 75.3 +/- 2.3%;
- Macro-F1: 74.3 +/- 2.3%;
- Balanced accuracy: 74.5 +/- 2.3%.

These are not subject-independent results because participant identifiers are unavailable.

## License

Suggested licensing:

- data, tables, figures, and documentation: Creative Commons Attribution 4.0 International (CC BY 4.0);
- code: MIT License.

Please confirm the license choices before publishing the Zenodo record.

## Citation

Use `CITATION.cff` for citation metadata. A Zenodo DOI should be added after the draft DOI is reserved or after publication.

Generated on 2026-05-26.
