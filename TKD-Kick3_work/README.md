# TKD-Kick3 Sensors Manuscript Reproducibility Package

This repository supports the manuscript `Recognition and Explainable Quantitative Evaluation of Fundamental Taekwondo Kicks from Smartphone Videos`.

## Current Evidence Status

Participant identifiers are not available in the current files. Therefore, the current recognition results are sample-level validation results using the existing cleaned file split. Subject-independent evaluation must wait until participant metadata are recovered.

## Main Commands

Run the data/code audit:

```powershell
.\.venv311\Scripts\python.exe scripts\run_project_audit.py
```

Run the recognition experiment matrix:

```powershell
.\.venv311\Scripts\python.exe scripts\run_sample_level_experiments.py
```

Run scoring, expert agreement, and runtime audits:

```powershell
.\.venv311\Scripts\python.exe scripts\run_post_experiment_audits.py
```

Generate manuscript revision tables, figures, and text:

```powershell
.\.venv311\Scripts\python.exe scripts\generate_manuscript_revision_materials.py
```

## Key Outputs

- `outputs/audit/`: data availability, inclusion/exclusion, and leakage audit.
- `outputs/results/`: recognition metrics, class-wise metrics, confusion matrices.
- `outputs/scoring/`: scoring rule configuration and terminology audit.
- `outputs/expert_agreement/`: expert agreement statistics and Bland-Altman plot.
- `outputs/runtime/`: runtime measurements on the desktop CPU prototype.
- `outputs/manuscript_revisions/`: English revision text, replacement map, and manuscript-ready tables.
