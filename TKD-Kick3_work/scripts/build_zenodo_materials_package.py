from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import zipfile
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DESKTOP = Path(r"<DESKTOP>")
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
PACKAGE_NAME = f"TKD-Kick3_Zenodo_Materials_{STAMP}"
DEST = DESKTOP / PACKAGE_NAME
ZIP_PATH = DESKTOP / f"{PACKAGE_NAME}.zip"

CLASSES = ["front", "roundhouse", "axe"]
SPLITS = ["train", "val", "test"]
TEXT_SUFFIXES = {".csv", ".json", ".md", ".txt", ".yaml", ".yml", ".cff", ".py", ".toml"}


def sanitize_text_string(text: str) -> str:
    """Remove local machine paths from text artifacts before public packaging."""
    replacements = {
        str(ROOT): "<PROJECT_ROOT>",
        str(ROOT).replace("\\", "/"): "<PROJECT_ROOT>",
        r"<PROJECT_ROOT>": "<PROJECT_ROOT>",
        "<PROJECT_ROOT>": "<PROJECT_ROOT>",
        r"<DESKTOP>": "<DESKTOP>",
        "<DESKTOP>": "<DESKTOP>",
        r"<USER_HOME>": "<USER_HOME>",
        "<USER_HOME>": "<USER_HOME>",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"E:[\\/]+PyCharm[\\/]+NewProject", "<PROJECT_ROOT>", text)
    text = re.sub(r"C:[\\/]+Users[\\/]+13261[\\/]+Desktop", "<DESKTOP>", text)
    text = re.sub(r"C:[\\/]+Users[\\/]+13261", "<USER_HOME>", text)
    return text


def sanitize_text_file(path: Path) -> None:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return
    cleaned = sanitize_text_string(text)
    if cleaned != text:
        path.write_text(cleaned, encoding="utf-8")


def sanitize_all_text_files(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file():
            sanitize_text_file(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_public_participant_template(path: Path) -> None:
    write_csv(
        path,
        [
            {
                "video_file": "<private_video_filename>",
                "sequence_path": "data/processed_pose_sequences/train/front/<sequence_id>.npz",
                "participant_id": "<participant_id_if_available>",
                "kick_label": "front",
                "attempt_id": "<attempt_id>",
                "teaching_class": "<class_id_if_available>",
            }
        ],
        ["video_file", "sequence_path", "participant_id", "kick_label", "attempt_id", "teaching_class"],
    )


def public_sequence_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    marker = "data/sequences/"
    if marker in normalized:
        return "data/processed_pose_sequences/" + normalized.split(marker, 1)[1]
    marker = "data/processed_pose_sequences/"
    if marker in normalized:
        return marker + normalized.split(marker, 1)[1]
    return sanitize_text_string(value)


def sanitize_runtime_per_video(path: Path) -> None:
    if not path.exists():
        return
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []
    if not fields:
        return
    for row in rows:
        if "sequence_path" in row:
            row["sequence_path"] = public_sequence_path(row.get("sequence_path", ""))
        if "source_video" in row:
            row["source_video"] = "<private_video_removed>"
    write_csv(path, rows, fields)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree_filtered(src: Path, dst: Path, ignore=None) -> None:
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=ignore)


def sanitize_npz(src: Path, dst: Path, split: str, label: str) -> dict[str, Any]:
    z = np.load(str(src), allow_pickle=True)
    allowed = {
        "kpts",
        "keypoints",
        "vis",
        "visibility",
        "fps",
        "bbox_xyxy",
        "cov_LA",
        "cov_RA",
        "cov_ankle_best",
        "max_missing_run_sec_LA",
        "max_missing_run_sec_RA",
        "max_missing_run_sec_best",
        "invalid_for_scoring",
        "win_start",
        "win_end",
        "win_pivot",
        "win_reason",
        "win_s",
        "win_e",
        "cov_best",
        "miss_best",
        "best_track",
        "best_score",
        "second_score",
        "size",
        "coord_mode",
        "pre_roll_sec_est",
        "win_dur_sec",
    }
    payload: dict[str, Any] = {}
    for key in z.files:
        if key in allowed:
            payload[key] = z[key]
    if "kpts" not in payload and "keypoints" not in payload:
        raise KeyError(f"No keypoints in {src}")
    if "vis" not in payload and "visibility" not in payload:
        raise KeyError(f"No visibility in {src}")
    payload["split"] = np.array(split)
    payload["class_label"] = np.array(label)
    payload["deidentified"] = np.array(True)
    payload["source_video_path_removed"] = np.array(True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **payload)

    kpts = payload.get("kpts", payload.get("keypoints"))
    vis = payload.get("vis", payload.get("visibility"))
    fps = payload.get("fps", np.array(30.0))
    return {
        "relative_path": str(dst.relative_to(DEST)).replace("\\", "/"),
        "split": split,
        "class_label": label,
        "sequence_id": src.stem,
        "frames": int(kpts.shape[0]),
        "keypoints": int(kpts.shape[1]) if kpts.ndim >= 2 else "",
        "coordinate_dims": int(kpts.shape[2]) if kpts.ndim >= 3 else "",
        "visibility_shape": "x".join(str(x) for x in vis.shape),
        "fps": float(np.asarray(fps).reshape(-1)[0]),
        "sha256": sha256_file(dst),
    }


def build_data() -> None:
    manifest_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        for label in CLASSES:
            src_dir = ROOT / "data" / "sequences" / split / label
            if not src_dir.exists():
                continue
            for src in sorted(src_dir.glob("*.npz")):
                dst = DEST / "data" / "processed_pose_sequences" / split / label / src.name
                manifest_rows.append(sanitize_npz(src, dst, split, label))

    write_csv(
        DEST / "data" / "metadata" / "dataset_manifest.csv",
        manifest_rows,
        [
            "relative_path",
            "split",
            "class_label",
            "sequence_id",
            "frames",
            "keypoints",
            "coordinate_dims",
            "visibility_shape",
            "fps",
            "sha256",
        ],
    )

    summary_rows = []
    for split in SPLITS:
        row: dict[str, Any] = {"split": split}
        total = 0
        for label in CLASSES:
            n = sum(1 for r in manifest_rows if r["split"] == split and r["class_label"] == label)
            row[label] = n
            total += n
        row["total"] = total
        summary_rows.append(row)
    row = {"split": "all"}
    total = 0
    for label in CLASSES:
        n = sum(1 for r in manifest_rows if r["class_label"] == label)
        row[label] = n
        total += n
    row["total"] = total
    summary_rows.append(row)
    write_csv(DEST / "data" / "metadata" / "split_summary.csv", summary_rows, ["split", *CLASSES, "total"])

    for src in [
        ROOT / "outputs" / "audit" / "data_inclusion_exclusion.csv",
        ROOT / "outputs" / "audit" / "data_inclusion_exclusion.md",
        ROOT / "outputs" / "audit" / "subject_leakage_report.md",
        ROOT / "outputs" / "method" / "keypoint_mapping.csv",
        ROOT / "outputs" / "method" / "keypoint_mapping.md",
        ROOT / "outputs" / "method" / "preprocessing_configuration.csv",
        ROOT / "outputs" / "method" / "preprocessing_configuration.md",
    ]:
        if src.exists():
            copy_file(src, DEST / "data" / "metadata" / src.name)

    write_public_participant_template(DEST / "data" / "metadata" / "participant_metadata_template.csv")


def build_code_and_models() -> None:
    scripts_dst = DEST / "code" / "scripts"
    scripts_dst.mkdir(parents=True, exist_ok=True)
    for src in sorted((ROOT / "scripts").glob("*.py")):
        copy_file(src, scripts_dst / src.name)
    for src in [ROOT / "README.md", ROOT / "AGENTS.md", ROOT / "EXPERIMENT_PLAN.md", ROOT / "EXPERIMENT_LOG.md", ROOT / "requirements.txt"]:
        if src.exists():
            copy_file(src, DEST / "code" / src.name)

    ckpt_src = ROOT / "models" / "checkpoints" / "sample_level"
    if ckpt_src.exists():
        copy_tree_filtered(ckpt_src, DEST / "models" / "checkpoints" / "sample_level")
    (DEST / "models" / "README_models.md").write_text(
        "\n".join(
            [
                "# Model Artifacts",
                "",
                "This folder contains trained recognition checkpoints generated by the reproduced sample-level validation experiment matrix.",
                "",
                "The third-party MediaPipe `pose_landmarker_full.task` model is not redistributed in this Zenodo package. The processed pose sequences are already included, so the recognition and scoring experiments can be reproduced without re-running raw video pose extraction.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def build_outputs() -> None:
    out_dst = DEST / "outputs"
    out_dst.mkdir(parents=True, exist_ok=True)
    for name in ["audit", "config", "method", "results", "scoring", "expert_agreement", "manuscript_revisions"]:
        copy_tree_filtered(
            ROOT / "outputs" / name,
            out_dst / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.docx", "*.mp4", "*.mov", "*.avi", "*.mkv"),
        )
    runtime_dst = out_dst / "runtime"
    runtime_dst.mkdir(parents=True, exist_ok=True)
    for src in (ROOT / "outputs" / "runtime").glob("*"):
        if src.is_file():
            copy_file(src, runtime_dst / src.name)
    sanitize_runtime_per_video(runtime_dst / "runtime_per_video.csv")
    write_public_participant_template(out_dst / "audit" / "participant_metadata_template.csv")
    figures_dst = DEST / "figures"
    copy_tree_filtered(ROOT / "outputs" / "figures", figures_dst)
    for src in [
        ROOT / "outputs" / "expert_agreement" / "bland_altman_plot.png",
        ROOT / "outputs" / "expert_agreement" / "bland_altman_plot.pdf",
    ]:
        if src.exists():
            copy_file(src, figures_dst / src.name)


def write_root_docs() -> None:
    readme = f"""# TKD-Kick3 Zenodo Materials

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

Generated on {date.today().isoformat()}.
"""
    (DEST / "README.md").write_text(readme, encoding="utf-8")

    citation = """cff-version: 1.2.0
message: "If you use these materials, please cite the Zenodo record and the associated manuscript."
title: "TKD-Kick3: De-identified pose sequences, code, and experiment outputs for smartphone-video taekwondo kick recognition and scoring"
type: dataset
authors:
  - family-names: Wang
    given-names: Zenan
    affiliation: Dankook University
  - family-names: Sun
    given-names: Shuo
    affiliation: Huzhou University
  - family-names: Liang
    given-names: Xilin
    affiliation: Zhejiang Business College
  - family-names: Chen
    given-names: Linhua
    affiliation: Dankook University
keywords:
  - taekwondo
  - action recognition
  - pose estimation
  - smartphone video
  - action quality assessment
license: CC-BY-4.0
version: "1.0.0"
date-released: "{today}"
repository-code: "TBD"
doi: "TBD"
""".format(today=date.today().isoformat())
    (DEST / "CITATION.cff").write_text(citation, encoding="utf-8")

    zenodo = {
        "title": "TKD-Kick3: De-identified pose sequences, code, and experiment outputs for smartphone-video taekwondo kick recognition and scoring",
        "upload_type": "dataset",
        "description": (
            "Public de-identified derived materials for a smartphone-video-based taekwondo kick recognition and scoring study. "
            "The package includes sanitized 13-keypoint MediaPipe pose sequences, code, trained recognition checkpoints, recognition metrics, "
            "scoring-rule audit outputs, expert-agreement analysis, runtime measurements, and manuscript-ready result tables. "
            "Raw videos, identifiable participant images, signed ethics forms, and local absolute source video paths are excluded."
        ),
        "creators": [
            {"name": "Wang, Zenan", "affiliation": "Dankook University"},
            {"name": "Sun, Shuo", "affiliation": "Huzhou University"},
            {"name": "Liang, Xilin", "affiliation": "Zhejiang Business College"},
            {"name": "Chen, Linhua", "affiliation": "Dankook University"},
        ],
        "keywords": [
            "taekwondo",
            "smartphone video",
            "pose estimation",
            "action recognition",
            "action quality assessment",
            "MediaPipe",
            "LSTM",
        ],
        "license": "cc-by-4.0",
        "access_right": "open",
        "version": "1.0.0",
        "publication_date": date.today().isoformat(),
        "notes": "Participant identifiers are unavailable; recognition results are sample-level validation results and should not be described as subject-independent.",
    }
    (DEST / ".zenodo.json").write_text(json.dumps(zenodo, ensure_ascii=False, indent=2), encoding="utf-8")
    (DEST / "zenodo_metadata_suggested.json").write_text(json.dumps(zenodo, ensure_ascii=False, indent=2), encoding="utf-8")

    licenses = DEST / "LICENSES"
    licenses.mkdir(parents=True, exist_ok=True)
    (licenses / "LICENSE_CODE_MIT.txt").write_text(
        """MIT License

Copyright (c) 2026 Zenan Wang, Shuo Sun, Xilin Liang, and Linhua Chen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
""",
        encoding="utf-8",
    )
    (licenses / "LICENSE_DATA_CC_BY_4.0.txt").write_text(
        "\n".join(
            [
                "Creative Commons Attribution 4.0 International (CC BY 4.0)",
                "",
                "Suggested license for data, result tables, figures, and documentation in this package.",
                "License deed: https://creativecommons.org/licenses/by/4.0/",
                "Legal code: https://creativecommons.org/licenses/by/4.0/legalcode",
                "",
                "Please confirm this license choice before publishing the Zenodo record.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (licenses / "LICENSE_NOTES.md").write_text(
        "# License Notes\n\nThis package uses a mixed-license recommendation: CC BY 4.0 for data/documentation/results and MIT for code. Confirm these choices before publishing.\n",
        encoding="utf-8",
    )


def write_manifest_and_zip() -> None:
    rows = []
    for path in sorted(DEST.rglob("*")):
        if path.is_file() and path.name not in {"MANIFEST.csv", "CHECKSUMS_SHA256.txt"}:
            rows.append(
                {
                    "relative_path": str(path.relative_to(DEST)).replace("\\", "/"),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    write_csv(DEST / "MANIFEST.csv", rows, ["relative_path", "bytes", "sha256"])
    with (DEST / "CHECKSUMS_SHA256.txt").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{row['sha256']}  {row['relative_path']}\n")

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(DEST.rglob("*")):
            if path.is_file():
                arcname = str(Path(PACKAGE_NAME) / path.relative_to(DEST)).replace("\\", "/")
                zf.write(path, arcname)


def validate_package() -> dict[str, Any]:
    forbidden_ext = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".docx"}
    forbidden_hits = []
    privacy_text_hits = []
    for path in DEST.rglob("*"):
        if path.is_file() and path.suffix.lower() in forbidden_ext:
            forbidden_hits.append(str(path.relative_to(DEST)).replace("\\", "/"))
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if re.search(r"E:[\\/]+PyCharm[\\/]+NewProject|C:[\\/]+Users[\\/]+13261", text):
                privacy_text_hits.append(str(path.relative_to(DEST)).replace("\\", "/"))
            private_video_matches = [
                match.group(0)
                for match in re.finditer(
                r"data[\\/]+videos[\\/]+[^,\s]+?\.(mp4|mov|avi|mkv|wmv)",
                text,
                flags=re.IGNORECASE,
                )
            ]
            if any("<" not in match and ">" not in match for match in private_video_matches):
                privacy_text_hits.append(str(path.relative_to(DEST)).replace("\\", "/"))
    npz_count = sum(1 for _ in (DEST / "data" / "processed_pose_sequences").rglob("*.npz"))
    total_bytes = sum(path.stat().st_size for path in DEST.rglob("*") if path.is_file())
    return {
        "package_folder": str(DEST),
        "zip_file": str(ZIP_PATH),
        "npz_pose_sequence_count": npz_count,
        "total_files": sum(1 for path in DEST.rglob("*") if path.is_file()),
        "total_bytes": total_bytes,
        "forbidden_public_file_hits": forbidden_hits,
        "privacy_text_hits": sorted(set(privacy_text_hits)),
    }


def main() -> None:
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)
    build_data()
    build_code_and_models()
    build_outputs()
    write_root_docs()
    sanitize_all_text_files(DEST)
    write_manifest_and_zip()
    summary = validate_package()
    (DEST / "PACKAGE_VALIDATION.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
