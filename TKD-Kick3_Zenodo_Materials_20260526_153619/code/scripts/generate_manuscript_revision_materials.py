from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
REV = OUT / "manuscript_revisions"
TABLES = REV / "tables"
FIGURES = OUT / "figures"
METHOD = OUT / "method"

CLASSES = ["front", "roundhouse", "axe"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: list[dict[str, Any]], fields: list[str], headers: list[str] | None = None) -> str:
    headers = headers or fields
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(out)


def pct(row: dict[str, str], key: str) -> str:
    return f"{float(row[key]):.1f}"


def pct_mean_sd(row: dict[str, str], metric: str) -> str:
    return f"{float(row[metric + '_pct']):.1f} +/- {float(row[metric + '_std_pct']):.1f}"


def metric_value(rows: list[dict[str, str]], metric: str) -> dict[str, str]:
    for row in rows:
        if row["metric"] == metric:
            return row
    raise KeyError(metric)


def export_method_files() -> None:
    METHOD.mkdir(parents=True, exist_ok=True)
    keypoints = [
        (0, "nose", "MediaPipe 33 landmark 0"),
        (1, "left shoulder", "MediaPipe 33 landmark 11"),
        (2, "right shoulder", "MediaPipe 33 landmark 12"),
        (3, "left elbow", "MediaPipe 33 landmark 13"),
        (4, "right elbow", "MediaPipe 33 landmark 14"),
        (5, "left wrist", "MediaPipe 33 landmark 15"),
        (6, "right wrist", "MediaPipe 33 landmark 16"),
        (7, "left hip", "MediaPipe 33 landmark 23"),
        (8, "right hip", "MediaPipe 33 landmark 24"),
        (9, "left knee", "MediaPipe 33 landmark 25"),
        (10, "right knee", "MediaPipe 33 landmark 26"),
        (11, "left ankle", "MediaPipe 33 landmark 27"),
        (12, "right ankle", "MediaPipe 33 landmark 28"),
    ]
    kp_rows = [{"index": i, "keypoint": name, "source": src} for i, name, src in keypoints]
    write_csv(METHOD / "keypoint_mapping.csv", kp_rows, ["index", "keypoint", "source"])
    (METHOD / "keypoint_mapping.md").write_text(
        "# Keypoint Mapping\n\n"
        + md_table(kp_rows, ["index", "keypoint", "source"], ["Index", "Keypoint", "Source landmark"])
        + "\n",
        encoding="utf-8",
    )

    config_rows = [
        {"item": "Pose estimator", "value": "MediaPipe PoseLandmarker, pose_landmarker_full.task", "code_source": "scripts/extract_pose_mediapipe.py"},
        {"item": "MediaPipe version", "value": "0.10.14 in the current environment", "code_source": "outputs/runtime/runtime_summary.md"},
        {"item": "Number of poses tracked", "value": "4 candidate poses, then kick-performer selection by lower-limb motion and coverage", "code_source": "scripts/extract_pose_mediapipe.py"},
        {"item": "Minimum detection confidence", "value": "0.5", "code_source": "scripts/extract_pose_mediapipe.py"},
        {"item": "Minimum presence confidence", "value": "0.5", "code_source": "scripts/extract_pose_mediapipe.py"},
        {"item": "Minimum tracking confidence", "value": "0.5", "code_source": "scripts/extract_pose_mediapipe.py"},
        {"item": "Coordinate type", "value": "2D image pixel coordinates reduced to 13 landmarks", "code_source": "scripts/extract_pose_mediapipe.py"},
        {"item": "Visibility threshold in LSTM/GCN baseline normalization", "value": "tau = 0.5", "code_source": "scripts/train_kick3_baselines.py"},
        {"item": "Visibility threshold in Transformer feature code", "value": "VTH = 0.2", "code_source": "scripts/train_kick3_transformer_ablation.py"},
        {"item": "Spatial center", "value": "hip midpoint if both hips are visible; shoulder midpoint fallback; otherwise joint mean", "code_source": "model training and evaluation scripts"},
        {"item": "Scale normalization", "value": "max(shoulder width, hip width, 1.0)", "code_source": "model training and evaluation scripts"},
        {"item": "Input feature dimension", "value": "52 per frame: 26 position values plus 26 first-order velocity values", "code_source": "model training and evaluation scripts"},
        {"item": "Sequence length", "value": "T = 96", "code_source": "training checkpoints and scripts"},
        {"item": "Uniform resampling", "value": "linear temporal resize to 96 frames", "code_source": "model training and evaluation scripts"},
        {"item": "Phase-aligned resampling", "value": "pivot-centered asymmetric window, pre = 0.45, post = 0.55", "code_source": "model training and evaluation scripts"},
        {"item": "Pivot rule", "value": "front/roundhouse: maximum attacking-ankle to ipsilateral-hip distance; axe: highest attacking ankle / minimum y", "code_source": "baseline code; Transformer code uses related candidate-class pivoting"},
        {"item": "Attacking leg handling", "value": "heuristic from lower-limb motion or ankle-hip excursion; no verified participant-level side metadata", "code_source": "scripts/train_kick3_baselines.py; scripts/train_kick3_transformer_ablation.py"},
    ]
    write_csv(METHOD / "preprocessing_configuration.csv", config_rows, ["item", "value", "code_source"])
    (METHOD / "preprocessing_configuration.md").write_text(
        "# Preprocessing Configuration\n\n"
        + md_table(config_rows, ["item", "value", "code_source"], ["Item", "Value", "Code source"])
        + "\n\n"
        + "Important manuscript note: the current codebase uses different visibility thresholds across model families. The manuscript should not state a single universal threshold unless the code is harmonized in a later run.\n",
        encoding="utf-8",
    )


def export_tables() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    summary = read_csv(OUT / "results" / "recognition_summary_mean_std.csv")
    classwise = read_csv(OUT / "results" / "classwise_metrics_summary.csv")
    runtime = read_csv(OUT / "runtime" / "runtime_summary.csv")
    expert = read_csv(OUT / "expert_agreement" / "expert_agreement_results.csv")

    table1 = [
        {"item": "Study setting", "value": "Routine first-year university physical-education classes"},
        {"item": "Participants", "value": "140 first-year students from four teaching classes"},
        {"item": "Skill background", "value": "Most participants were beginners or had only initial exposure to taekwondo"},
        {"item": "Recorded kick types", "value": "Front kick, roundhouse kick, and axe kick"},
        {"item": "Attempts per participant", "value": "Two videos per kick type, six videos per participant by protocol"},
        {"item": "Protocol-level videos", "value": "840 smartphone videos"},
        {"item": "Recording device", "value": "Smartphone camera"},
        {"item": "Recording environment", "value": "Routine indoor teaching environment under available lighting"},
        {"item": "Additional sensors", "value": "None"},
    ]
    write_csv(TABLES / "table1_acquisition_protocol.csv", table1, ["item", "value"])
    (TABLES / "table1_acquisition_protocol.md").write_text(
        "Table 1. Smartphone-video acquisition protocol\n\n" + md_table(table1, ["item", "value"], ["Item", "Description"]) + "\n",
        encoding="utf-8",
    )

    table2 = [
        {"split_or_stage": "Protocol expected", "participants": "140", "front": "280", "roundhouse": "280", "axe": "280", "total": "840", "note": "Protocol design"},
        {"split_or_stage": "Retained active sequences", "participants": "not encoded", "front": "288", "roundhouse": "190", "axe": "287", "total": "765", "note": "Cleaned pose sequences"},
        {"split_or_stage": "Current train split", "participants": "unavailable", "front": "245", "roundhouse": "159", "axe": "244", "total": "648", "note": "File-level train"},
        {"split_or_stage": "Current validation split", "participants": "unavailable", "front": "38", "roundhouse": "25", "axe": "41", "total": "104", "note": "File-level validation"},
        {"split_or_stage": "Release/example set", "participants": "unavailable", "front": "5", "roundhouse": "6", "axe": "2", "total": "13", "note": "Examples only"},
        {"split_or_stage": "Protocol minus retained", "participants": "", "front": "", "roundhouse": "", "axe": "", "total": "75", "note": "No per-reason log"},
    ]
    write_csv(TABLES / "table2_dataset_split_summary.csv", table2, ["split_or_stage", "participants", "front", "roundhouse", "axe", "total", "note"])
    (TABLES / "table2_dataset_split_summary.md").write_text(
        "Table 2. Dataset inclusion, exclusion, and current split summary\n\n"
        + md_table(table2, ["split_or_stage", "participants", "front", "roundhouse", "axe", "total", "note"], ["Stage/Split", "Participants", "Front", "Roundhouse", "Axe", "Total", "Note"])
        + "\n",
        encoding="utf-8",
    )

    table3 = [
        {"model": "BiLSTM", "input": "96 x 52", "main_dimensions": "hidden = 128, bidirectional, layers = 2", "dropout": "0.2", "parameters": "582,403", "training": "Adam, lr = 1e-3, batch = 8, epochs = 40"},
        {"model": "GCN", "input": "96 x 52", "main_dimensions": "spatial skeleton GCN, d_model = 64, layers = 2", "dropout": "0.2", "parameters": "171,203", "training": "Adam, lr = 1e-3, batch = 8, epochs = 40"},
        {"model": "Transformer", "input": "96 x 52", "main_dimensions": "d_model = 128, layers = 2, heads = 4, FFN = 256", "dropout": "0.2", "parameters": "272,387", "training": "Adam, lr = 1e-3, batch = 8, epochs = 40, label smoothing = 0.05"},
    ]
    write_csv(TABLES / "table3_model_training_configuration.csv", table3, ["model", "input", "main_dimensions", "dropout", "parameters", "training"])
    (TABLES / "table3_model_training_configuration.md").write_text(
        "Table 3. Model architecture and training configuration\n\n"
        + md_table(table3, ["model", "input", "main_dimensions", "dropout", "parameters", "training"], ["Model", "Input", "Architecture", "Dropout", "Parameters", "Training"])
        + "\n",
        encoding="utf-8",
    )

    order4 = [("LSTM", "UNI-Argmax"), ("LSTM", "PA-Sweep"), ("Transformer", "PA-Sweep"), ("Transformer", "UNI-Argmax"), ("GCN", "PA-Sweep"), ("GCN", "UNI-Argmax")]
    table4 = []
    for family, setting in order4:
        row = next(r for r in summary if r["family"] == family and r["setting"] == setting)
        table4.append(
            {
                "model": family,
                "setting": setting,
                "accuracy": pct_mean_sd(row, "accuracy"),
                "macro_f1": pct_mean_sd(row, "macro_f1"),
                "weighted_f1": pct_mean_sd(row, "weighted_f1"),
                "balanced_accuracy": pct_mean_sd(row, "balanced_accuracy"),
                "n_runs": row["n_runs"],
            }
        )
    write_csv(TABLES / "table4_backbone_comparison.csv", table4, ["model", "setting", "accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "n_runs"])
    (TABLES / "table4_backbone_comparison.md").write_text(
        "Table 4. Recognition backbone comparison under the current sample-level validation protocol\n\n"
        + md_table(table4, ["model", "setting", "accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "n_runs"], ["Model", "Configuration", "Accuracy (%)", "Macro-F1 (%)", "Weighted-F1 (%)", "Balanced Acc. (%)", "Seeds"])
        + "\n",
        encoding="utf-8",
    )

    order5 = [("Transformer", "UNI-Argmax"), ("Transformer", "PA-SingleHypothesis"), ("Transformer", "PA-Sweep"), ("Transformer", "PA-Oracle")]
    table5 = []
    for family, setting in order5:
        row = next(r for r in summary if r["family"] == family and r["setting"] == setting)
        deployable = "Yes" if setting in {"UNI-Argmax", "PA-SingleHypothesis", "PA-Sweep"} else "No, analysis upper bound"
        table5.append(
            {
                "configuration": "PA-SingleHyp." if setting == "PA-SingleHypothesis" else setting,
                "deployable": deployable,
                "accuracy": pct_mean_sd(row, "accuracy"),
                "macro_f1": pct_mean_sd(row, "macro_f1"),
                "balanced_accuracy": pct_mean_sd(row, "balanced_accuracy"),
                "interpretation": "Combined PA+sweep" if setting == "PA-Sweep" else "Controlled comparison",
            }
        )
    write_csv(TABLES / "table5_temporal_ablation.csv", table5, ["configuration", "deployable", "accuracy", "macro_f1", "balanced_accuracy", "interpretation"])
    (TABLES / "table5_temporal_ablation.md").write_text(
        "Table 5. Ablation of temporal normalization and inference strategy for the Transformer model\n\n"
        + md_table(table5, ["configuration", "deployable", "accuracy", "macro_f1", "balanced_accuracy", "interpretation"], ["Configuration", "Deployable", "Accuracy (%)", "Macro-F1 (%)", "Balanced Acc. (%)", "Interpretation"])
        + "\n",
        encoding="utf-8",
    )

    table6 = []
    for cls in CLASSES:
        row = next(r for r in classwise if r["family"] == "LSTM" and r["setting"] == "UNI-Argmax" and r["class"] == cls)
        table6.append(
            {
                "class": cls,
                "precision": f"{float(row['precision_pct']):.1f} +/- {float(row['precision_std_pct']):.1f}",
                "recall": f"{float(row['recall_pct']):.1f} +/- {float(row['recall_std_pct']):.1f}",
                "f1": f"{float(row['f1_pct']):.1f} +/- {float(row['f1_std_pct']):.1f}",
                "support": row["support"],
            }
        )
    write_csv(TABLES / "table6_classwise_final_model.csv", table6, ["class", "precision", "recall", "f1", "support"])
    (TABLES / "table6_classwise_final_model.md").write_text(
        "Table 6. Class-wise performance of the selected LSTM-UNI configuration\n\n"
        + md_table(table6, ["class", "precision", "recall", "f1", "support"], ["Class", "Precision (%)", "Recall (%)", "F1 (%)", "Support"])
        + "\n",
        encoding="utf-8",
    )

    stage_labels = {
        "pose_extraction": "Pose extraction",
        "feature_preprocessing": "Feature preprocessing",
        "recognition_inference": "Recognition inference",
        "rule_based_scoring": "Rule-based scoring",
        "total_processing": "Total excluding feedback rendering",
    }
    table7 = []
    for row in runtime:
        label = stage_labels.get(row["stage"], row["stage"])
        mean = "" if row["mean_seconds"] == "" else f"{float(row['mean_seconds']):.4f}"
        sd = "" if row["std_seconds"] == "" else f"{float(row['std_seconds']):.4f}"
        min_s = "" if row["min_seconds"] == "" else f"{float(row['min_seconds']):.4f}"
        max_s = "" if row["max_seconds"] == "" else f"{float(row['max_seconds']):.4f}"
        table7.append({"stage": label, "n": row["n"], "mean_seconds": mean, "std_seconds": sd, "min_seconds": min_s, "max_seconds": max_s})
    write_csv(TABLES / "table7_runtime.csv", table7, ["stage", "n", "mean_seconds", "std_seconds", "min_seconds", "max_seconds"])
    (TABLES / "table7_runtime.md").write_text(
        "Table 7. Runtime performance on a desktop CPU prototype\n\n"
        + md_table(table7, ["stage", "n", "mean_seconds", "std_seconds", "min_seconds", "max_seconds"], ["Stage", "N", "Mean (s/video)", "SD", "Min", "Max"])
        + "\n",
        encoding="utf-8",
    )

    table8 = [
        {"metric": "Expert score reliability", "result": f"ICC(A,3) = {float(metric_value(expert, 'ICC(A,3)')['value']):.3f}, 95% bootstrap CI [{float(metric_value(expert, 'ICC(A,3)')['ci_low']):.3f}, {float(metric_value(expert, 'ICC(A,3)')['ci_high']):.3f}]"},
        {"metric": "Class agreement vs expert majority", "result": f"{float(metric_value(expert, 'classification_accuracy_vs_expert_majority')['value']) * 100:.1f}% ({metric_value(expert, 'classification_correct')['value']}/30), kappa = {float(metric_value(expert, 'Cohen_kappa')['value']):.3f}"},
        {"metric": "Score rank correlation", "result": f"Spearman rho = {float(metric_value(expert, 'Spearman_rho')['value']):.3f}"},
        {"metric": "Score linear correlation", "result": f"Pearson r = {float(metric_value(expert, 'Pearson_r')['value']):.3f}"},
        {"metric": "Score error", "result": f"MAE = {float(metric_value(expert, 'MAE')['value']):.3f}; MSE = {float(metric_value(expert, 'MSE')['value']):.3f}"},
        {"metric": "Bland-Altman", "result": f"bias = {float(metric_value(expert, 'Bland_Altman_bias_machine_minus_expert')['value']):+.3f}; 95% LOA [{float(metric_value(expert, 'Bland_Altman_LOA_low')['value']):.3f}, {float(metric_value(expert, 'Bland_Altman_LOA_high')['value']):.3f}]"},
    ]
    write_csv(TABLES / "table8_human_ai_agreement.csv", table8, ["metric", "result"])
    (TABLES / "table8_human_ai_agreement.md").write_text(
        "Table 8. Human-AI agreement on the expert-annotated subset\n\n" + md_table(table8, ["metric", "result"], ["Metric", "Result"]) + "\n",
        encoding="utf-8",
    )


def export_workflow_figure() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    labels = [
        "140 first-year students\n840 smartphone videos",
        "MediaPipe Pose\n13 keypoints",
        "Cleaning\n765 pose sequences",
        "Current file-level split\n648 train / 104 val",
        "Recognition models\nLSTM, GCN, Transformer",
        "Validation metrics\nand ablations",
        "Rule-based scoring\n10-point feedback",
        "Expert subset\nN = 30",
    ]
    x = [0, 1, 2, 3, 4, 5, 5, 5]
    y = [0, 0, 0, 0, 0, 0, -1, -2]
    fig, ax = plt.subplots(figsize=(10.5, 3.4), dpi=300)
    ax.axis("off")
    for i, label in enumerate(labels):
        ax.text(
            x[i],
            y[i],
            label,
            ha="center",
            va="center",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#f6f8fa", edgecolor="#4d5966", linewidth=1.0),
        )
    for i in range(5):
        ax.annotate("", xy=(x[i + 1] - 0.28, y[i + 1]), xytext=(x[i] + 0.28, y[i]), arrowprops=dict(arrowstyle="->", lw=1.1, color="#333333"))
    ax.annotate("", xy=(4.98, -0.72), xytext=(4.98, -0.25), arrowprops=dict(arrowstyle="->", lw=1.1, color="#333333"))
    ax.annotate("", xy=(4.98, -1.72), xytext=(4.98, -1.25), arrowprops=dict(arrowstyle="->", lw=1.1, color="#333333"))
    ax.set_xlim(-0.55, 5.55)
    ax.set_ylim(-2.45, 0.6)
    fig.tight_layout()
    fig.savefig(FIGURES / "experimental_workflow.png", bbox_inches="tight")
    fig.savefig(FIGURES / "experimental_workflow.pdf", bbox_inches="tight")
    plt.close(fig)


def write_texts() -> None:
    REV.mkdir(parents=True, exist_ok=True)
    summary = read_csv(OUT / "results" / "recognition_summary_mean_std.csv")
    expert = read_csv(OUT / "expert_agreement" / "expert_agreement_results.csv")
    runtime = read_csv(OUT / "runtime" / "runtime_summary.csv")

    lstm_uni = next(r for r in summary if r["family"] == "LSTM" and r["setting"] == "UNI-Argmax")
    lstm_pa = next(r for r in summary if r["family"] == "LSTM" and r["setting"] == "PA-Sweep")
    trans_pa = next(r for r in summary if r["family"] == "Transformer" and r["setting"] == "PA-Sweep")
    trans_uni = next(r for r in summary if r["family"] == "Transformer" and r["setting"] == "UNI-Argmax")
    trans_oracle = next(r for r in summary if r["family"] == "Transformer" and r["setting"] == "PA-Oracle")
    total_runtime = next(r for r in runtime if r["stage"] == "total_processing")
    pose_runtime = next(r for r in runtime if r["stage"] == "pose_extraction")
    infer_runtime = next(r for r in runtime if r["stage"] == "recognition_inference")

    abstract = (
        "This study developed a smartphone-video-based pipeline for recognizing and quantitatively evaluating three fundamental taekwondo kicks: front kick, roundhouse kick, and axe kick. "
        "The TKD-Kick3 dataset was collected from 140 first-year university students in four physical-education classes, with 840 videos planned by protocol and 765 cleaned pose sequences retained after data screening. "
        "MediaPipe Pose was used to extract 13 body keypoints from monocular smartphone videos, and 52-dimensional frame features were constructed from normalized two-dimensional positions and first-order velocities. "
        "Because participant identifiers were not available in the current files, recognition experiments were evaluated on the existing cleaned sample-level validation split rather than a confirmed subject-independent test set. "
        f"Across three random seeds, the strongest recognition backbone was a BiLSTM with uniform resampling and argmax inference, achieving {pct_mean_sd(lstm_uni, 'accuracy')}% accuracy, {pct_mean_sd(lstm_uni, 'macro_f1')}% macro-F1, and {pct_mean_sd(lstm_uni, 'balanced_accuracy')}% balanced accuracy. "
        f"The Transformer with combined phase-aligned resampling and sweep inference reached {pct_mean_sd(trans_pa, 'accuracy')}% accuracy. "
        f"On 30 expert-annotated videos, the system showed preliminary agreement with expert ratings, including {float(metric_value(expert, 'classification_accuracy_vs_expert_majority')['value']) * 100:.1f}% class agreement, Spearman rho = {float(metric_value(expert, 'Spearman_rho')['value']):.3f}, and MAE = {float(metric_value(expert, 'MAE')['value']):.3f} points. "
        f"The desktop CPU prototype required {float(total_runtime['mean_seconds']):.2f} +/- {float(total_runtime['std_seconds']):.2f} s per video. "
        "These findings support feasibility while highlighting the need for participant-level metadata and independent validation before deployment claims."
    )
    (REV / "abstract_updated.md").write_text("# Updated Abstract\n\n" + abstract + "\n", encoding="utf-8")

    methods = f"""# Revised Materials and Methods

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
"""
    (REV / "materials_and_methods_revised.md").write_text(methods, encoding="utf-8")

    results = f"""# Revised Results

## Dataset Summary

The active TKD-Kick3 repository contained 765 retained three-class pose sequences from a protocol-level collection of 840 smartphone videos. The active file-level training split contained 648 sequences, and the validation split contained 104 sequences. Because participant identifiers were unavailable, subject leakage could not be excluded and the validation results are reported as sample-level validation.

## Backbone Comparison

The best recognition result was obtained by the BiLSTM with uniform temporal resampling and argmax inference. Across three seeds, this configuration achieved {pct_mean_sd(lstm_uni, 'accuracy')}% accuracy, {pct_mean_sd(lstm_uni, 'macro_f1')}% macro-F1, {pct_mean_sd(lstm_uni, 'weighted_f1')}% weighted-F1, and {pct_mean_sd(lstm_uni, 'balanced_accuracy')}% balanced accuracy. The added LSTM-PA-Sweep experiment reached {pct_mean_sd(lstm_pa, 'accuracy')}% accuracy, confirming that the phase-aligned sweep configuration did not improve the LSTM under the current split.

The Transformer with combined phase-aligned resampling and sweep inference reached {pct_mean_sd(trans_pa, 'accuracy')}% accuracy, compared with {pct_mean_sd(trans_uni, 'accuracy')}% for Transformer-UNI-Argmax. The GCN configurations were weaker, with approximately 51% accuracy. These results do not support retaining the Transformer as the primary recognition backbone on accuracy alone; the current evidence favors LSTM-UNI as the selected recognition module.

## Ablation Study

For the Transformer, UNI-Argmax achieved {pct_mean_sd(trans_uni, 'accuracy')}% accuracy. The deployable PA-Sweep configuration achieved {pct_mean_sd(trans_pa, 'accuracy')}%, while PA-SingleHypothesis reached {pct_mean_sd(next(r for r in summary if r['family'] == 'Transformer' and r['setting'] == 'PA-SingleHypothesis'), 'accuracy')}%. The PA-Oracle analysis upper bound achieved {pct_mean_sd(trans_oracle, 'accuracy')}%. Therefore, the observed improvement should be described as the effect of the combined phase-aligned resampling and sweep-inference configuration, not as the independent effect of temporal alignment alone.

## Runtime Analysis

Runtime was measured on a desktop CPU prototype using 30 validation videos, 10 per class. Pose extraction required {float(pose_runtime['mean_seconds']):.2f} +/- {float(pose_runtime['std_seconds']):.2f} s per video, whereas recognition inference required only {float(infer_runtime['mean_seconds']):.4f} +/- {float(infer_runtime['std_seconds']):.4f} s per video. The total processing time excluding feedback rendering was {float(total_runtime['mean_seconds']):.2f} +/- {float(total_runtime['std_seconds']):.2f} s per video. The pipeline should therefore be described as a desktop-CPU prototype using smartphone-captured videos, not as a validated on-device mobile deployment.

## Human-AI Agreement

The expert workbook contained 30 videos scored independently by three taekwondo experts. Expert score reliability was high, with ICC(A,3) = {float(metric_value(expert, 'ICC(A,3)')['value']):.3f}. The system classification matched the expert majority class in {float(metric_value(expert, 'classification_accuracy_vs_expert_majority')['value']) * 100:.1f}% of samples, with Cohen's kappa = {float(metric_value(expert, 'Cohen_kappa')['value']):.3f}. For total scores, Spearman rho was {float(metric_value(expert, 'Spearman_rho')['value']):.3f}, Pearson r was {float(metric_value(expert, 'Pearson_r')['value']):.3f}, MAE was {float(metric_value(expert, 'MAE')['value']):.3f} points, and Bland-Altman bias was {float(metric_value(expert, 'Bland_Altman_bias_machine_minus_expert')['value']):+.3f} points. These findings provide preliminary evidence of agreement but should not be described as expert-equivalent scoring.
"""
    (REV / "results_revised.md").write_text(results, encoding="utf-8")

    discussion = """# Revised Discussion

This study shows that a smartphone-video-based pose pipeline can provide useful recognition and interpretable scoring outputs for three fundamental taekwondo kicks. The most important experimental change is that the LSTM baseline outperformed the Transformer under the current cleaned sample-level validation protocol. The final recognition backbone should therefore be revised to LSTM-UNI unless participant-level metadata are recovered and a new subject-independent experiment changes the conclusion.

The temporal-alignment results should also be interpreted carefully. The Transformer improved when phase-aligned resampling was combined with sweep inference, but the available controls do not justify attributing the full gain to phase alignment alone. The correct phrasing is the combined phase-aligned resampling and sweep-inference configuration. In the current data, this configuration was useful for the Transformer but did not improve the LSTM.

The explainable scoring module is useful as a rule-based feedback layer, but its kinematic indices are derived from two-dimensional monocular pose estimates. The variable previously described as power should be renamed as a kinematic vigor index or impact-speed proxy. It should not be presented as biomechanical power because no force, torque, or three-dimensional dynamics were measured.

Several limitations need to be stated explicitly. First, participant identifiers are unavailable, so a subject-independent split cannot be verified and subject leakage cannot be excluded. Second, monocular 2D pose estimation is sensitive to occlusion, clothing, camera angle, distance, and multi-person scenes. Third, the runtime measurements were obtained on a desktop CPU, not on a smartphone. Fourth, the expert agreement subset was small and could not be linked to participant-independent test membership. Finally, the study evaluates recognition and scoring feasibility, not whether the system improves student learning outcomes.
"""
    (REV / "discussion_revised.md").write_text(discussion, encoding="utf-8")

    conclusion = f"""# Revised Conclusion

This work developed and audited a smartphone-video-based pipeline for recognizing and scoring front, roundhouse, and axe kicks from monocular videos. Using 13 MediaPipe pose keypoints and 52-dimensional temporal features, the best current recognition result was achieved by the LSTM-UNI configuration, with {pct_mean_sd(lstm_uni, 'accuracy')}% sample-level validation accuracy across three seeds. The expert-annotated subset showed preliminary agreement between system outputs and human ratings, but the evidence does not support claims of expert-equivalent scoring or real-time on-device deployment.

The main next step is to recover or reconstruct participant-level metadata so that leakage-free subject-independent validation can be performed. Future work should also evaluate the pipeline on additional recording environments, camera viewpoints, skill levels, and true mobile hardware.
"""
    (REV / "conclusion_revised.md").write_text(conclusion, encoding="utf-8")

    replacement = """# Replacement Map

## Global Changes

- Replace Transformer-as-final-system wording with LSTM-UNI as the selected recognition backbone.
- Replace any claim of subject-independent evaluation with sample-level validation unless participant metadata are recovered.
- Replace mobile deployment language with desktop-CPU prototype using smartphone-captured videos.
- Replace biomechanical power terminology with kinematic vigor index or impact-speed proxy.

## Abstract

- Update dataset count to 765 retained pose sequences from 840 planned videos.
- Insert the LSTM-UNI result as the main recognition result.
- Include expert agreement only as preliminary evidence.
- Remove state-of-the-art, expert-equivalent, and on-device deployment claims.

## Methods

- Add the leakage audit and participant-ID limitation.
- Add exact 13-keypoint mapping, 52-dimensional feature construction, and 96-frame sequence length.
- State MediaPipe PoseLandmarker configuration from code.
- Clarify that visibility threshold handling differs across scripts unless harmonized.
- Describe LSTM, GCN, and Transformer as compared backbones.
- Describe the scoring rules from `scripts/scoring_api.py`.

## Results

- Replace old recognition tables with Tables 4-6 from `outputs/manuscript_revisions/tables/`.
- Add Table 7 runtime results.
- Add Table 8 expert agreement results.
- Use the LSTM-UNI confusion matrix as the final selected model figure.

## Discussion

- Explain why LSTM-UNI is selected over Transformer under the current data.
- State that the Transformer phase-aligned sweep result is a combined alignment plus inference effect.
- Add limitations on participant leakage, monocular 2D sensing, occlusion, camera view, desktop CPU runtime, and small expert subset.

## Conclusion

- Avoid claims of proven teaching improvement, expert-equivalent scoring, real-time mobile deployment, or state-of-the-art classification.

## Figures and Tables

- Table 1: replace with acquisition protocol table.
- Table 2: replace with dataset inclusion and current split summary.
- Table 3: replace with model architecture/training table.
- Table 4: replace with recognition backbone comparison.
- Table 5: replace with temporal ablation.
- Table 6: replace with class-wise final model performance.
- Table 7: add runtime table.
- Table 8: add human-AI agreement table.
- Figure A: use row-normalized confusion matrix for the selected LSTM-UNI model.
- Figure B: use Bland-Altman plot from `outputs/expert_agreement/`.
- Figure C: optional workflow diagram from `outputs/figures/experimental_workflow.*`.
"""
    (REV / "replacement_map.md").write_text(replacement, encoding="utf-8")

    refs = """# Reference Updates

Add or verify references for the following source categories:

1. MediaPipe/BlazePose pose extraction: Bazarevsky et al., "BlazePose: On-device Real-time Body Pose Tracking", arXiv:2006.10204.
2. Transformer encoder baseline: Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
3. LSTM baseline: Hochreiter and Schmidhuber, "Long Short-Term Memory", Neural Computation, 1997, doi:10.1162/neco.1997.9.8.1735.
4. GCN baseline: Kipf and Welling, "Semi-Supervised Classification with Graph Convolutional Networks", ICLR 2017.
5. WT poomsae 10-point scoring structure: World Taekwondo Poomsae Competition Rules and Interpretation, or the current official WT rule document used by the authors.
6. Intraclass correlation reporting: Koo and Li, "A Guideline of Selecting and Reporting Intraclass Correlation Coefficients for Reliability Research", Journal of Chiropractic Medicine, 2016, doi:10.1016/j.jcm.2016.02.012.
7. Cohen's kappa: Cohen, "A Coefficient of Agreement for Nominal Scales", Educational and Psychological Measurement, 1960, doi:10.1177/001316446002000104.
8. Bland-Altman agreement analysis: Bland and Altman, "Statistical methods for assessing agreement between two methods of clinical measurement", Lancet, 1986, doi:10.1016/S0140-6736(86)90837-8.
"""
    (REV / "reference_updates.md").write_text(refs, encoding="utf-8")


def update_project_docs() -> None:
    ag = """# AGENTS.md

Long-term project rules for this manuscript and experiment repository:

1. Do not fabricate or manually adjust experimental numbers.
2. All reported results must come from code-generated CSV, JSON, log, or figure outputs.
3. Participant-independent evaluation cannot be claimed unless a participant metadata table is available and used for splitting.
4. All tables and figures must be traceable to exported CSV or JSON files.
5. Raw videos and identifiable participant images must not be uploaded to a public repository.
6. Do not overwrite the original manuscript file; create revised copies or exported revision materials.
7. Any unconfirmed study information must be listed as author-confirmation needed.
8. The variable historically named `power` must not be described as true biomechanical power unless force or torque data are collected.
9. Do not claim smartphone on-device or real-time mobile deployment unless measured on smartphone hardware.
"""
    (ROOT / "AGENTS.md").write_text(ag, encoding="utf-8")

    readme = """# TKD-Kick3 Sensors Manuscript Reproducibility Package

This repository supports the manuscript `Recognition and Explainable Quantitative Evaluation of Fundamental Taekwondo Kicks from Smartphone Videos`.

## Current Evidence Status

Participant identifiers are not available in the current files. Therefore, the current recognition results are sample-level validation results using the existing cleaned file split. Subject-independent evaluation must wait until participant metadata are recovered.

## Main Commands

Run the data/code audit:

```powershell
.\\.venv311\\Scripts\\python.exe scripts\\run_project_audit.py
```

Run the recognition experiment matrix:

```powershell
.\\.venv311\\Scripts\\python.exe scripts\\run_sample_level_experiments.py
```

Run scoring, expert agreement, and runtime audits:

```powershell
.\\.venv311\\Scripts\\python.exe scripts\\run_post_experiment_audits.py
```

Generate manuscript revision tables, figures, and text:

```powershell
.\\.venv311\\Scripts\\python.exe scripts\\generate_manuscript_revision_materials.py
```

## Key Outputs

- `outputs/audit/`: data availability, inclusion/exclusion, and leakage audit.
- `outputs/results/`: recognition metrics, class-wise metrics, confusion matrices.
- `outputs/scoring/`: scoring rule configuration and terminology audit.
- `outputs/expert_agreement/`: expert agreement statistics and Bland-Altman plot.
- `outputs/runtime/`: runtime measurements on the desktop CPU prototype.
- `outputs/manuscript_revisions/`: English revision text, replacement map, and manuscript-ready tables.
"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8")

    req = """numpy
torch
matplotlib
opencv-python
mediapipe==0.10.14
pandas
scikit-learn
python-docx
"""
    (ROOT / "requirements.txt").write_text(req, encoding="utf-8")


def main() -> None:
    export_method_files()
    export_tables()
    export_workflow_figure()
    write_texts()
    update_project_docs()
    log = ROOT / "EXPERIMENT_LOG.md"
    with log.open("a", encoding="utf-8") as f:
        f.write(f"## Manuscript revision materials generated at {datetime.now().isoformat(timespec='seconds')}\n\n")
        f.write("- Generated method configuration files.\n")
        f.write("- Generated manuscript-ready tables 1-8.\n")
        f.write("- Generated workflow figure and manuscript revision text files.\n\n")
    print("Manuscript revision materials generated.")


if __name__ == "__main__":
    main()
