from __future__ import annotations

import csv
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv311" / "Scripts" / "python.exe"
if not PY.exists():
    PY = Path(sys.executable)

OUT = ROOT / "outputs"
RESULTS = OUT / "results"
CONFIG = OUT / "config"
FIGURES = OUT / "figures"
LOGS = RESULTS / "logs"
CM_DIR = RESULTS / "confusion_matrices"
CKPT_DIR = ROOT / "models" / "checkpoints" / "sample_level"

CLASSES = ["front", "roundhouse", "axe"]


@dataclass
class Job:
    experiment_id: str
    family: str
    setting: str
    seed: int
    cmd: list[str]
    ckpt_path: Path
    do_train: bool = False


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def append_log(text: str) -> None:
    log = ROOT / "EXPERIMENT_LOG.md"
    with log.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n\n")


def run_cmd(cmd: list[str], log_path: Path) -> tuple[int, float, str]:
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.perf_counter() - start
    text = proc.stdout + proc.stderr
    log_path.write_text(text, encoding="utf-8")
    return proc.returncode, elapsed, text


def parse_eval_output(text: str) -> dict[str, object]:
    out: dict[str, object] = {}
    m = re.search(r"evaluated samples=(\d+) acc=([0-9.]+)", text)
    if m:
        out["n"] = int(m.group(1))
        out["accuracy"] = float(m.group(2))

    m = re.search(r"Macro-F1=([0-9.]+)\s+Weighted-F1=([0-9.]+)\s+BalancedAcc=([0-9.]+)", text)
    if m:
        out["macro_f1"] = float(m.group(1))
        out["weighted_f1"] = float(m.group(2))
        out["balanced_accuracy"] = float(m.group(3))

    class_rows = []
    for cls, p, r, f1, sup in re.findall(
        r"^\s*(front|roundhouse|axe)\s+precision=([0-9.]+)\s+recall=([0-9.]+)\s+f1=([0-9.]+)\s+support=(\d+)",
        text,
        flags=re.MULTILINE,
    ):
        class_rows.append(
            {
                "class": cls,
                "precision": float(p),
                "recall": float(r),
                "f1": float(f1),
                "support": int(sup),
            }
        )

    if not class_rows:
        for cls, p, r, sup in re.findall(
            r"^\s*(front|roundhouse|axe)\s+precision=([0-9.]+)\s+recall=([0-9.]+)\s+support=(\d+)",
            text,
            flags=re.MULTILINE,
        ):
            p_f = float(p)
            r_f = float(r)
            f1 = 2 * p_f * r_f / (p_f + r_f) if (p_f + r_f) else 0.0
            class_rows.append(
                {
                    "class": cls,
                    "precision": p_f,
                    "recall": r_f,
                    "f1": f1,
                    "support": int(sup),
                }
            )
    out["class_rows"] = class_rows

    if class_rows and "macro_f1" not in out:
        f1s = [float(r["f1"]) for r in class_rows]
        recalls = [float(r["recall"]) for r in class_rows]
        supports = [int(r["support"]) for r in class_rows]
        n = sum(supports)
        out["macro_f1"] = sum(f1s) / len(f1s)
        out["weighted_f1"] = sum(f * s for f, s in zip(f1s, supports)) / n if n else 0.0
        out["balanced_accuracy"] = sum(recalls) / len(recalls)

    cm = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "Confusion Matrix" not in line:
            continue
        for row_line in lines[i + 2 : i + 8]:
            parts = row_line.split()
            if parts and parts[0] in CLASSES:
                vals = []
                for token in parts[1:]:
                    try:
                        vals.append(int(token))
                    except ValueError:
                        pass
                if len(vals) == len(CLASSES):
                    cm.append(vals)
        break
    out["confusion_matrix"] = cm
    return out


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_cm(exp_id: str, cm: list[list[int]]) -> None:
    if not cm:
        return
    path = CM_DIR / f"{exp_id}.csv"
    rows = []
    for cls, vals in zip(CLASSES, cm):
        row = {"true_class": cls}
        for pred, val in zip(CLASSES, vals):
            row[pred] = val
        rows.append(row)
    write_csv(path, rows, ["true_class", *CLASSES])

    norm_rows = []
    for cls, vals in zip(CLASSES, cm):
        s = sum(vals)
        row = {"true_class": cls}
        for pred, val in zip(CLASSES, vals):
            row[pred] = (val / s) if s else 0.0
        norm_rows.append(row)
    write_csv(CM_DIR / f"{exp_id}_row_normalized.csv", norm_rows, ["true_class", *CLASSES])


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        if row.get("status") != "completed":
            continue
        grouped.setdefault((str(row["family"]), str(row["setting"])), []).append(row)

    summary = []
    for (family, setting), items in sorted(grouped.items()):
        rec: dict[str, object] = {
            "family": family,
            "setting": setting,
            "seeds": ",".join(str(x["seed"]) for x in items),
            "n_runs": len(items),
        }
        for metric in ["accuracy", "macro_f1", "weighted_f1", "balanced_accuracy"]:
            vals = [float(x[metric]) for x in items if x.get(metric) not in ("", None)]
            rec[f"{metric}_mean"] = mean(vals) if vals else ""
            rec[f"{metric}_std"] = stdev(vals) if len(vals) > 1 else 0.0 if vals else ""
            rec[f"{metric}_pct"] = mean(vals) * 100.0 if vals else ""
            rec[f"{metric}_std_pct"] = (stdev(vals) * 100.0 if len(vals) > 1 else 0.0) if vals else ""
        summary.append(rec)
    return summary


def summarize_classwise(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["family"]), str(row["setting"]), str(row["class"])), []).append(row)
    out = []
    for (family, setting, cls), items in sorted(grouped.items()):
        rec: dict[str, object] = {
            "family": family,
            "setting": setting,
            "class": cls,
            "n_runs": len(items),
        }
        for metric in ["precision", "recall", "f1"]:
            vals = [float(x[metric]) for x in items]
            rec[f"{metric}_mean"] = mean(vals)
            rec[f"{metric}_std"] = stdev(vals) if len(vals) > 1 else 0.0
            rec[f"{metric}_pct"] = mean(vals) * 100.0
            rec[f"{metric}_std_pct"] = (stdev(vals) * 100.0 if len(vals) > 1 else 0.0)
        rec["support"] = items[0].get("support", "")
        out.append(rec)
    return out


def baseline_train_jobs() -> list[Job]:
    jobs = []
    for family in ["lstm", "gcn"]:
        for align in ["uniform", "phase"]:
            for seed in [0, 1, 2]:
                ckpt = CKPT_DIR / f"sample_{family}_{align}_seed{seed}.pt"
                cmd = [
                    str(PY),
                    str(ROOT / "scripts" / "train_kick3_baselines.py"),
                    "--model",
                    family,
                    "--train_align",
                    align,
                    "--epochs",
                    "40",
                    "--batch_size",
                    "8",
                    "--lr",
                    "1e-3",
                    "--dropout",
                    "0.2",
                    "--seed",
                    str(seed),
                    "--device",
                    "cpu",
                    "--save_dir",
                    str(CKPT_DIR),
                    "--save_name",
                    ckpt.name,
                    "--seq_root",
                    str(ROOT / "data" / "sequences"),
                ]
                jobs.append(Job(f"train_{family}_{align}_seed{seed}", family.upper(), align, seed, cmd, ckpt, True))
    return jobs


def transformer_train_jobs() -> list[Job]:
    jobs = []
    for align in ["uniform", "phase"]:
        for seed in [0, 1, 2]:
            ckpt = CKPT_DIR / f"sample_transformer_{align}_seed{seed}.pt"
            cmd = [
                str(PY),
                str(ROOT / "scripts" / "train_kick3_transformer_ablation.py"),
                "--align",
                align,
                "--seed",
                str(seed),
                "--deterministic",
                "--epochs",
                "40",
                "--batch_size",
                "8",
                "--lr",
                "1e-3",
                "--label_smoothing",
                "0.05",
                "--dropout",
                "0.2",
                "--ckpt_dir",
                str(CKPT_DIR),
                "--ckpt_prefix",
                "sample_transformer",
            ]
            jobs.append(Job(f"train_transformer_{align}_seed{seed}", "Transformer", align, seed, cmd, ckpt, True))
    return jobs


def eval_jobs() -> list[Job]:
    jobs: list[Job] = []
    for family in ["lstm", "gcn"]:
        upper = family.upper()
        for seed in [0, 1, 2]:
            ckpt = CKPT_DIR / f"sample_{family}_uniform_seed{seed}.pt"
            cmd = [
                str(PY),
                str(ROOT / "scripts" / "eval_kick3_baselines.py"),
                "--ckpt",
                str(ckpt),
                "--split",
                "val",
                "--align",
                "uniform",
                "--decision",
                "argmax",
                "--device",
                "cpu",
                "--seq_root",
                str(ROOT / "data" / "sequences"),
            ]
            jobs.append(Job(f"{family}_uni_argmax_seed{seed}", upper, "UNI-Argmax", seed, cmd, ckpt))

            ckpt = CKPT_DIR / f"sample_{family}_phase_seed{seed}.pt"
            cmd = [
                str(PY),
                str(ROOT / "scripts" / "eval_kick3_baselines.py"),
                "--ckpt",
                str(ckpt),
                "--split",
                "val",
                "--align",
                "phase",
                "--decision",
                "sweep",
                "--device",
                "cpu",
                "--seq_root",
                str(ROOT / "data" / "sequences"),
            ]
            jobs.append(Job(f"{family}_pa_sweep_seed{seed}", upper, "PA-Sweep", seed, cmd, ckpt))

    for seed in [0, 1, 2]:
        ckpt = CKPT_DIR / f"sample_transformer_uniform_seed{seed}.pt"
        cmd = [
            str(PY),
            str(ROOT / "scripts" / "eval_val_confusion_v3.py"),
            "--ckpt",
            str(ckpt),
            "--split",
            "val",
            "--align",
            "uniform",
            "--decision",
            "argmax",
            "--device",
            "cpu",
        ]
        jobs.append(Job(f"transformer_uni_argmax_seed{seed}", "Transformer", "UNI-Argmax", seed, cmd, ckpt))

        ckpt = CKPT_DIR / f"sample_transformer_phase_seed{seed}.pt"
        for setting, align, decision in [
            ("PA-Sweep", "phase", "sweep"),
            ("PA-Oracle", "gt", "argmax"),
            ("PA-SingleHypothesis", "phase", "two_pass"),
        ]:
            cmd = [
                str(PY),
                str(ROOT / "scripts" / "eval_val_confusion_v3.py"),
                "--ckpt",
                str(ckpt),
                "--split",
                "val",
                "--align",
                align,
                "--decision",
                decision,
                "--device",
                "cpu",
            ]
            jobs.append(Job(f"transformer_{setting.lower().replace('-', '_')}_seed{seed}", "Transformer", setting, seed, cmd, ckpt))

    return jobs


def write_config_files() -> None:
    CONFIG.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "model": "LSTM",
            "input_dim": 52,
            "sequence_length": 96,
            "hidden_or_model_dim": 128,
            "layers": 2,
            "dropout": 0.2,
            "notes": "Bidirectional LSTM, mean pooling over time",
        },
        {
            "model": "GCN",
            "input_dim": 52,
            "sequence_length": 96,
            "hidden_or_model_dim": 64,
            "layers": 2,
            "dropout": 0.2,
            "notes": "Spatial skeleton GCN with temporal convolution",
        },
        {
            "model": "Transformer",
            "input_dim": 52,
            "sequence_length": 96,
            "hidden_or_model_dim": 128,
            "layers": 2,
            "dropout": 0.2,
            "notes": "4 heads, FFN dimension 256, sinusoidal positional encoding",
        },
    ]
    write_csv(CONFIG / "model_configuration.csv", rows, ["model", "input_dim", "sequence_length", "hidden_or_model_dim", "layers", "dropout", "notes"])
    (CONFIG / "training_configuration.yaml").write_text(
        "\n".join(
            [
                "protocol: current_cleaned_sample_level_validation",
                "participant_independent: false",
                "participant_independence_note: participant identifiers unavailable",
                "split: train=648, val=104, test_examples=13",
                "seeds: [0, 1, 2]",
                "optimizer: Adam",
                "learning_rate: 0.001",
                "epochs: 40",
                "batch_size: 8",
                "transformer_label_smoothing: 0.05",
                "baseline_label_smoothing: 0.0",
                "weight_decay: 0.0",
                "scheduler: none",
                "sampler: class-balanced WeightedRandomSampler",
                "checkpoint_selection: best validation accuracy",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    for d in [RESULTS, CONFIG, FIGURES, LOGS, CM_DIR, CKPT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    append_log(f"## Sample-level experiment run started\n\n- Time: {now_iso()}\n- Python: `{PY}`\n- Protocol: current cleaned sample-level train/validation split.")
    write_config_files()

    manifest_rows: list[dict[str, object]] = []
    train_jobs = baseline_train_jobs() + transformer_train_jobs()
    for job in train_jobs:
        log_path = LOGS / f"{job.experiment_id}.txt"
        status = "skipped_existing" if job.ckpt_path.exists() else "running"
        start_ts = now_iso()
        elapsed = 0.0
        rc = 0
        if not job.ckpt_path.exists():
            rc, elapsed, _ = run_cmd(job.cmd, log_path)
            status = "completed" if rc == 0 and job.ckpt_path.exists() else "failed"
        else:
            log_path.write_text("Skipped because checkpoint already exists.\n", encoding="utf-8")
        manifest_rows.append(
            {
                "experiment_id": job.experiment_id,
                "stage": "train",
                "model": job.family,
                "normalization": job.setting,
                "inference_strategy": "",
                "seed": job.seed,
                "split_file": "current filesystem split",
                "checkpoint_path": str(job.ckpt_path),
                "metrics_file_path": "",
                "execution_timestamp": start_ts,
                "elapsed_sec": round(elapsed, 3),
                "status": status,
                "command": " ".join(job.cmd),
            }
        )
        append_log(f"### {job.experiment_id}\n\n- Status: {status}\n- Elapsed seconds: {elapsed:.1f}\n- Log: `{log_path}`")
        if status == "failed":
            write_csv(CONFIG / "experiment_manifest.csv", manifest_rows, list(manifest_rows[0].keys()))
            return 1

    overall_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    eval_rows = eval_jobs()
    for job in eval_rows:
        log_path = LOGS / f"{job.experiment_id}.txt"
        start_ts = now_iso()
        rc, elapsed, text = run_cmd(job.cmd, log_path)
        status = "completed" if rc == 0 else "failed"
        parsed = parse_eval_output(text) if rc == 0 else {}
        metrics_file = RESULTS / "recognition_overall_by_seed.csv"
        row = {
            "experiment_id": job.experiment_id,
            "family": job.family,
            "setting": job.setting,
            "seed": job.seed,
            "n": parsed.get("n", ""),
            "accuracy": parsed.get("accuracy", ""),
            "macro_f1": parsed.get("macro_f1", ""),
            "weighted_f1": parsed.get("weighted_f1", ""),
            "balanced_accuracy": parsed.get("balanced_accuracy", ""),
            "status": status,
            "checkpoint_path": str(job.ckpt_path),
            "log_path": str(log_path),
            "command": " ".join(job.cmd),
        }
        overall_rows.append(row)
        for c in parsed.get("class_rows", []):
            class_rows.append(
                {
                    "experiment_id": job.experiment_id,
                    "family": job.family,
                    "setting": job.setting,
                    "seed": job.seed,
                    **c,
                }
            )
        save_cm(job.experiment_id, parsed.get("confusion_matrix", []))

        manifest_rows.append(
            {
                "experiment_id": job.experiment_id,
                "stage": "evaluate",
                "model": job.family,
                "normalization": job.setting,
                "inference_strategy": job.setting,
                "seed": job.seed,
                "split_file": "current filesystem val split",
                "checkpoint_path": str(job.ckpt_path),
                "metrics_file_path": str(metrics_file),
                "execution_timestamp": start_ts,
                "elapsed_sec": round(elapsed, 3),
                "status": status,
                "command": " ".join(job.cmd),
            }
        )
        append_log(f"### {job.experiment_id}\n\n- Status: {status}\n- Elapsed seconds: {elapsed:.1f}\n- Log: `{log_path}`")
        if status == "failed":
            write_csv(CONFIG / "experiment_manifest.csv", manifest_rows, list(manifest_rows[0].keys()))
            return 1

    write_csv(
        RESULTS / "recognition_overall_by_seed.csv",
        overall_rows,
        ["experiment_id", "family", "setting", "seed", "n", "accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "status", "checkpoint_path", "log_path", "command"],
    )
    write_csv(
        RESULTS / "classwise_metrics_by_seed.csv",
        class_rows,
        ["experiment_id", "family", "setting", "seed", "class", "precision", "recall", "f1", "support"],
    )
    summary = summarize(overall_rows)
    write_csv(
        RESULTS / "recognition_summary_mean_std.csv",
        summary,
        [
            "family",
            "setting",
            "seeds",
            "n_runs",
            "accuracy_mean",
            "accuracy_std",
            "accuracy_pct",
            "accuracy_std_pct",
            "macro_f1_mean",
            "macro_f1_std",
            "macro_f1_pct",
            "macro_f1_std_pct",
            "weighted_f1_mean",
            "weighted_f1_std",
            "weighted_f1_pct",
            "weighted_f1_std_pct",
            "balanced_accuracy_mean",
            "balanced_accuracy_std",
            "balanced_accuracy_pct",
            "balanced_accuracy_std_pct",
        ],
    )
    class_summary = summarize_classwise(class_rows)
    write_csv(
        RESULTS / "classwise_metrics_summary.csv",
        class_summary,
        ["family", "setting", "class", "n_runs", "precision_mean", "precision_std", "precision_pct", "precision_std_pct", "recall_mean", "recall_std", "recall_pct", "recall_std_pct", "f1_mean", "f1_std", "f1_pct", "f1_std_pct", "support"],
    )
    write_csv(CONFIG / "experiment_manifest.csv", manifest_rows, list(manifest_rows[0].keys()))

    md_lines = [
        "# Recognition Summary",
        "",
        "Protocol: current cleaned sample-level validation split. Participant identifiers are unavailable, so these results are not subject-independent.",
        "",
        "| Model | Setting | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | Balanced accuracy (%) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        def fmt_pct(metric: str) -> str:
            val = row.get(f"{metric}_pct", "")
            sd = row.get(f"{metric}_std_pct", "")
            if val in ("", None) or sd in ("", None):
                return "NA"
            return f"{float(val):.1f} +/- {float(sd):.1f}"

        md_lines.append(
            f"| {row['family']} | {row['setting']} | "
            f"{fmt_pct('accuracy')} | "
            f"{fmt_pct('macro_f1')} | "
            f"{fmt_pct('weighted_f1')} | "
            f"{fmt_pct('balanced_accuracy')} |"
        )
    (RESULTS / "recognition_summary_mean_std.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    append_log(f"## Sample-level experiment run completed\n\n- Time: {now_iso()}\n- Overall results: `{RESULTS / 'recognition_summary_mean_std.csv'}`")
    print(json.dumps({"status": "completed", "summary": str(RESULTS / "recognition_summary_mean_std.csv")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
