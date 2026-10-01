from __future__ import annotations

import csv
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
CLASSES = ["front", "roundhouse", "axe"]
CLASS_MAP = {1: "front", 2: "roundhouse", 3: "axe"}
EXPERT_XLSX_CANDIDATES = [
    Path(r"<DESKTOP>/Sensors_Submission_Package_20260524_081544/source_backups/human_ai_agreement.xlsx"),
    Path(r"<DESKTOP>/0009-0007-8295-8405/Sensors/人机一致性(1).xlsx"),
]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def append_experiment_log(text: str) -> None:
    with (ROOT / "EXPERIMENT_LOG.md").open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n\n")


def fmt_mean_sd(values: list[float], unit: str = "s") -> str:
    vals = [float(v) for v in values if np.isfinite(v)]
    if not vals:
        return "not measured"
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{statistics.mean(vals):.4f} +/- {sd:.4f} {unit}"


def export_scoring_audit() -> None:
    out_dir = OUT / "scoring"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "kick_type": "front",
            "component": "accuracy",
            "code": "FR_A1",
            "item": "Chamber/lift before extension and straight kicking path",
            "max_points": 1.6,
            "implementation": "1.6 * (0.60 * map(knee_lift_vertical, 0.04, 0.12) + 0.40 * map(straightness, 0.72, 0.96))",
            "raw_metric_or_range": "knee_lift_vertical; straightness",
            "note": "Front-kick specific item.",
        },
        {
            "kick_type": "front",
            "component": "accuracy",
            "code": "FR_A2",
            "item": "Kick height and peak knee extension",
            "max_points": 1.2,
            "implementation": "1.2 * (0.55 * map(height_raw, 0.15, 0.65) + 0.45 * map(kick_knee_angle_peak, 135 deg, 175 deg))",
            "raw_metric_or_range": "height_raw; kick_knee_angle_peak",
            "note": "Uses code threshold ranges rather than prose-only manuscript weights.",
        },
        {
            "kick_type": "front",
            "component": "accuracy",
            "code": "FR_A3",
            "item": "Recovery to the initial position",
            "max_points": 0.8,
            "implementation": "0.8 * map(recovery_ratio, 0.25, 0.75)",
            "raw_metric_or_range": "recovery_ratio",
            "note": "Shared recovery logic across kicks.",
        },
        {
            "kick_type": "front",
            "component": "accuracy",
            "code": "FR_A4",
            "item": "Body stability and hand guard",
            "max_points": 0.4,
            "implementation": "0.4 * (0.65 * map(stability, 0.35, 0.80) + 0.35 * (1 - hand_guard_drop / 0.25))",
            "raw_metric_or_range": "stability; hand_guard_drop",
            "note": "Minor accuracy component.",
        },
        {
            "kick_type": "roundhouse",
            "component": "accuracy",
            "code": "RH_A1",
            "item": "Kicking-plane alignment and path straightness",
            "max_points": 1.6,
            "implementation": "1.6 * (0.70 * map(30 - roundhouse_align_dev_deg, 5, 25) + 0.30 * map(straightness, 0.70, 0.95))",
            "raw_metric_or_range": "roundhouse_align_dev_deg; straightness",
            "note": "Roundhouse-kick specific alignment item.",
        },
        {
            "kick_type": "roundhouse",
            "component": "accuracy",
            "code": "RH_A2",
            "item": "Kick height and peak knee extension",
            "max_points": 1.2,
            "implementation": "1.2 * (0.55 * map(height_raw, 0.15, 0.65) + 0.45 * map(kick_knee_angle_peak, 135 deg, 175 deg))",
            "raw_metric_or_range": "height_raw; kick_knee_angle_peak",
            "note": "Same height and extension structure as front kick.",
        },
        {
            "kick_type": "roundhouse",
            "component": "accuracy",
            "code": "RH_A3",
            "item": "Recovery to the initial position",
            "max_points": 0.8,
            "implementation": "0.8 * map(recovery_ratio, 0.25, 0.75)",
            "raw_metric_or_range": "recovery_ratio",
            "note": "Shared recovery logic across kicks.",
        },
        {
            "kick_type": "roundhouse",
            "component": "accuracy",
            "code": "RH_A4",
            "item": "Body stability and landing control",
            "max_points": 0.4,
            "implementation": "0.4 * (0.60 * map(stability, 0.35, 0.80) + 0.40 * map(landing_forward, -0.08, 0.08))",
            "raw_metric_or_range": "stability; landing_forward",
            "note": "Minor accuracy component.",
        },
        {
            "kick_type": "axe",
            "component": "accuracy",
            "code": "AX_A1",
            "item": "Leg-lift height and upward path",
            "max_points": 1.6,
            "implementation": "1.6 * (0.55 * map(height_raw, 0.15, 0.65) + 0.45 * map(straightness, 0.70, 0.95))",
            "raw_metric_or_range": "height_raw; straightness",
            "note": "Axe-kick specific upward phase item.",
        },
        {
            "kick_type": "axe",
            "component": "accuracy",
            "code": "AX_A2",
            "item": "Downward speed",
            "max_points": 1.2,
            "implementation": "1.2 * map(down_speed, 0.20, 0.90)",
            "raw_metric_or_range": "down_speed",
            "note": "Axe-kick specific downward phase item.",
        },
        {
            "kick_type": "axe",
            "component": "accuracy",
            "code": "AX_A3",
            "item": "Recovery to the initial position",
            "max_points": 0.8,
            "implementation": "0.8 * map(recovery_ratio, 0.25, 0.75)",
            "raw_metric_or_range": "recovery_ratio",
            "note": "Shared recovery logic across kicks.",
        },
        {
            "kick_type": "axe",
            "component": "accuracy",
            "code": "AX_A4",
            "item": "Body stability and hand guard",
            "max_points": 0.4,
            "implementation": "0.4 * (0.60 * map(stability, 0.35, 0.80) + 0.40 * (1 - hand_guard_drop / 0.25))",
            "raw_metric_or_range": "stability; hand_guard_drop",
            "note": "Minor accuracy component.",
        },
        {
            "kick_type": "all",
            "component": "expression",
            "code": "EX_E1",
            "item": "Speed",
            "max_points": 2.0,
            "implementation": "2.0 * map(speed_raw, 0.20, 0.75)",
            "raw_metric_or_range": "speed_raw",
            "note": "Common expression item.",
        },
        {
            "kick_type": "all",
            "component": "expression",
            "code": "EX_E2",
            "item": "Kinematic vigor index",
            "max_points": 1.6,
            "implementation": "1.6 * map(power_raw, 0.18, 0.70)",
            "raw_metric_or_range": "power_raw in active scoring_api.py is peak normalized ankle velocity; legacy scoring_rules.py used a speed-acceleration composite.",
            "note": "Do not call this biomechanical power; use kinematic vigor index or impact-speed proxy.",
        },
        {
            "kick_type": "all",
            "component": "expression",
            "code": "EX_E3",
            "item": "Height",
            "max_points": 1.4,
            "implementation": "1.4 * map(height_raw, 0.15, 0.65)",
            "raw_metric_or_range": "height_raw",
            "note": "Common expression item.",
        },
        {
            "kick_type": "all",
            "component": "expression",
            "code": "EX_E4",
            "item": "Smoothness",
            "max_points": 1.0,
            "implementation": "1.0 * map(smoothness, 0.45, 0.85)",
            "raw_metric_or_range": "smoothness",
            "note": "Common expression item based on trajectory smoothness.",
        },
    ]
    fields = [
        "kick_type",
        "component",
        "code",
        "item",
        "max_points",
        "implementation",
        "raw_metric_or_range",
        "note",
    ]
    write_csv(out_dir / "scoring_rule_configuration.csv", rows, fields)

    md = [
        "# Scoring Rule Audit",
        "",
        "The active scoring implementation is `scripts/scoring_api.py`. It implements a 10-point score composed of a 4-point accuracy component and a 6-point expressiveness component.",
        "",
        "Important terminology correction: the variable named `power_raw` in the active implementation is not a biomechanical power measurement. It is derived from two-dimensional pose kinematics and should be described as a kinematic vigor index or impact-speed proxy. The manuscript should avoid the term mechanical power unless force or torque data are collected.",
        "",
        "| Component | Points | Implementation Status |",
        "|---|---:|---|",
        "| Accuracy | 4.0 | Implemented with kick-specific items FR/RH/AX_A1-A4 |",
        "| Expressiveness | 6.0 | Implemented with common speed, kinematic vigor, height, and smoothness items |",
        "| Total | 10.0 | `total10 = accuracy4 + expression6`, clipped to [0, 10] |",
        "",
        "See `scoring_rule_configuration.csv` for item-level weights and code-derived thresholds.",
        "",
        "Known limitation: `scripts/scoring_rules.py` is a compact legacy scoring module and does not exactly match the active `scoring_api.py`. The manuscript should cite the active API rules as the source of truth.",
    ]
    (out_dir / "scoring_rule_description.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def xlsx_col_to_idx(ref: str) -> int:
    letters = "".join(ch for ch in ref if ch.isalpha())
    out = 0
    for ch in letters:
        out = out * 26 + (ord(ch.upper()) - ord("A") + 1)
    return out


def parse_excel_scalar(cell: ET.Element, shared_strings: list[str], ns: dict[str, str]) -> str:
    ctype = cell.attrib.get("t", "")
    v = cell.find("a:v", ns)
    if v is None:
        inline = cell.find("a:is", ns)
        if inline is not None:
            texts = [t.text or "" for t in inline.findall(".//a:t", ns)]
            return "".join(texts)
        return ""
    raw = v.text or ""
    if ctype == "s":
        return shared_strings[int(raw)]
    return raw


def parse_expert_xlsx(path: Path) -> list[dict[str, Any]]:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as zf:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            ss_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in ss_root.findall("a:si", ns):
                texts = [t.text or "" for t in si.findall(".//a:t", ns)]
                shared_strings.append("".join(texts))

        sheet_root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        grid: dict[int, dict[int, str]] = {}
        for row in sheet_root.findall(".//a:sheetData/a:row", ns):
            ridx = int(row.attrib["r"])
            grid.setdefault(ridx, {})
            for cell in row.findall("a:c", ns):
                cidx = xlsx_col_to_idx(cell.attrib.get("r", "A1"))
                grid[ridx][cidx] = parse_excel_scalar(cell, shared_strings, ns)

    sample_cols = []
    for col, value in sorted(grid.get(1, {}).items()):
        if col >= 3:
            try:
                sample_id = int(float(value))
            except ValueError:
                continue
            sample_cols.append((col, sample_id))
    sample_cols = sample_cols[:30]
    if len(sample_cols) != 30:
        raise ValueError(f"Expected 30 expert samples, found {len(sample_cols)} in {path}")

    blocks: dict[str, tuple[int, int]] = {}
    valid_names = {"教师1": "expert1", "教师2": "expert2", "教师3": "expert3", "机器": "machine"}
    for ridx, row in sorted(grid.items()):
        who = row.get(1, "")
        kind = row.get(2, "")
        if who in valid_names and kind == "动作名称":
            score_row = ridx + 1
            if grid.get(score_row, {}).get(2, "") != "总分":
                raise ValueError(f"Could not find score row after {who} at row {ridx}")
            blocks[valid_names[who]] = (ridx, score_row)

    if set(blocks) != {"expert1", "expert2", "expert3", "machine"}:
        raise ValueError(f"Expert sheet blocks incomplete: {sorted(blocks)}")

    rows: list[dict[str, Any]] = []
    for col, sample_id in sample_cols:
        rec: dict[str, Any] = {"sample": sample_id}
        for who, (class_row, score_row) in blocks.items():
            cls_code = int(float(grid[class_row][col]))
            score = float(grid[score_row][col])
            rec[f"{who}_class"] = CLASS_MAP[cls_code]
            rec[f"{who}_score"] = score
        rows.append(rec)
    return rows


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    den = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    return float(np.dot(x, y) / den) if den else float("nan")


def spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    return pearsonr(rankdata(np.asarray(x, dtype=float)), rankdata(np.asarray(y, dtype=float)))


def permutation_p(x: np.ndarray, y: np.ndarray, stat_fn, rng: np.random.Generator, n_perm: int = 20000) -> float:
    observed = abs(stat_fn(x, y))
    hits = 0
    y_work = np.asarray(y).copy()
    for _ in range(n_perm):
        rng.shuffle(y_work)
        if abs(stat_fn(x, y_work)) >= observed:
            hits += 1
    return float((hits + 1) / (n_perm + 1))


def bootstrap_ci(x: np.ndarray, y: np.ndarray, stat_fn, rng: np.random.Generator, n_boot: int = 10000) -> tuple[float, float]:
    vals = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(stat_fn(x[idx], y[idx]))
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def cohen_kappa(a: list[str], b: list[str]) -> float:
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca = Counter(a)
    cb = Counter(b)
    pe = sum((ca[label] / n) * (cb[label] / n) for label in labels)
    return float((po - pe) / (1 - pe)) if (1 - pe) else float("nan")


def icc_a_k(scores: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float)
    n, k = scores.shape
    grand = scores.mean()
    row_means = scores.mean(axis=1)
    col_means = scores.mean(axis=0)
    ss_rows = k * np.sum((row_means - grand) ** 2)
    ss_cols = n * np.sum((col_means - grand) ** 2)
    ss_total = np.sum((scores - grand) ** 2)
    ss_err = ss_total - ss_rows - ss_cols
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))
    return float((ms_rows - ms_err) / (ms_rows + (ms_cols - ms_err) / n))


def majority_vote(values: list[str]) -> str:
    counts = Counter(values)
    top = counts.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return values[0]
    return top[0][0]


def analyze_expert_agreement() -> None:
    out_dir = OUT / "expert_agreement"
    out_dir.mkdir(parents=True, exist_ok=True)
    xlsx = next((p for p in EXPERT_XLSX_CANDIDATES if p.exists()), None)
    if xlsx is None:
        report = [
            "# Expert Agreement Audit",
            "",
            "No expert raw score workbook was found. Formal human-AI agreement results were not recalculated.",
        ]
        (out_dir / "expert_agreement_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        return

    rng = np.random.default_rng(20260524)
    rows = parse_expert_xlsx(xlsx)
    checked_rows: list[dict[str, Any]] = []

    expert_scores = []
    expert_classes = []
    machine_scores = []
    machine_classes = []
    for row in rows:
        e_scores = [float(row["expert1_score"]), float(row["expert2_score"]), float(row["expert3_score"])]
        e_classes = [str(row["expert1_class"]), str(row["expert2_class"]), str(row["expert3_class"])]
        maj = majority_vote(e_classes)
        e_mean = float(np.mean(e_scores))
        m_score = float(row["machine_score"])
        m_class = str(row["machine_class"])
        diff = m_score - e_mean
        rec = {
            "sample": int(row["sample"]),
            "expert1_class": row["expert1_class"],
            "expert2_class": row["expert2_class"],
            "expert3_class": row["expert3_class"],
            "expert_majority_class": maj,
            "machine_class": m_class,
            "class_match": m_class == maj,
            "expert1_score": e_scores[0],
            "expert2_score": e_scores[1],
            "expert3_score": e_scores[2],
            "expert_mean_score": e_mean,
            "machine_score": m_score,
            "score_difference_machine_minus_expert": diff,
        }
        checked_rows.append(rec)
        expert_scores.append(e_scores)
        expert_classes.append(e_classes)
        machine_scores.append(m_score)
        machine_classes.append(m_class)

    expert_scores_arr = np.asarray(expert_scores, dtype=float)
    expert_mean = expert_scores_arr.mean(axis=1)
    ai_scores = np.asarray(machine_scores, dtype=float)
    expert_majority = [str(r["expert_majority_class"]) for r in checked_rows]
    diff = ai_scores - expert_mean

    class_accuracy = float(np.mean([a == b for a, b in zip(machine_classes, expert_majority)]))
    kappa = cohen_kappa(machine_classes, expert_majority)
    pearson = pearsonr(ai_scores, expert_mean)
    spearman = spearmanr(ai_scores, expert_mean)
    pearson_ci = bootstrap_ci(ai_scores, expert_mean, pearsonr, rng)
    spearman_ci = bootstrap_ci(ai_scores, expert_mean, spearmanr, rng)
    pearson_p = permutation_p(ai_scores, expert_mean, pearsonr, rng)
    spearman_p = permutation_p(ai_scores, expert_mean, spearmanr, rng)

    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff**2))
    bias = float(diff.mean())
    loa_sd = float(diff.std(ddof=1))
    loa_low = bias - 1.96 * loa_sd
    loa_high = bias + 1.96 * loa_sd
    loa_coverage = float(np.mean((diff >= loa_low) & (diff <= loa_high)))
    icc = icc_a_k(expert_scores_arr)
    icc_ci = bootstrap_ci(
        np.arange(len(expert_scores_arr)),
        np.arange(len(expert_scores_arr)),
        lambda idx_a, _idx_b: icc_a_k(expert_scores_arr[idx_a.astype(int)]),
        rng,
    )

    per_class_mae = {}
    for cls in CLASSES:
        vals = [abs(float(r["score_difference_machine_minus_expert"])) for r in checked_rows if r["expert_majority_class"] == cls]
        per_class_mae[cls] = float(np.mean(vals)) if vals else float("nan")

    fields = [
        "sample",
        "expert1_class",
        "expert2_class",
        "expert3_class",
        "expert_majority_class",
        "machine_class",
        "class_match",
        "expert1_score",
        "expert2_score",
        "expert3_score",
        "expert_mean_score",
        "machine_score",
        "score_difference_machine_minus_expert",
    ]
    write_csv(out_dir / "expert_raw_data_checked.csv", checked_rows, fields)

    metric_rows = [
        {"metric": "n", "value": len(checked_rows), "ci_low": "", "ci_high": "", "unit": "videos"},
        {"metric": "ICC(A,3)", "value": icc, "ci_low": icc_ci[0], "ci_high": icc_ci[1], "unit": ""},
        {"metric": "classification_accuracy_vs_expert_majority", "value": class_accuracy, "ci_low": "", "ci_high": "", "unit": "proportion"},
        {"metric": "classification_correct", "value": int(sum(r["class_match"] for r in checked_rows)), "ci_low": "", "ci_high": "", "unit": "videos"},
        {"metric": "Cohen_kappa", "value": kappa, "ci_low": "", "ci_high": "", "unit": ""},
        {"metric": "Spearman_rho", "value": spearman, "ci_low": spearman_ci[0], "ci_high": spearman_ci[1], "unit": f"permutation_p={spearman_p:.6f}"},
        {"metric": "Pearson_r", "value": pearson, "ci_low": pearson_ci[0], "ci_high": pearson_ci[1], "unit": f"permutation_p={pearson_p:.6f}"},
        {"metric": "MAE", "value": mae, "ci_low": "", "ci_high": "", "unit": "points"},
        {"metric": "MSE", "value": mse, "ci_low": "", "ci_high": "", "unit": "points^2"},
        {"metric": "Bland_Altman_bias_machine_minus_expert", "value": bias, "ci_low": "", "ci_high": "", "unit": "points"},
        {"metric": "Bland_Altman_LOA_low", "value": loa_low, "ci_low": "", "ci_high": "", "unit": "points"},
        {"metric": "Bland_Altman_LOA_high", "value": loa_high, "ci_low": "", "ci_high": "", "unit": "points"},
        {"metric": "Bland_Altman_coverage", "value": loa_coverage, "ci_low": "", "ci_high": "", "unit": "proportion"},
    ]
    for cls, value in per_class_mae.items():
        metric_rows.append({"metric": f"MAE_{cls}", "value": value, "ci_low": "", "ci_high": "", "unit": "points"})
    write_csv(out_dir / "expert_agreement_results.csv", metric_rows, ["metric", "value", "ci_low", "ci_high", "unit"])

    mean_pair = (ai_scores + expert_mean) / 2.0
    fig, ax = plt.subplots(figsize=(5.4, 4.2), dpi=300)
    ax.scatter(mean_pair, diff, s=36, color="#1f77b4", alpha=0.82, edgecolor="white", linewidth=0.4)
    ax.axhline(bias, color="#d62728", linewidth=1.5, label=f"Bias = {bias:+.2f}")
    ax.axhline(loa_low, color="#555555", linestyle="--", linewidth=1.2, label="95% limits of agreement")
    ax.axhline(loa_high, color="#555555", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Mean score of system and experts")
    ax.set_ylabel("System score - expert mean score")
    ax.set_title("Bland-Altman Agreement")
    ax.grid(True, color="#e0e0e0", linewidth=0.7)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "bland_altman_plot.png", bbox_inches="tight")
    fig.savefig(out_dir / "bland_altman_plot.pdf", bbox_inches="tight")
    plt.close(fig)

    report = [
        "# Expert Agreement Audit",
        "",
        f"Source workbook: `{xlsx}`.",
        "",
        f"N = {len(checked_rows)} videos. The workbook contains 30 expert-evaluated samples.",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Expert score reliability | ICC(A,3) = {icc:.3f}, 95% bootstrap CI [{icc_ci[0]:.3f}, {icc_ci[1]:.3f}] |",
        f"| Classification accuracy vs. expert majority | {class_accuracy * 100:.1f}% ({int(sum(r['class_match'] for r in checked_rows))}/{len(checked_rows)}) |",
        f"| Classification agreement | Cohen's kappa = {kappa:.3f} |",
        f"| Spearman correlation | rho = {spearman:.3f}, permutation p = {spearman_p:.4f}, 95% CI [{spearman_ci[0]:.3f}, {spearman_ci[1]:.3f}] |",
        f"| Pearson correlation | r = {pearson:.3f}, permutation p = {pearson_p:.4f}, 95% CI [{pearson_ci[0]:.3f}, {pearson_ci[1]:.3f}] |",
        f"| Mean absolute error | {mae:.3f} points |",
        f"| Mean squared error | {mse:.3f} points^2 |",
        f"| Bland-Altman bias | {bias:+.3f} points |",
        f"| Bland-Altman 95% limits of agreement | [{loa_low:.3f}, {loa_high:.3f}] |",
        f"| Samples within limits of agreement | {loa_coverage * 100:.1f}% |",
        "",
        "Per-class MAE by expert-majority class:",
        "",
        "| Class | MAE (points) |",
        "|---|---:|",
    ]
    for cls in CLASSES:
        report.append(f"| {cls} | {per_class_mae[cls]:.3f} |")
    report.extend(
        [
            "",
            "Important limitation: because participant identifiers are unavailable, this expert subset cannot currently be verified as participant-independent from the development data. The manuscript should describe it as an expert-annotated subset unless participant-level metadata are later supplied.",
        ]
    )
    (out_dir / "expert_agreement_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    summary_json = {
        "source_workbook": str(xlsx),
        "n": len(checked_rows),
        "icc_a_3": icc,
        "icc_a_3_bootstrap_ci": icc_ci,
        "classification_accuracy_vs_expert_majority": class_accuracy,
        "cohen_kappa": kappa,
        "spearman_rho": spearman,
        "spearman_permutation_p": spearman_p,
        "spearman_bootstrap_ci": spearman_ci,
        "pearson_r": pearson,
        "pearson_permutation_p": pearson_p,
        "pearson_bootstrap_ci": pearson_ci,
        "mae": mae,
        "mse": mse,
        "bland_altman_bias": bias,
        "bland_altman_loa_low": loa_low,
        "bland_altman_loa_high": loa_high,
        "bland_altman_coverage": loa_coverage,
        "per_class_mae": per_class_mae,
    }
    (out_dir / "expert_agreement_results.json").write_text(json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_shell_text(command: str) -> str:
    try:
        return subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", command],
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        ).strip()
    except Exception:
        return ""


def choose_runtime_samples(n_per_class: int = 10) -> list[Path]:
    rng = np.random.default_rng(2025)
    samples: list[Path] = []
    for cls in CLASSES:
        paths = sorted((ROOT / "data" / "sequences" / "val" / cls).glob("*.npz"))
        if len(paths) <= n_per_class:
            samples.extend(paths)
        else:
            idx = sorted(rng.choice(len(paths), size=n_per_class, replace=False).tolist())
            samples.extend([paths[i] for i in idx])
    return samples


def measure_runtime() -> None:
    out_dir = OUT / "runtime"
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str((ROOT / "scripts").resolve()))
    import eval_kick3_baselines as eb  # type: ignore
    import scoring_api  # type: ignore

    device = "cpu"
    ckpt_path = ROOT / "models" / "checkpoints" / "sample_level" / "sample_lstm_uniform_seed0.pt"
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model = eb.build_model(ckpt["model_type"], ckpt["hparams"], num_classes=len(ckpt["classes"]))
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()
    param_count = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    t_out = int(ckpt["hparams"].get("T", 96))

    samples = choose_runtime_samples(n_per_class=10)
    rows: list[dict[str, Any]] = []

    for p in samples:
        cls = next((c for c in CLASSES if c in [part.lower() for part in p.parts]), "")
        z = np.load(str(p), allow_pickle=True)
        kpts = z["kpts"].astype(np.float32)
        vis = z["vis"].astype(np.float32)
        fps = float(z["fps"]) if "fps" in z.files else 30.0
        src_video = str(z["src_video"]) if "src_video" in z.files else ""

        start = time.perf_counter()
        kpts_n = eb.normalize_kpts(kpts, vis, tau=0.5)
        kpts_a, _ = eb.resample_kpts(kpts_n, vis, t_out)
        feat = eb.build_feat_52(kpts_a)
        preprocessing_s = time.perf_counter() - start

        start = time.perf_counter()
        _ = eb.forward_one(model, feat, device=device)
        recognition_s = time.perf_counter() - start

        start = time.perf_counter()
        _ = scoring_api.score_one_video((kpts, vis, fps), cls)
        scoring_s = time.perf_counter() - start

        rows.append(
            {
                "sequence_path": str(p),
                "source_video": src_video,
                "class": cls,
                "frames_in_sequence": int(kpts.shape[0]),
                "fps": fps,
                "pose_extraction_seconds": "",
                "feature_preprocessing_seconds": preprocessing_s,
                "recognition_inference_seconds": recognition_s,
                "rule_based_scoring_seconds": scoring_s,
                "feedback_rendering_seconds": "",
                "total_processing_seconds": "",
                "notes": "Feature, recognition, and scoring timed on the existing cleaned pose sequence.",
            }
        )

    pose_note = ""
    try:
        import extract_pose_mediapipe as ep  # type: ignore

        task_path = ROOT / "models" / "pose_landmarker_full.task"
        if task_path.exists():
            for row in rows:
                src = Path(str(row["source_video"]))
                if not src.exists():
                    row["notes"] += " Source video missing; pose extraction not timed."
                    continue
                tmp = out_dir / "pose_tmp" / str(row["class"]) / (src.stem + ".npz")
                landmarker = ep.build_landmarker(task_path, num_poses=4, min_det=0.5, min_pres=0.5, min_track=0.5)
                try:
                    start = time.perf_counter()
                    ep.extract_one_video(landmarker, src, tmp, num_poses=4, skip_existing=False, verbose=False)
                    row["pose_extraction_seconds"] = time.perf_counter() - start
                finally:
                    landmarker.close()
        else:
            pose_note = "MediaPipe task model not found; pose extraction not timed."
    except Exception as exc:
        pose_note = f"Pose extraction timing failed: {type(exc).__name__}: {exc}"

    for row in rows:
        parts = [
            row["pose_extraction_seconds"],
            row["feature_preprocessing_seconds"],
            row["recognition_inference_seconds"],
            row["rule_based_scoring_seconds"],
        ]
        if all(isinstance(x, (int, float)) and np.isfinite(float(x)) for x in parts):
            row["total_processing_seconds"] = float(sum(float(x) for x in parts))
            row["notes"] = "Pose extraction, preprocessing, recognition, and scoring timed; feedback rendering not available as a standalone callable stage."
        else:
            row["total_processing_seconds"] = ""

    fields = [
        "sequence_path",
        "source_video",
        "class",
        "frames_in_sequence",
        "fps",
        "pose_extraction_seconds",
        "feature_preprocessing_seconds",
        "recognition_inference_seconds",
        "rule_based_scoring_seconds",
        "feedback_rendering_seconds",
        "total_processing_seconds",
        "notes",
    ]
    write_csv(out_dir / "runtime_per_video.csv", rows, fields)

    summary_rows = []
    for field in [
        "pose_extraction_seconds",
        "feature_preprocessing_seconds",
        "recognition_inference_seconds",
        "rule_based_scoring_seconds",
        "total_processing_seconds",
    ]:
        vals = [float(r[field]) for r in rows if isinstance(r[field], (int, float)) and np.isfinite(float(r[field]))]
        summary_rows.append(
            {
                "stage": field.replace("_seconds", ""),
                "n": len(vals),
                "mean_seconds": statistics.mean(vals) if vals else "",
                "std_seconds": statistics.stdev(vals) if len(vals) > 1 else 0.0 if vals else "",
                "min_seconds": min(vals) if vals else "",
                "max_seconds": max(vals) if vals else "",
            }
        )
    write_csv(out_dir / "runtime_summary.csv", summary_rows, ["stage", "n", "mean_seconds", "std_seconds", "min_seconds", "max_seconds"])

    cpu = safe_shell_text("(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)")
    ram = safe_shell_text("[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)")
    try:
        import cv2  # type: ignore

        cv2_version = cv2.__version__
    except Exception:
        cv2_version = "unknown"
    try:
        import mediapipe as mp  # type: ignore

        mp_version = mp.__version__
    except Exception:
        mp_version = "unknown"

    summary = {r["stage"]: r for r in summary_rows}
    md = [
        "# Runtime Summary",
        "",
        "Protocol: fixed validation runtime subset with 30 videos (10 per class) from the current cleaned sample-level split. Because participant identifiers are unavailable, this is not a subject-independent runtime subset.",
        "",
        f"Recognition model: LSTM-UNI seed 0 checkpoint `{ckpt_path}`.",
        f"Trainable parameters: {param_count:,}.",
        "",
        "Hardware and software:",
        "",
        f"- OS: {platform.platform()}",
        f"- CPU: {cpu or platform.processor() or 'not reported'}",
        f"- RAM: {ram + ' GB' if ram else 'not reported'}",
        f"- Python: {platform.python_version()}",
        f"- PyTorch: {torch.__version__}",
        f"- MediaPipe: {mp_version}",
        f"- OpenCV: {cv2_version}",
        f"- CUDA available: {torch.cuda.is_available()}",
        "",
        "| Stage | N | Mean +/- SD (s/video) |",
        "|---|---:|---:|",
    ]
    for field, label in [
        ("pose_extraction", "Pose extraction"),
        ("feature_preprocessing", "Feature preprocessing"),
        ("recognition_inference", "Recognition inference"),
        ("rule_based_scoring", "Rule-based scoring"),
        ("total_processing", "Total excluding feedback rendering"),
    ]:
        row = summary.get(field, {})
        vals = [float(r[field + "_seconds"]) for r in rows if isinstance(r[field + "_seconds"], (int, float)) and np.isfinite(float(r[field + "_seconds"]))]
        md.append(f"| {label} | {row.get('n', 0)} | {fmt_mean_sd(vals)} |")
    md.extend(
        [
            "",
            "Feedback rendering was not measured because the current repository does not expose a standalone production feedback-rendering function. The available video writer is a quality-inspection utility rather than the final scoring-feedback renderer.",
            "",
            "Manuscript wording should state `desktop-CPU prototype using smartphone-captured videos` or `smartphone-video-based assessment pipeline`, not `on-device`, `real-time mobile deployment`, or `fully deployed mobile system`.",
        ]
    )
    if pose_note:
        md.extend(["", f"Pose extraction timing note: {pose_note}"])
    (out_dir / "runtime_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    export_scoring_audit()
    analyze_expert_agreement()
    measure_runtime()
    append_experiment_log(
        "\n".join(
            [
                f"## Post-experiment audits completed at {now_iso()}",
                "",
                "- Exported scoring rule audit to `outputs/scoring/`.",
                "- Recalculated expert agreement from the source workbook to `outputs/expert_agreement/`.",
                "- Measured runtime on the current desktop CPU prototype to `outputs/runtime/`.",
            ]
        )
    )
    print("Post-experiment audits completed.")


if __name__ == "__main__":
    main()
