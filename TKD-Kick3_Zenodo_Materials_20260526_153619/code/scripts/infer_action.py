# scripts/infer_action.py
"""
Chấm điểm động tác võ từ video — dùng chung cho đá, đấm và các động tác train thêm.

  video -> MediaPipe (2D + 3D) -> chọn người & cắt đoạn -> model nhận dạng (LSTM/GCN)
        -> luật chấm của động tác (action_scorers.py) -> điểm 10 = Chính xác 4 + Biểu hiện 6

Khi chấm 1 video, một cửa sổ tự mở hiển thị pipeline chạy trực tiếp (pipeline_viewer.py).

Ví dụ:
  python code/scripts/infer_action.py "video.mp4"                       # model đá của tác giả (LSTM ×3)
  python code/scripts/infer_action.py "video.mp4" --action hook         # chỉ định động tác cần chấm
  python code/scripts/infer_action.py "video.mp4" --ckpt "models/checkpoints/my_actions/*.pt"
  python code/scripts/infer_action.py "E:/martial arts/clips" --action hook --csv scores.csv

Luật chấm thêm: đặt file scripts/scorers_<tên>.py có dùng @register_scorer (xem ACTIONS.md),
file sẽ được nạp tự động.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import importlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
PKG_ROOT = SCRIPTS_DIR.parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

from action_classifier import ActionClassifier, default_ckpts, resolve_ckpts  # noqa: E402
from action_pose import extract_pose  # noqa: E402
from action_scorers import SCORERS, ScoreResult, action_label, canonical, score_action  # noqa: E402
from viewer_text_en import action_en, en  # noqa: E402  (chữ trong cửa sổ pipeline)

TASK_PATH = PKG_ROOT / "models" / "pose_landmarker_full.task"
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def load_plugins() -> List[str]:
    """Nạp mọi scripts/scorers_*.py để chúng tự đăng ký luật chấm."""
    loaded = []
    for p in sorted(SCRIPTS_DIR.glob("scorers_*.py")):
        importlib.import_module(p.stem)
        loaded.append(p.stem)
    return loaded


def _to_jsonable(o):
    if isinstance(o, dict):
        return {k: _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _to_jsonable(o.tolist())
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def _phase_text(r: ScoreResult) -> str:
    p = r.phases
    if not r.valid or p.get("t0") is None:
        return ""
    lab = r.phase_labels
    end = f"f{p['t_end']}" if p.get("t_end") is not None else "—"
    return f"{lab[0]} f{p['t0']} → {lab[1]} f{p['t_peak']} → {lab[2]} {end}"


def save_keyframes(video: Path, kpts: np.ndarray, r: ScoreResult, out: Path) -> None:
    """Ảnh ghép 3 frame mốc có vẽ khung xương (tay/chân ra đòn tô đỏ)."""
    from pipeline_viewer import _draw_skeleton
    p = r.phases
    frames = [(p["t0"], r.phase_labels[0]), (p["t_peak"], r.phase_labels[1]),
              (p["t_end"] if p.get("t_end") is not None else kpts.shape[0] - 1, r.phase_labels[2])]
    cap = cv2.VideoCapture(str(video))
    tiles = []
    for idx, label in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, img = cap.read()
        if not ok:
            continue
        sc = 480 / img.shape[0]
        img = cv2.resize(img, (int(img.shape[1] * sc), 480))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        _draw_skeleton(rgb, kpts[idx], sc, r.highlight, thick=3)
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.rectangle(img, (0, 0), (img.shape[1], 40), (0, 0, 0), -1)
        txt = f"frame {idx}"  # OpenCV không vẽ được chữ có dấu -> chỉ ghi số frame
        cv2.putText(img, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        tiles.append(img)
    cap.release()
    if tiles:
        cv2.imwrite(str(out), np.hstack(tiles))


def run_one(video: Path, args, clf: Optional[ActionClassifier], viewer=None) -> Dict:
    if viewer:
        cap = cv2.VideoCapture(str(video))
        # chữ trong cửa sổ pipeline: tiếng Anh (kết quả in ra terminal vẫn tiếng Việt)
        viewer.step(0, "active", f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, "
                                 f"{cap.get(cv2.CAP_PROP_FPS):.0f} fps\n{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames")
        cap.release()
        viewer.step(1, "active")

    # 1-2. pose của cả video
    pose = extract_pose(video, TASK_PATH, num_poses=4, select="limbs",
                        on_frame=viewer.on_frame if viewer else None)
    if args.save_pose:
        np.savez_compressed(args.save_pose, **pose)
    kpts, vis, fps = pose["kpts"], pose["vis"], float(pose["fps"])
    T = kpts.shape[0]
    if viewer:
        viewer.step(1, "done", f"done: {T} frames")

    # 3-4. cắt đoạn + nhận dạng
    forced = canonical(args.action) if args.action else None
    rec: Dict = {"used": False}
    model_window = None
    can_classify = clf is not None and T >= 2 and (forced is None or forced in [canonical(c) for c in clf.classes])
    if can_classify:
        probs, (s, e, mode) = clf.predict(kpts, vis, fps)
        i = int(np.argmax(probs))
        rec = {"used": True, "model": clf.name, "classes": clf.classes, "probs": probs.tolist(),
               "pred": clf.classes[i], "pred_prob": float(probs[i]), "window": [s, e], "window_mode": mode}
        model_window = (s, e)
        if viewer:
            viewer.step(2, "done", f"{pose.get('n_tracks', 0)} person(s) · frames f{s}–f{e}\n(window: {mode})")
            viewer.step(3, "done", f"{action_en(canonical(rec['pred']))}\n{rec['pred_prob'] * 100:.0f}%")
    elif viewer:
        viewer.step(2, "done", f"{pose.get('n_tracks', 0)} person(s) detected")
        why = ("skipped\n(kicks only)" if clf is None
               else f"forced: {forced}\n(not in model)" if forced else "no person")
        viewer.step(3, "skip", why)

    action = forced or rec.get("pred")
    if action is None:
        res = ScoreResult(action="?", valid=False, reason="Không phát hiện được người trong video.")
    else:
        res = score_action(action, pose, guard=args.guard, side=args.side,
                           kick_window=args.kick_window, fix_leg=args.fix_leg)
    if rec.get("used") and forced is None and rec["pred_prob"] < 0.5:
        res.warnings.append(f"Model không chắc chắn ({rec['pred_prob'] * 100:.0f}%) — có thể là động tác "
                            "model chưa được học, hoặc góc quay khác dữ liệu train.")

    # 5-6. chia pha & chấm điểm
    if viewer:
        if rec.get("used"):
            note = "" if forced is None else f"(forced to score as '{forced}')"
            viewer.show_probs([action_en(canonical(c)) for c in rec["classes"]], rec["probs"], rec["model"],
                              scored_as=action_en(canonical(action)) if action else None, note=note)
        else:
            viewer.skip_probs("Recognition skipped — this step is used for kicks only.\n" + (
                f"Scored as the given action: {action_en(forced)}" if forced else "No model."))
        if res.valid:
            p, lab = res.phases, [en(x) for x in res.phase_labels]
            end = f"f{p['t_end']}" if p.get("t_end") is not None else "—"
            viewer.step(4, "done", f"{lab[0]} f{p['t0']} → {lab[1]} f{p['t_peak']}\n→ {lab[2]} {end}")
            viewer.step(5, "done", f"{res.total:.2f} / 10\n{en(res.status.splitlines()[0]) if res.status else ''}")
        else:
            no_rule = action is not None and canonical(action) not in SCORERS
            viewer.step(4, "fail", "no scoring rule" if no_rule else "no complete\nmovement found")
            viewer.step(5, "skip", "not scored")
        viewer.show_result(kpts, fps, res, model_window)

    if args.keyframes and res.valid and res.phases.get("t0") is not None:
        save_keyframes(video, kpts, res, Path(args.keyframes))
    return {"video": str(video), "recognition": rec, "action": action,
            "action_label": action_label(action) if action else "", "result": res}


def print_result(out: Dict) -> None:
    r: ScoreResult = out["result"]
    rec = out["recognition"]
    print(f"\n=== {Path(out['video']).name} ===")
    if rec.get("used"):
        pairs = sorted(zip(rec["classes"], rec["probs"]), key=lambda x: -x[1])
        print(f"Nhận dạng ({rec['model']}): " + " | ".join(f"{c} {p * 100:.1f}%" for c, p in pairs))
    else:
        print("Nhận dạng: bỏ qua")
    side = f" — {r.side_text}" if r.side_text else ""
    print(f"Chấm theo: {out['action']} ({out['action_label']}){side}")
    if not r.valid:
        print(f"KHÔNG CHẤM ĐƯỢC: {r.reason}")
    else:
        print(f"Điểm : {r.total:.2f}/10  (Chính xác {r.acc:.2f}/4 + Biểu hiện {r.expr:.2f}/6)")
        print(f"Pha  : {_phase_text(r)}")
        for it in r.acc_items + r.expr_items:
            got = "  — " if it.get("got") is None else f"{it['got']:.2f}"
            print(f"  [{it['code']}] {it['label']:<48s} {got}/{it['maxi']:.1f}   {it.get('reason', '')}")
        if r.top_issues:
            print("Cần cải thiện:")
            for it in r.top_issues[:4]:
                print(f"  - {it['label']} (-{it['deduct']:.2f}): {it['tip']}")
    for w in r.warnings:
        print(f"  [!] {w}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Chấm điểm động tác võ từ video (nhận dạng + luật chấm)")
    ap.add_argument("input", help="file video hoặc thư mục chứa video")
    ap.add_argument("--action", default=None,
                    help="chỉ định động tác cần chấm (vd hook, front). Mặc định: dùng kết quả nhận dạng")
    ap.add_argument("--ckpt", default="", help="checkpoint nhận dạng: 'a.pt,b.pt', glob, hoặc thư mục")
    ap.add_argument("--model", choices=["lstm", "gcn"], default="lstm", help="(khi không có --ckpt) model của tác giả")
    ap.add_argument("--align", choices=["uniform", "phase"], default="uniform")
    ap.add_argument("--seeds", default="0,1,2", help="ensemble các seed của model tác giả")
    ap.add_argument("--no_model", action="store_true", help="không dùng model nhận dạng (cần --action)")
    ap.add_argument("--guard", choices=["chin", "waist"], default="chin", help="(đấm) kiểu thủ")
    ap.add_argument("--side", choices=["L", "R"], default=None, help="(đấm) tay đấm; mặc định tự nhận")
    ap.add_argument("--kick_window", choices=["ankle", "none"], default="ankle",
                    help="(đá) ankle = đoạn cắt của tác giả như predict_video.py; none = cả video 1 cú đá")
    ap.add_argument("--fix_leg", action="store_true",
                    help="(đá) tự chọn chân đá theo cổ chân nâng cao hơn (luật tác giả có thể chọn nhầm chân trụ)")
    ap.add_argument("--out", default="", help="lưu JSON (1 video)")
    ap.add_argument("--csv", default="", help="lưu bảng điểm CSV")
    ap.add_argument("--keyframes", default="", help="lưu ảnh 3 frame mốc (1 video)")
    ap.add_argument("--save_pose", default="", help="lưu pose .npz (1 video)")
    ap.add_argument("--window", choices=["auto", "on", "off"], default="auto",
                    help="cửa sổ pipeline: auto = bật khi chấm 1 video, tắt khi chấm thư mục")
    ap.add_argument("--save_dashboard", default="", help="lưu ảnh cửa sổ pipeline (.png, 1 video)")
    ap.add_argument("--device", default="cpu")
    return ap


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    load_plugins()
    if not TASK_PATH.is_file():
        sys.exit(f"Thiếu {TASK_PATH}. Tải tại: https://storage.googleapis.com/mediapipe-models/"
                 "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task")
    if args.no_model and not args.action:
        sys.exit("--no_model cần đi kèm --action")

    clf = None
    if not args.no_model:
        paths = resolve_ckpts(args.ckpt) if args.ckpt else default_ckpts(
            args.model, args.align, [int(s) for s in args.seeds.split(",") if s.strip()])
        clf = ActionClassifier(paths, device=args.device)
        print(f"[model] {clf.name}: {len(clf.classes)} động tác {clf.classes} · cắt đoạn: {clf.window_mode}")
    if args.action and canonical(args.action) not in SCORERS:
        print(f"[!] Chưa có luật chấm cho '{args.action}' — sẽ không có điểm. Có luật cho: {sorted(SCORERS)}")

    inp = Path(args.input)
    if inp.is_dir():
        videos = sorted(p for p in inp.iterdir() if p.suffix.lower() in VIDEO_EXTS)
        args.keyframes = args.save_pose = ""
    elif inp.is_file():
        videos = [inp]
    else:
        sys.exit(f"Không tìm thấy: {inp}")

    show = args.window == "on" or (args.window == "auto" and inp.is_file())
    use_viewer = show or bool(args.save_dashboard)
    if use_viewer:
        if not show:
            import matplotlib
            matplotlib.use("Agg")  # chỉ lưu ảnh, không mở cửa sổ
        from pipeline_viewer import PipelineViewer

    results, viewer = [], None
    for v in videos:
        if use_viewer:
            title = v.name + (f" — {action_en(canonical(args.action))}" if args.action else "")
            viewer = PipelineViewer(title, interactive=show)
        try:
            out = run_one(v, args, clf, viewer)
        except Exception as e:  # 1 video lỗi không làm dừng cả thư mục
            out = {"video": str(v), "recognition": {"used": False}, "action": args.action, "action_label": "",
                   "result": ScoreResult(action=args.action or "?", valid=False, reason=f"Lỗi: {e}")}
        print_result(out)
        results.append(out)
        if viewer and args.save_dashboard and len(videos) == 1:
            viewer.save(args.save_dashboard)
            print(f"[SAVED] {args.save_dashboard}")
        if viewer and show and len(videos) > 1:
            print("(Đóng cửa sổ để chấm video tiếp theo)")
            viewer.wait_close()

    if args.out and len(results) == 1:
        o = dict(results[0])
        o["result"] = {k: v for k, v in dataclasses.asdict(o["result"]).items() if k != "series"}
        Path(args.out).write_text(json.dumps(_to_jsonable(o), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[SAVED] {args.out}")
    if args.keyframes and Path(args.keyframes).is_file():
        print(f"[SAVED] {args.keyframes}")

    if args.csv:
        codes: List[str] = []
        for o in results:
            for it in o["result"].acc_items + o["result"].expr_items:
                if it["code"] not in codes:
                    codes.append(it["code"])
        with open(args.csv, "w", encoding="utf-8-sig", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["video", "pred", "pred_prob", "action_scored", "valid", "total10", "acc4", "expr6",
                         *codes, "warnings_or_reason"])
            for o in results:
                r, rec = o["result"], o["recognition"]
                got = {it["code"]: it.get("got") for it in r.acc_items + r.expr_items}
                wr.writerow([
                    Path(o["video"]).name, rec.get("pred", ""),
                    f"{rec['pred_prob']:.3f}" if rec.get("used") else "", o["action"] or "", int(r.valid),
                    *(f"{x:.2f}" if x is not None else "" for x in (r.total, r.acc, r.expr)),
                    *("" if got.get(c) is None else f"{got[c]:.2f}" for c in codes),
                    " | ".join(r.warnings) if r.valid else r.reason,
                ])
        print(f"\n[SAVED] {args.csv}")

    if viewer and show and len(videos) == 1:
        print("\nCửa sổ pipeline đang mở — đóng cửa sổ để kết thúc.")
        viewer.wait_close()


if __name__ == "__main__":
    main()
