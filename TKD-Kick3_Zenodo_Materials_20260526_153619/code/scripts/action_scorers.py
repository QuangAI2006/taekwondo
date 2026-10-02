# scripts/action_scorers.py
"""
Bảng tra "động tác -> luật chấm". Mỗi luật nhận pose của cả video và trả về ScoreResult cùng định dạng,
để infer_action.py / pipeline_viewer.py dùng chung cho mọi động tác.

Có sẵn:
  front, roundhouse, axe, side  -> luật đá của tác giả (scoring_api.py), trên đoạn cắt theo cổ chân
  hook, straight                -> luật đấm (punch_scoring.py), trên cả video + tọa độ 3D

THÊM ĐỘNG TÁC MỚI (vd "elbow" = đánh chỏ):
    from action_scorers import register_scorer, ScoreResult

    @register_scorer("elbow", "Đánh chỏ")
    def score_elbow(pose, **opts) -> ScoreResult:
        kpts, vis, world, fps = pose["kpts"], pose["vis"], pose["world"], float(pose["fps"])
        ...  # đo chỉ số, tạo acc_items / expr_items bằng make_item(...)
        return ScoreResult(action="elbow", valid=True, total=..., acc=..., expr=..., ...)

  rồi import module đó trong infer_action.py (xem ACTIONS.md). Động tác chưa có luật vẫn được nhận
  dạng và hiện xác suất, chỉ không có điểm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from action_pose import action_window
from scoring_api import score_one_video
from punch_scoring import build_punch_metrics, score_punch

HEAD, LS, RS, LE, RE, LW, RW, LH, RH, LK, RK, LA, RA = range(13)


# ========== kết quả chung ==========
@dataclass
class ScoreResult:
    action: str
    valid: bool
    reason: str = ""                     # lý do khi valid=False
    total: Optional[float] = None        # /10
    acc: Optional[float] = None          # /4
    expr: Optional[float] = None         # /6
    acc_items: List[Dict] = field(default_factory=list)
    expr_items: List[Dict] = field(default_factory=list)
    top_issues: List[Dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    # mốc thời gian trên CẢ video (frame index); t_end có thể None
    phases: Dict[str, Optional[int]] = field(default_factory=dict)       # t0, t_peak, t_end
    phase_labels: Tuple[str, str, str] = ("Bắt đầu", "Đỉnh", "Kết thúc")  # nhãn 3 mốc trên biểu đồ
    phase_names: Tuple[str, str, str, str, str] = ("Chuẩn bị", "Ra đòn", "ĐỈNH", "Thu về", "Kết thúc")
    series: Dict = field(default_factory=dict)  # sig, sig_name, speed, speed_name (theo frame của cả video)
    highlight: Tuple[int, ...] = ()       # khớp tô đỏ khi vẽ (tay/chân ra đòn)
    side_text: str = ""                   # vd "tay phải", "chân trái"
    status: str = ""                      # tóm tắt chỉ số cho sơ đồ pipeline
    raw: Dict = field(default_factory=dict)


def make_item(category: str, code: str, label: str, maxi: float, got: Optional[float],
              reason: str = "", tip: str = "") -> Dict:
    """Tạo 1 hạng mục điểm (category: 'accuracy' hoặc 'expression'; got=None = không đánh giá)."""
    g = None if got is None else float(max(0.0, min(maxi, got)))
    return {"category": category, "code": code, "label": label, "maxi": float(maxi), "got": g,
            "deduct": 0.0 if g is None else float(maxi - g), "reason": reason, "tip": tip,
            "assessable": g is not None}


@dataclass
class ScorerInfo:
    name: str
    label_vi: str
    fn: Callable[..., ScoreResult]


SCORERS: Dict[str, ScorerInfo] = {}
ACTION_VI: Dict[str, str] = {}
# tên thư mục/lớp khác nhau nhưng cùng động tác -> dùng chung luật
ALIASES: Dict[str, str] = {
    "ap_chagi": "front", "dollyo_chagi": "roundhouse", "naeryo_chagi": "axe", "yeop_chagi": "side",
    "dam_moc": "hook", "dam_thang": "straight", "jireugi": "straight",
}


def register_scorer(name: str, label_vi: str):
    def deco(fn):
        SCORERS[name] = ScorerInfo(name, label_vi, fn)
        ACTION_VI[name] = label_vi
        return fn
    return deco


def canonical(action: str) -> str:
    a = action.strip().lower()
    return ALIASES.get(a, a)


def action_label(action: str) -> str:
    return ACTION_VI.get(canonical(action), action)


def get_scorer(action: str) -> Optional[ScorerInfo]:
    return SCORERS.get(canonical(action))


def score_action(action: str, pose: Dict, **opts) -> ScoreResult:
    info = get_scorer(action)
    if info is None:
        return ScoreResult(action=action, valid=False,
                           reason=f"Chưa có luật chấm cho động tác '{action}'. Xem ACTIONS.md để thêm luật.")
    if pose["kpts"].shape[0] < 2:
        return ScoreResult(action=action, valid=False, reason="Không phát hiện được người trong video.")
    res = info.fn(pose, action=canonical(action), **opts)
    if pose.get("invalid_for_scoring", 0):
        res.warnings.append("Nhiều người cùng cử động trong video — có thể chọn nhầm người.")
    return res


def _torso_len(k: np.ndarray) -> float:
    t = np.linalg.norm(0.5 * (k[:, LS] + k[:, RS]) - 0.5 * (k[:, LH] + k[:, RH]), axis=1)
    return float(np.nanmedian(t)) if np.any(np.isfinite(t)) else 1.0


# ========== ĐÁ (luật gốc của tác giả) ==========
KICK_ITEM_VI = {
    "FR_A1": ("Nâng gối trước & đường đá thẳng", "Nâng gối trước rồi mới bật cẳng chân; gối hướng thẳng trước, bàn chân đi theo đường thẳng."),
    "FR_A2": ("Độ cao & duỗi gối", "Tăng dần độ cao; duỗi gối nhanh nhưng không vung chân, đá tới đích rồi thu về được."),
    "FR_A3": ("Thu chân về", "Đá xong thu gối trước rồi mới hạ chân, trở về tư thế chuẩn bị."),
    "FR_A4": ("Trọng tâm & tay thủ", "Chân trụ vững, hông không lắc; giữ tay thủ, không buông tay xuống."),
    "RH_A1": ("Mặt phẳng vung chân & hướng đá", "Vung chân trong mặt phẳng ngang; mu bàn chân/cẳng chân cùng hướng với mục tiêu."),
    "RH_A2": ("Độ cao & duỗi gối", "Nâng chân cao dần; chân vào vị trí rồi mới quất ra, nhớ thu về."),
    "RH_A3": ("Thu chân về", "Đá xong thu gối rồi mới hạ chân, sẵn sàng nối đòn tiếp theo."),
    "RH_A4": ("Trọng tâm & kiểm soát tiếp đất", "Không đổ trọng tâm về trước; tiếp đất nhẹ, vững, vào ngay động tác tiếp."),
    "SD_A1": ("Thu gối & đường ra chân", "Đá ngang: thu gối sát ngực, gót chân hướng mục tiêu, đẩy theo đường thẳng."),
    "SD_A2": ("Độ cao & chân tới đích", "Đẩy gót chân ra; tới đích rồi thu về nhanh."),
    "SD_A3": ("Thu chân về", "Đá xong thu chân ngay rồi mới hạ, giữ nhịp thu – ra – thu."),
    "SD_A4": ("Trọng tâm & tay thủ", "Người hơi nghiêng nhưng không sụp eo; giữ tay thủ, chân trụ vững."),
    "AX_A1": ("Độ cao nâng chân & quỹ đạo", "Nâng cao rồi mới bổ xuống; quỹ đạo nâng chân ổn định, không lắc."),
    "AX_A2": ("Tốc độ bổ xuống", "Bổ xuống phải có tốc độ nhưng kiểm soát, đừng đập mạnh gây hại gối."),
    "AX_A3": ("Thu chân về", "Bổ xong cũng phải thu chân; đừng thả rơi chân xuống đất."),
    "AX_A4": ("Trọng tâm & tay thủ", "Khi nâng chân không nghiêng hông; tay thủ không buông."),
    "EX_E1": ("Tốc độ", "Tăng tốc độ đá: thả lỏng rồi bộc phát; không hy sinh đường đá."),
    "EX_E2": ("Lực", "Lực đến từ hông và chân trụ; tới đích phải dứt khoát."),
    "EX_E3": ("Độ cao", "Tăng dần độ cao, đừng gồng ép; giữ thân thẳng, chân trụ vững."),
    "EX_E4": ("Độ mượt", "Động tác không bị khựng; tập xen kẽ chậm–nhanh để quỹ đạo mượt hơn."),
}


# thuật ngữ trong phần "reason" của scoring_api.py (tiếng Trung) -> tiếng Việt
KICK_REASON_VI = [
    ("提膝", "nâng gối"), ("直线度", "độ thẳng"), ("踝高", "cổ chân cao"), ("最大膝角", "góc gối lớn nhất"),
    ("回收比例", "tỉ lệ thu về"), ("再弯膝", "gập gối lại"), ("稳定", "ổn định"), ("手位下沉", "tay thủ hạ"),
    ("对齐偏差", "lệch hướng"), ("落脚前后", "tiếp đất trước/sau"), ("侧向准备", "chuẩn bị ngang"),
    ("下行速度", "tốc độ bổ xuống"), ("速度指标", "chỉ số tốc độ"), ("爆发指标", "chỉ số bùng nổ"),
    ("高度指标", "chỉ số độ cao"), ("流畅", "độ mượt"), ("，", ", "),
]


def _translate_kick_items(items: List[Dict]) -> List[Dict]:
    out = []
    for it in items:
        it = dict(it)
        vi = KICK_ITEM_VI.get(it["code"])
        if vi:
            it["label_zh"], it["tip_zh"], it["reason_zh"] = it["label"], it["tip"], it["reason"]
            it["label"], it["tip"] = vi
            for zh, v in KICK_REASON_VI:
                it["reason"] = it["reason"].replace(zh, v)
        it.setdefault("assessable", True)
        out.append(it)
    return out


def kicking_leg(k: np.ndarray, v: np.ndarray) -> str:
    """Chân đá = chân có cổ chân lên cao hơn cổ chân kia nhiều nhất (2D). Hàm chọn chân của tác giả
    (_kick_side_indices: cổ chân xa tâm hông nhất) dễ chọn nhầm chân trụ khi chân đá duỗi thẳng."""
    yL, yR = k[:, LA, 1].astype(float), k[:, RA, 1].astype(float)
    ok = (v[:, LA] >= 0.2) & (v[:, RA] >= 0.2)
    if not np.any(ok):
        return "L"
    d = yR[ok] - yL[ok]  # y ảnh hướng xuống: > 0 khi cổ chân trái cao hơn
    return "L" if float(np.max(d)) >= float(np.max(-d)) else "R"


def kick_input(k: np.ndarray, v: np.ndarray, fps: float, window: str = "ankle",
               fix_leg: bool = False) -> Tuple[int, int, np.ndarray]:
    """Đoạn (s, e) và tọa độ đưa vào luật đá của tác giả.
    window="ankle": đoạn cắt của tác giả (như predict_video.py); "none": cả clip (dạng dữ liệu public của tác giả).
    fix_leg: đặt cổ chân TRỤ trùng tâm hông để hàm chọn chân của tác giả (cổ chân xa hông nhất) chắc chắn lấy
    đúng chân đá. Cổ chân trụ không dùng trong hạng mục nào nên công thức và ngưỡng không đổi."""
    if fix_leg:
        k = k.copy()
        k[:, RA if kicking_leg(k, v) == "L" else LA] = 0.5 * (k[:, LH] + k[:, RH])
    s, e, _ = action_window(k, v, fps, window)
    return s, e, k


def _score_kick(pose: Dict, action: str, kick_window: str = "ankle", fix_leg: bool = False, **_) -> ScoreResult:
    k, v, fps = pose["kpts"], pose["vis"], float(pose["fps"])
    # mặc định: đúng đoạn cắt của tác giả (cổ chân so với hông), như predict_video.py
    s, e, k_in = kick_input(k, v, fps, kick_window, fix_leg)
    acc4, expr6, total10, d = score_one_video((k_in[s:e + 1], v[s:e + 1], fps), action)
    raw = d["raw"]
    acc_items = _translate_kick_items(d["acc_items"])
    expr_items = _translate_kick_items(d["expr_items"])
    top = sorted(acc_items + expr_items, key=lambda x: x["deduct"], reverse=True)
    top = [x for x in top if x["deduct"] > 1e-6][:6]

    side = raw["kick_side"]
    A, H = (LA, LH) if side == "L" else (RA, RH)
    hip = 0.5 * (k[:, LH] + k[:, RH])
    L = _torso_len(k)
    dist = np.linalg.norm(k[:, A] - hip, axis=1) / L
    dist[v[:, A] < 0.2] = np.nan
    sp = np.r_[np.nan, np.linalg.norm(np.diff(k[:, A], axis=0), axis=1) * fps / L]

    warnings = []
    if not raw.get("has_enough", True):
        warnings.append("Cổ chân bị che/khó thấy trong nhiều frame — điểm kém tin cậy.")
    return ScoreResult(
        action=action, valid=True, total=total10, acc=acc4, expr=expr6,
        acc_items=acc_items, expr_items=expr_items, top_issues=top, warnings=warnings,
        phases={"t0": s + raw["t0"], "t_peak": s + raw["t_peak"], "t_end": s + raw["t1"], "win_s": s, "win_e": e},
        phase_labels=("Bắt đầu", "Đỉnh đá", "Kết thúc"),
        phase_names=("Chuẩn bị", "Ra chân", "ĐỈNH ĐÁ", "Thu chân", "Kết thúc"),
        series={"sig": dist, "sig_name": "Cổ chân cách hông (× thân)", "speed": sp,
                "speed_name": "Tốc độ cổ chân (thân/s)"},
        highlight=(LH, LK, LA) if side == "L" else (RH, RK, RA),
        side_text="chân trái" if side == "L" else "chân phải",
        status=f"gối duỗi {raw['kick_knee_angle_peak']:.0f}°, cao {raw['kick_height_over_hip']:.2f}\n"
               f"thu về {raw['recovery_ratio'] * 100:.0f}%",
        raw=raw,
    )


for _name, _vi in [("front", "Ap chagi (đá tống trước)"), ("roundhouse", "Dollyo chagi (đá vòng cầu)"),
                   ("axe", "Naeryo chagi (đá bổ)"), ("side", "Yeop chagi (đá ngang)")]:
    register_scorer(_name, _vi)(_score_kick)


# ========== ĐẤM (punch_scoring.py) ==========
def _score_punch(pose: Dict, action: str, guard: str = "chin", side: Optional[str] = None, **_) -> ScoreResult:
    series: Dict = {}
    m = build_punch_metrics(pose["kpts"], pose["vis"], float(pose["fps"]), pose["world"],
                            punch_type=action, side=side, guard_mode=guard, debug=series)
    acc4, expr6, total10, d = score_punch(m)
    hl = {"R": (RS, RE, RW), "L": (LS, LE, LW)}.get(series.get("side"), ())
    side_text = {"R": "tay phải", "L": "tay trái"}.get(series.get("side"), "")
    sres = {"sig": series.get("sig"), "sig_name": series.get("sig_name", ""),
            "speed": series.get("speed"), "speed_name": "Tốc độ cổ tay (thân/s)"} if series else {}
    common = dict(action=action, phase_labels=("Bắt đầu", "Chạm", "Về thủ"),
                  phase_names=("Thủ", "Ra đòn", "ĐIỂM CHẠM", "Thu tay", "Về thế thủ"),
                  series=sres, highlight=hl, side_text=side_text, raw=d.get("raw", {}))
    if not m.valid:
        return ScoreResult(valid=False, reason=m.invalid_reason, **common)
    rot = "—" if m.shoulder_rot_deg is None else f"{m.shoulder_rot_deg:.0f}°"
    shape = (f"khuỷu {m.elbow_flex_deg:.0f}°, nâng {m.upper_arm_elev_deg:.0f}°" if action == "hook"
             else f"khuỷu {m.elbow_flex_deg:.0f}°, thẳng {m.path_straightness:.2f}")
    return ScoreResult(
        valid=True, total=total10, acc=acc4, expr=expr6,
        acc_items=d["acc_items"], expr_items=d["expr_items"], top_issues=d["top_issues"],
        warnings=list(d["warnings"]),
        phases={"t0": m.t0, "t_peak": m.t_imp, "t_end": m.t_rec},
        status=f"{shape}\nxoay vai {rot}, v={m.peak_speed:.1f} thân/s",
        **common,
    )


register_scorer("hook", "Đấm móc")(_score_punch)
register_scorer("straight", "Đấm thẳng")(_score_punch)
