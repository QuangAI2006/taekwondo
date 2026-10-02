# scripts/scorers_hands.py
"""
Luật chấm các kỹ thuật TAY không phải đấm: đỡ cao, gạt thấp, chém cạnh 2 tay, chém đao 1 tay, chọc mũi tay.

Cùng khung với luật đá của tác giả (scoring_api.py) và luật đấm (punch_scoring.py):
  Chính xác 4 = A1 1.6 (hình dạng tay ở điểm đến) + A2 1.2 (quỹ đạo / tay còn lại / tấn)
              + A3 0.8 (thu tay về) + A4 0.4 (thăng bằng)
  Biểu hiện 6 = E1 tốc độ 2.0 + E2 lực (xoay thân) 1.6 + E3 dứt khoát 1.4 + E4 độ mượt 1.0
Biểu hiện dùng lại đúng ngưỡng của luật đấm (PunchRanges) để các kỹ thuật tay so sánh được với nhau.
Hạng mục không đo được (vd video bắt đầu giữa động tác) -> không chấm, quy đổi theo tỉ lệ (như luật đấm).

Chỉ số đo ở hand_scoring.py. Mọi ngưỡng nằm trong HandRanges. Xem OTHER_TECHNIQUES_SCORING.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Tuple

from action_scorers import ALIASES, ScoreResult, register_scorer
from hand_scoring import HandMetrics, build_hand_metrics, rep_signal
from punch_scoring import PunchRanges, _add_item, _band01, _component, _map01
from scorers_stances import stance_at

HEAD, LS, RS, LE, RE, LW, RW, LH, RH, LK, RK, LA, RA = range(13)
PR = PunchRanges()


@dataclass
class HandRanges:
    # đỡ cao (olgul makki)
    hb_above_head: Tuple[float, float, float, float] = (-0.05, 0.10, 0.55, 0.80)  # cổ tay cao hơn mũi (× thân)
    hb_elbow_up: Tuple[float, float, float, float] = (-0.15, 0.05, 0.60, 0.80)    # khuỷu cao hơn vai
    hb_flex: Tuple[float, float, float, float] = (55.0, 85.0, 145.0, 170.0)       # góc khuỷu
    hb_tilt: Tuple[float, float] = (50.0, 20.0)                                   # |cẳng tay so với ngang|
    hb_rise: Tuple[float, float] = (0.50, 1.10)                                   # biên độ nâng tay (× thân)
    # gạt thấp (arae makki)
    lb_flex: Tuple[float, float] = (125.0, 160.0)
    lb_vs_hip: Tuple[float, float, float, float] = (-0.85, -0.60, -0.05, 0.15)    # cổ tay cao hơn hông (âm = thấp hơn)
    lb_start_high: Tuple[float, float] = (-0.60, -0.05)                           # cổ tay cao nhất trước khi gạt, so với vai
    lb_stop: Tuple[float, float] = (0.30, 0.10)                                   # cổ tay dịch trong 0.25 s sau điểm gạt
    # chém / đỡ cạnh tay (sonnal)
    kn_height: Tuple[float, float, float, float] = (-0.45, -0.20, 0.25, 0.45)     # cổ tay so với vai
    kb_flex: Tuple[float, float, float, float] = (95.0, 120.0, 170.0, 181.0)      # đỡ cạnh tay: hơi gập
    ks_flex: Tuple[float, float] = (125.0, 160.0)                                 # chém: gần duỗi
    ks_reach: Tuple[float, float] = (0.70, 0.90)
    windup_high: Tuple[float, float] = (-0.50, 0.00)                              # lấy đà: tay lên ngang vai/tai
    windup_cross: Tuple[float, float] = (-0.20, 0.25)                             # lấy đà: tay sang bên kia giữa thân
    sternum: Tuple[float, float] = (0.70, 0.25)                                   # tay sau trước ngực
    # chọc mũi tay
    sp_height: Tuple[float, float, float, float] = (-0.80, -0.55, 0.35, 0.60)     # từ thượng vị tới tầm mặt
    sp_support: Tuple[float, float] = (0.70, 0.25)                                # tay kia đỡ dưới khuỷu
    # tay còn lại giữ thủ (cằm hoặc hông, tùy kiểu thủ)
    other_guard: Tuple[float, float] = (0.85, 0.40)
    other_hip: Tuple[float, float] = (0.80, 0.35)


HR = HandRanges()


def _expression(m: HandMetrics) -> List[Dict]:
    expr: List[Dict] = []
    _add_item(expr, "expression", "HX_E1", "Tốc độ", 2.0, _map01(m.peak_speed, *PR.speed) * 2.0,
              f"tốc độ cổ tay≈{m.peak_speed:.1f} thân/giây",
              "Thả lỏng rồi bộc phát; tay đi nhanh nhất ngay trước điểm kết thúc động tác.", key=True)
    _add_item(expr, "expression", "HX_E2", "Lực (xoay thân truyền lực)", 1.6,
              None if m.peak_rot_speed is None else _map01(m.peak_rot_speed, *PR.rot_speed) * 1.6,
              "video bắt đầu giữa động tác — không đánh giá" if m.peak_rot_speed is None
              else f"tốc độ xoay vai≈{m.peak_rot_speed:.0f}°/s",
              "Xoay hông – vai cùng lúc với tay; lực đến từ thân chứ không chỉ từ cánh tay.", key=True)
    _add_item(expr, "expression", "HX_E3", "Dứt khoát (thời gian ra đòn)", 1.4,
              None if m.exec_time is None else _map01(m.exec_time, *PR.exec_time) * 1.4,
              "video bắt đầu giữa động tác — không đánh giá" if m.exec_time is None
              else f"từ lúc bắt đầu tới điểm kết thúc≈{m.exec_time:.2f}s",
              "Ra đòn gọn, dừng dứt khoát ở điểm kết thúc.")
    _add_item(expr, "expression", "HX_E4", "Độ mượt", 1.0, _map01(m.smoothness, *PR.smooth) * 1.0,
              f"độ mượt≈{m.smoothness:.2f}", "Một nhịp tăng tốc liền mạch, không khựng giữa chừng.")
    return expr


def _recovery(code: str, m: HandMetrics) -> List[Dict]:
    out: List[Dict] = []
    _add_item(out, "accuracy", code, "Thu tay về", 0.8,
              None if m.recovery_ratio is None else _map01(m.recovery_ratio, *PR.recovery) * 0.8,
              "video kết thúc trước khi thu tay — không đánh giá" if m.recovery_ratio is None
              else f"tay trở về vị trí ban đầu≈{m.recovery_ratio * 100:.0f}%",
              "Xong động tác thu tay về tư thế chuẩn bị, không để tay treo.", key=True)
    return out


def _balance(code: str, m: HandMetrics, stance: Optional[Dict] = None) -> List[Dict]:
    bal = 0.5 * _map01(m.trunk_lean_deg, *PR.lean) + 0.5 * _map01(m.hip_sway, *PR.sway)
    reason = f"nghiêng thân≈{m.trunk_lean_deg:.0f}°, lắc hông≈{m.hip_sway:.2f}"
    tip = "Thân thẳng, trọng tâm ổn định suốt động tác."
    if stance is not None:  # động tác có bước mở chân: thêm độ rộng tấn và độ khuỵu gối
        bal = 0.5 * bal + 0.5 * stance["open01"]
        reason += f", tấn rộng≈{stance['width']:.1f}× vai, gối≈{stance['knee_min']:.0f}°"
        tip = "Bước mở chân đủ rộng (~2 lần vai), khuỵu gối, thân thẳng."
    out: List[Dict] = []
    _add_item(out, "accuracy", code, "Thăng bằng & tấn", 0.4, bal * 0.4, reason, tip)
    return out


def _other_guard(m: HandMetrics) -> float:
    return _map01(min(m.other_to_chin, m.other_to_hip), *HR.other_guard)


# ========== kỹ thuật ==========
def _acc_high_block(m: HandMetrics, st) -> List[Dict]:
    acc: List[Dict] = []
    a1 = (0.35 * _band01(m.wrist_vs_head, *HR.hb_above_head) + 0.25 * _band01(m.elbow_vs_shoulder, *HR.hb_elbow_up)
          + 0.20 * _band01(m.elbow_flex_deg, *HR.hb_flex) + 0.20 * _map01(abs(m.forearm_tilt_deg), *HR.hb_tilt))
    _add_item(acc, "accuracy", "HB_A1", "Tay đỡ nằm trên trán", 1.6, a1 * 1.6,
              f"cổ tay cao hơn mũi≈{m.wrist_vs_head:.2f}, khuỷu cao hơn vai≈{m.elbow_vs_shoulder:.2f}, "
              f"góc khuỷu≈{m.elbow_flex_deg:.0f}°, cẳng tay nghiêng≈{m.forearm_tilt_deg:.0f}°",
              "Cẳng tay nằm ngang cách trán khoảng một nắm tay, khuỷu gập ~90–120°, mặt ngoài cẳng tay hướng lên.",
              key=True)
    a2 = 0.5 * _map01(m.rise, *HR.hb_rise) + 0.5 * _other_guard(m)
    _add_item(acc, "accuracy", "HB_A2", "Quỹ đạo nâng tay & tay còn lại", 1.2, a2 * 1.2,
              f"biên độ nâng tay≈{m.rise:.2f} thân, tay còn lại cách cằm/hông≈"
              f"{min(m.other_to_chin, m.other_to_hip):.2f}",
              "Đỡ từ dưới lên qua trước mặt; tay còn lại kéo về hông (hoặc giữ thủ ở cằm), không vung theo.",
              key=True)
    return acc + _recovery("HB_A3", m) + _balance("HB_A4", m)


def _acc_low_block(m: HandMetrics, st) -> List[Dict]:
    acc: List[Dict] = []
    a1 = 0.5 * _map01(m.elbow_flex_deg, *HR.lb_flex) + 0.5 * _band01(m.wrist_vs_hip, *HR.lb_vs_hip)
    _add_item(acc, "accuracy", "LB_A1", "Tay gạt duỗi, dừng trên gối", 1.6, a1 * 1.6,
              f"góc khuỷu≈{m.elbow_flex_deg:.0f}°, cổ tay so với hông≈{m.wrist_vs_hip:+.2f} thân",
              "Kết thúc tay gần duỗi thẳng, nắm tay cách đùi trước khoảng 2 nắm tay.", key=True)
    a2 = (0.35 * _map01(m.start_wrist_vs_shoulder, *HR.lb_start_high)
          + 0.25 * _map01(-m.start_lateral_min, *HR.windup_cross) + 0.40 * _other_guard(m))
    _add_item(acc, "accuracy", "LB_A2", "Gạt từ vai đối diện & tay còn lại", 1.2, a2 * 1.2,
              f"tay lấy đà cao≈{m.start_wrist_vs_shoulder:+.2f} so với vai, sang bên kia≈{-m.start_lateral_min:.2f} "
              f"vai, tay còn lại cách hông/cằm≈{min(m.other_to_chin, m.other_to_hip):.2f}",
              "Nắm tay gạt bắt đầu ở vai đối diện, quét chéo xuống; tay kia kéo mạnh về hông.", key=True)
    # bài bước gạt liên tục: gạt xong giữ tay ở dưới rồi mới lấy đà cho lần sau -> A3 chấm độ dừng dứt khoát
    # ở điểm gạt thay vì thu tay về
    _add_item(acc, "accuracy", "LB_A3", "Dừng dứt khoát ở điểm gạt", 0.8,
              None if m.stop_disp is None else _map01(m.stop_disp, *HR.lb_stop) * 0.8,
              "video kết thúc ngay sau điểm gạt — không đánh giá" if m.stop_disp is None
              else f"cổ tay còn trôi≈{m.stop_disp:.2f} thân trong 0.25 s sau điểm gạt",
              "Gạt tới đích thì khóa tay lại một nhịp, không để tay trôi tiếp.", key=True)
    return acc + _balance("LB_A4", m, st)


def _windup(m: HandMetrics) -> float:
    return 0.5 * _map01(m.start_wrist_vs_shoulder, *HR.windup_high) + 0.5 * _map01(-m.start_lateral_min,
                                                                                  *HR.windup_cross)


def _acc_knife_block(m: HandMetrics, st) -> List[Dict]:
    acc: List[Dict] = []
    a1 = (0.35 * _band01(m.elbow_flex_deg, *HR.kb_flex) + 0.30 * _band01(m.wrist_vs_shoulder, *HR.kn_height)
          + 0.35 * _map01(m.other_to_sternum, *HR.sternum))
    _add_item(acc, "accuracy", "KB_A1", "Tay trước ngang vai, tay sau trước ngực", 1.6, a1 * 1.6,
              f"góc khuỷu tay trước≈{m.elbow_flex_deg:.0f}°, cổ tay so với vai≈{m.wrist_vs_shoulder:+.2f}, "
              f"tay sau cách giữa ngực≈{m.other_to_sternum:.2f}",
              "Cạnh bàn tay trước ngang vai, khuỷu hơi gập; bàn tay sau ngửa, đặt trước thượng vị.", key=True)
    a2 = 0.5 * _windup(m) + 0.5 * st["open01"]
    _add_item(acc, "accuracy", "KB_A2", "Lấy đà từ vai đối diện & bước mở tấn", 1.2, a2 * 1.2,
              f"tay lấy đà cao≈{m.start_wrist_vs_shoulder:+.2f}, sang bên kia≈{-m.start_lateral_min:.2f} vai, "
              f"tấn rộng≈{st['width']:.1f}× vai, gối≈{st['knee_min']:.0f}°",
              "Hai tay đưa lên cạnh tai bên kia rồi mới chém ra; bước mở chân và khuỵu gối cùng lúc.", key=True)
    return acc + _recovery("KB_A3", m) + _balance("KB_A4", m)


def _acc_knife_strike(m: HandMetrics, st) -> List[Dict]:
    acc: List[Dict] = []
    a1 = (0.40 * _map01(m.elbow_flex_deg, *HR.ks_flex) + 0.30 * _band01(m.wrist_vs_shoulder, *HR.kn_height)
          + 0.30 * _map01(m.reach, *HR.ks_reach))
    _add_item(acc, "accuracy", "KS_A1", "Tay chém duỗi, ngang tầm cổ", 1.6, a1 * 1.6,
              f"góc khuỷu≈{m.elbow_flex_deg:.0f}°, cổ tay so với vai≈{m.wrist_vs_shoulder:+.2f}, "
              f"độ vươn≈{m.reach:.2f}",
              "Chém bằng cạnh bàn tay, tay gần duỗi khi tới đích, ngang tầm cổ/vai.", key=True)
    a2 = 0.4 * _windup(m) + 0.3 * _map01(m.other_to_hip, *HR.other_hip) + 0.3 * st["open01"]
    _add_item(acc, "accuracy", "KS_A2", "Lấy đà, tay kéo về hông & bước mở tấn", 1.2, a2 * 1.2,
              f"tay lấy đà cao≈{m.start_wrist_vs_shoulder:+.2f}, sang bên kia≈{-m.start_lateral_min:.2f} vai, "
              f"tay kia cách hông≈{m.other_to_hip:.2f}, tấn rộng≈{st['width']:.1f}× vai",
              "Tay chém bắt đầu từ vai đối diện; tay kia kéo về hông; bước mở chân cùng lúc.", key=True)
    return acc + _recovery("KS_A3", m) + _balance("KS_A4", m)


def _acc_spear(m: HandMetrics, st) -> List[Dict]:
    acc: List[Dict] = []
    a1 = 0.6 * _map01(m.elbow_flex_deg, *PR.straight_flex) + 0.4 * _map01(m.path_straightness, *PR.straight_path)
    _add_item(acc, "accuracy", "SP_A1", "Duỗi tay & đường chọc thẳng", 1.6, a1 * 1.6,
              f"góc khuỷu≈{m.elbow_flex_deg:.0f}°, độ thẳng quỹ đạo≈{m.path_straightness:.2f}",
              "Chọc thẳng bằng đầu các ngón tay, tay gần duỗi hết khi tới đích.", key=True)
    other = max(_map01(m.other_to_elbow, *HR.sp_support), _map01(m.other_to_hip, *HR.other_hip))
    a2 = 0.5 * _band01(m.wrist_vs_shoulder, *HR.sp_height) + 0.5 * other
    _add_item(acc, "accuracy", "SP_A2", "Đúng tầm & tay còn lại", 1.2, a2 * 1.2,
              f"cổ tay so với vai≈{m.wrist_vs_shoulder:+.2f}, tay kia cách khuỷu≈{m.other_to_elbow:.2f} / "
              f"cách hông≈{m.other_to_hip:.2f}",
              "Mũi tay ở tầm thượng vị đến mặt; tay kia úp dưới khuỷu tay chọc hoặc kéo về hông.", key=True)
    return acc + _recovery("SP_A3", m) + _balance("SP_A4", m, st)


TECHNIQUES: Dict[str, Tuple[str, str, Callable, float, float]] = {
    # tên: (nhãn, tín hiệu, hàm hạng mục Chính xác, biên độ tối thiểu, dung sai đỉnh)
    "high_block": ("Olgul makki (đỡ cao)", "head_up", _acc_high_block, 0.40, 0.03),
    "low_block": ("Arae makki (gạt thấp)", "drop", _acc_low_block, 0.40, 0.03),
    "knife_block": ("Sonnal makki (chém cạnh 2 tay)", "out", _acc_knife_block, 0.50, 0.05),
    "knife_strike": ("Sonnal chigi (chém đao 1 tay)", "out", _acc_knife_strike, 0.50, 0.05),
    "spear_hand": ("Pyeonsonkkeut tulki (chọc mũi tay)", "thrust", _acc_spear, 0.35, 0.02),
}

SIG_NAME = {"head_up": "Cổ tay cao hơn mũi (× thân)", "drop": "Cổ tay thấp hơn vai (× thân)",
            "reach": "Độ vươn tay (× dài tay)", "out": "Cổ tay vươn ngang (× rộng vai)",
            "thrust": "Độ vươn tay (× dài tay)"}


def score_hand(pose: Dict, action: str, side: Optional[str] = None, **_) -> ScoreResult:
    label, signal, acc_fn, need, tol = TECHNIQUES[action]
    series: Dict = {}
    m = build_hand_metrics(pose["kpts"], pose["vis"], float(pose["fps"]), pose.get("world"), signal,
                           side=side, need_rise=need, tol=tol, debug=series)
    hl = {"R": (RS, RE, RW), "L": (LS, LE, LW)}.get(series.get("side"), ())
    common = dict(action=action, phase_labels=("Bắt đầu", "Điểm đến", "Thu về"),
                  phase_names=("Chuẩn bị", "Ra đòn", "ĐIỂM ĐẾN", "Thu tay", "Kết thúc"),
                  series={"sig": series.get("sig"), "sig_name": SIG_NAME[signal], "speed": series.get("speed"),
                          "speed_name": "Tốc độ cổ tay (thân/s)"} if series else {},
                  highlight=hl, side_text={"R": "tay phải", "L": "tay trái"}.get(series.get("side"), ""))
    if not m.valid:
        return ScoreResult(valid=False, reason=m.invalid_reason, raw=asdict(m), **common)
    st = stance_at(pose.get("world"), pose["kpts"], m.t_imp)
    acc = acc_fn(m, st)
    expr = _expression(m)
    acc4, expr6 = _component(acc, 4.0), _component(expr, 6.0)
    top = sorted([d for d in acc + expr if d["assessable"] and d["deduct"] > 1e-6], key=lambda d: -d["deduct"])[:6]
    warnings = []
    if m.exec_time is None:
        warnings.append("Video bắt đầu giữa động tác — các mục xoay thân/thời gian ra đòn không tính.")
    if m.recovery_ratio is None:
        warnings.append("Video kết thúc trước khi thu tay — hạng mục thu tay không tính, điểm Chính xác quy đổi theo tỉ lệ.")
    if m.wrist_visibility < 0.6:
        warnings.append("Cổ tay bị che khuất nhiều — số đo kém tin cậy.")
    raw = asdict(m)
    raw.update({f"stance_{k}": v for k, v in st.items()})
    return ScoreResult(
        valid=True, total=acc4 + expr6, acc=acc4, expr=expr6, acc_items=acc, expr_items=expr, top_issues=top,
        warnings=warnings, phases={"t0": m.t0, "t_peak": m.t_imp, "t_end": m.t_rec},
        status=f"khuỷu {m.elbow_flex_deg:.0f}°, v={m.peak_speed:.1f} thân/s", raw=raw, **common)


score_hand.rep_signal_fn = lambda pose, action, side=None, **_: rep_signal(pose, TECHNIQUES[action][1], side)
score_hand.rep_cfg = {"min_gap_s": 0.8}
for _name, (_label, *_rest) in TECHNIQUES.items():
    register_scorer(_name, _label)(score_hand)

ALIASES.update({"olgul_makki": "high_block", "do_cao": "high_block", "arae_makki": "low_block", "gat_thap": "low_block",
                "sonnal_makki": "knife_block", "chem_canh": "knife_block", "sonnal_chigi": "knife_strike",
                "chem_dao": "knife_strike", "pyeonsonkkeut_tulki": "spear_hand", "choc_mui_tay": "spear_hand",
                "dam_vong": "hook"})
