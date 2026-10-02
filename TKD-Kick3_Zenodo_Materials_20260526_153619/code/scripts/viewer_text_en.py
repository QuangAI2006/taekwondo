# scripts/viewer_text_en.py
"""
Bảng dịch sang tiếng Anh cho cửa sổ pipeline (pipeline_viewer.py).

Luật chấm (action_scorers.py, punch_scoring.py) và kết quả in ra PowerShell vẫn dùng tiếng Việt; cửa sổ
dịch các chuỗi đó qua en() / item_en() / action_en(). Chuỗi chưa có trong bảng được giữ nguyên — khi thêm
luật chấm mới, thêm bản dịch vào đây (hoặc đặt "label_en" cho từng hạng mục).
"""

from __future__ import annotations

import re
from typing import Dict

ACTION_EN: Dict[str, str] = {
    "front": "Ap chagi (front kick)", "roundhouse": "Dollyo chagi (roundhouse kick)",
    "axe": "Naeryo chagi (axe kick)", "side": "Yeop chagi (side kick)",
    "hook": "Hook punch", "straight": "Straight punch",
    # scorers_*.py
    "back": "Dwit chagi (back kick)", "hook_kick": "Huryeo chagi (hook kick)", "push": "Mireo chagi (push kick)",
    "side_low": "Low side kick", "roundhouse_low": "Low roundhouse kick",
    "high_block": "Olgul makki (high block)", "low_block": "Arae makki (low block)",
    "knife_block": "Sonnal makki (double knife-hand block)", "knife_strike": "Sonnal chigi (knife-hand strike)",
    "spear_hand": "Pyeonsonkkeut tulki (spear-hand thrust)",
    "horse_stance": "Juchum seogi (horse stance)", "front_stance": "Ap kubi (long front stance)",
    "block_punch": "Low block + straight punch", "block_kick": "Low block + front kick",
    "horse_punch": "Wide-stance straight punch", "footwork": "Footwork (hop forward/back)",
    "duck": "Duck (evasion)", "knee_chamber": "Knee chamber",
}

ITEM_EN: Dict[str, str] = {
    # đá (scoring_api.py)
    "FR_A1": "Knee chamber & straight path", "FR_A2": "Height & knee extension",
    "FR_A3": "Leg retraction", "FR_A4": "Balance & guard",
    "RH_A1": "Swing plane & direction", "RH_A2": "Height & knee extension",
    "RH_A3": "Leg retraction", "RH_A4": "Balance & landing control",
    "SD_A1": "Chamber & kick path", "SD_A2": "Height & reaching target",
    "SD_A3": "Leg retraction", "SD_A4": "Balance & guard",
    "AX_A1": "Leg lift height & path", "AX_A2": "Downward speed",
    "AX_A3": "Leg retraction", "AX_A4": "Balance & guard",
    "EX_E1": "Speed", "EX_E2": "Power", "EX_E3": "Height", "EX_E4": "Smoothness",
    # đấm (punch_scoring.py)
    "HK_A1": "Arm shape at impact", "HK_A2": "Hip & shoulder rotation",
    "ST_A1": "Arm extension & straight path", "ST_A2": "Shoulder rotation & target height",
    "PU_A3": "Return to guard", "PU_A4": "Guard hand & balance",
    "PX_E1": "Speed", "PX_E2": "Power (trunk rotation)", "PX_E3": "Snap (execution time)",
    "PX_E4": "Smoothness",
    # đá thêm (scorers_kicks_extra.py)
    "BK_A1": "Heel thrust straight back", "BK_A2": "Height & knee extension", "BK_A3": "Leg retraction",
    "BK_A4": "Balance & guard",
    "HKK_A1": "Extend then hook the heel", "HKK_A2": "Height & arc path", "HKK_A3": "Leg retraction",
    "HKK_A4": "Balance & guard",
    "PK_A1": "High chamber, straight push", "PK_A2": "Leg horizontal at impact", "PK_A3": "Leg retraction",
    "PK_A4": "Balance & guard",
    # kỹ thuật tay (scorers_hands.py)
    "HB_A1": "Blocking forearm above forehead", "HB_A2": "Rising path & other hand", "HB_A3": "Arm recovery",
    "HB_A4": "Balance",
    "LB_A1": "Blocking arm extended above knee", "LB_A2": "From opposite shoulder & other hand",
    "LB_A3": "Crisp stop at block end", "LB_A4": "Balance & stance",
    "KB_A1": "Front hand at shoulder, rear hand at chest", "KB_A2": "Wind-up & stepping out",
    "KB_A3": "Arm recovery", "KB_A4": "Balance",
    "KS_A1": "Striking arm extended at neck height", "KS_A2": "Wind-up, pull-back hand & stance",
    "KS_A3": "Arm recovery", "KS_A4": "Balance",
    "SP_A1": "Arm extension & straight thrust", "SP_A2": "Target height & other hand",
    "SP_A3": "Arm recovery", "SP_A4": "Balance & stance",
    "HX_E1": "Speed", "HX_E2": "Power (trunk rotation)", "HX_E3": "Snap (execution time)", "HX_E4": "Smoothness",
    "HP_A4": "Pull-back hand, balance & wide stance",
    # tấn (scorers_stances.py)
    "HS_A1": "Knee bend", "HS_A2": "Stance width & knees out", "HS_A3": "Upright trunk", "HS_A4": "Left/right balance",
    "FS_A1": "Front knee bent, rear leg straight", "FS_A2": "Stance length", "FS_A3": "Upright trunk",
    "FS_A4": "Feet not on one line",
    "SX_E1": "Stability (holding still)", "SX_E2": "Low centre of gravity", "SX_E3": "Stance held",
    "SX_E4": "Steady upper body",
    # bộ pháp / phòng thủ (scorers_drills.py)
    "FW_A1": "Fighting stance kept while moving", "FW_A2": "Guard", "FW_A3": "Upright trunk",
    "FW_A4": "Back to stance after step",
    "FX_E1": "Speed", "FX_E2": "Step length", "FX_E3": "Snap (execution time)", "FX_E4": "Smoothness",
    "DK_A1": "Lowering with the knees", "DK_A2": "Guard covering face", "DK_A3": "Spring back to guard",
    "DK_A4": "Back not over-bent",
    "DX_E1": "Speed", "DX_E2": "Spring up", "DX_E3": "Snap (execution time)", "DX_E4": "Smoothness",
    "KC_A1": "Knee high & tightly folded", "KC_A2": "Guard", "KC_A3": "Leg back to guard",
    "KC_A4": "Support-leg balance",
    "KX_E1": "Speed", "KX_E2": "Knee lift range", "KX_E3": "Snap (execution time)", "KX_E4": "Smoothness",
}

# khớp nguyên chuỗi
PHRASE_EN: Dict[str, str] = {
    # nhãn mốc / tên pha
    "Bắt đầu": "Start", "Đỉnh": "Peak", "Kết thúc": "End", "Đỉnh đá": "Kick peak", "Chạm": "Impact",
    "Về thủ": "Back to guard", "Chuẩn bị": "Ready", "Ra chân": "Kicking", "ĐỈNH ĐÁ": "KICK PEAK",
    "Thu chân": "Retracting leg", "Thủ": "Guard", "Ra đòn": "Striking", "ĐIỂM CHẠM": "IMPACT",
    "Thu tay": "Retracting arm", "Về thế thủ": "In guard", "Thu về": "Recovering", "ĐỈNH": "PEAK",
    # tên tín hiệu
    "Cổ chân cách hông (× thân)": "Ankle–hip distance (× torso)",
    "Tốc độ cổ chân (thân/s)": "Ankle speed (torso/s)",
    "Độ nâng khuỷu (°)": "Elbow elevation (°)",
    "Độ vươn tay (× dài tay)": "Arm reach (× arm length)",
    "Tốc độ cổ tay (thân/s)": "Wrist speed (torso/s)",
    "Cổ tay cao hơn mũi (× thân)": "Wrist above nose (× torso)",
    "Cổ tay thấp hơn vai (× thân)": "Wrist below shoulder (× torso)",
    "Góc gối nhỏ nhất (°)": "Smallest knee angle (°)", "Độ rộng tấn (× vai)": "Stance width (× shoulders)",
    "Dịch chuyển hông (× thân)": "Hip displacement (× torso)", "Tốc độ hông (thân/s)": "Hip speed (torso/s)",
    "Đầu hạ xuống (× thân)": "Head drop (× torso)", "Tốc độ đầu (thân/s)": "Head speed (torso/s)",
    "Gối so với hông (× thân)": "Knee above hip (× torso)", "Tốc độ gối (thân/s)": "Knee speed (torso/s)",
    "Điểm đến": "End point", "ĐIỂM ĐẾN": "END POINT", "Bắt đầu giữ": "Hold start", "Giữa": "Middle",
    "Kết thúc giữ": "Hold end", "Vào tấn": "Entering", "Giữ tấn": "Holding", "GIỮ TẤN": "HOLDING",
    "Ra tấn": "Leaving", "Bắt đầu gạt": "Block start", "Gạt": "Block", "ĐẤM": "PUNCH", "ĐÁ": "KICK",
    "Gạt thấp": "Low block", "Đấm thẳng": "Straight punch", "Đá trước": "Front kick",
    "Xa nhất": "Farthest", "Ổn định": "Settled", "XA NHẤT": "FARTHEST", "Di chuyển": "Moving",
    "Thấp nhất": "Lowest", "THẤP NHẤT": "LOWEST", "Hạ người": "Ducking", "Bật lên": "Springing up",
    "Gối cao nhất": "Knee highest", "CAO NHẤT": "HIGHEST", "Nâng gối": "Lifting knee", "Hạ chân": "Lowering leg",
    "Chân đá đi ra phía trước mặt — có thể không phải đá sau.":
        "The kicking leg goes forward — this may not be a back kick.",
    # cảnh báo
    "Video bắt đầu gần như ngay khi ra đòn — tư thế thủ ban đầu đo không chắc chắn.":
        "Video starts almost at the strike — the initial guard posture is uncertain.",
    "Video bắt đầu giữa động tác — các mục xoay thân/thời gian ra đòn không tính.":
        "Video starts mid-movement — trunk rotation and execution time are not scored.",
    "Thân xoay rất nhiều (có thể quay lưng lại camera) — MediaPipe dễ nhầm trái/phải, số đo kém tin cậy.":
        "Very large trunk rotation (possibly back to the camera) — MediaPipe may swap left/right; "
        "measurements are unreliable.",
    "Video kết thúc trước khi thu tay — hạng mục thu tay không tính, điểm Chính xác quy đổi theo tỉ lệ.":
        "Video ends before the arm returns — the recovery item is not scored; Accuracy is rescaled.",
    "Cổ tay đấm bị che khuất nhiều — số đo kém tin cậy.":
        "The punching wrist is often hidden — measurements are unreliable.",
    "Nhiều người cùng cử động trong video — có thể chọn nhầm người.":
        "Several people are moving — the wrong person may have been selected.",
    "Cổ chân bị che/khó thấy trong nhiều frame — điểm kém tin cậy.":
        "The ankle is hidden in many frames — the score is unreliable.",
    # lý do không chấm được
    "Không thấy một cú đấm trọn vẹn: video bắt đầu/kết thúc giữa động tác, không có cú đấm, "
    "hoặc góc quay (nghiêng hẳn/quay lưng) khiến không đo được tay. Nên quay chính diện hoặc chéo ~45°.":
        "No complete punch found: the video starts or ends mid-movement, contains no punch, or the camera "
        "angle (side-on / back view) prevents measuring the arm. Film from the front or at ~45°.",
    "Video quá ngắn hoặc thiếu tọa độ 3D": "Video too short or 3D coordinates missing",
    "Không ước lượng được kích thước cơ thể": "Could not estimate body size",
    "Không phát hiện được người trong video.": "No person detected in the video.",
}

# chuỗi có số liệu thay đổi
REGEX_EN = [
    (r"^Model không chắc chắn \((\d+)%\).*$",
     r"Model is not confident (\1%) — possibly an action it was not trained on, or a different camera angle."),
    (r"^Chưa có luật chấm cho động tác '(.+)'\..*$", r"No scoring rule for action '\1' yet. See ACTIONS.md."),
    (r"^Lỗi: (.*)$", r"Error: \1"),
    (r"gối duỗi (\S+)°, cao (\S+)", r"knee ext. \1°, height \2"),
    (r"khuỷu (\S+)°, nâng (\S+)°", r"elbow \1°, elevation \2°"),
    (r"khuỷu (\S+)°, thẳng (\S+)", r"elbow \1°, straightness \2"),
    (r"khuỷu (\S+)°", r"elbow \1°"),
    (r"thu về (\S+)%", r"recovery \1%"),
    (r"xoay vai (\S+), v=(\S+) thân/s", r"shoulder rot. \1, v=\2 torso/s"),
]


def en(text: str) -> str:
    if not text:
        return text
    if text in PHRASE_EN:
        return PHRASE_EN[text]
    out = text
    for pat, rep in REGEX_EN:
        out = re.sub(pat, rep, out)
    return out


def action_en(name: str) -> str:
    return ACTION_EN.get((name or "").strip().lower(), name)


def item_en(item: dict) -> str:
    return item.get("label_en") or ITEM_EN.get(item.get("code", ""), en(item.get("label", "")))
