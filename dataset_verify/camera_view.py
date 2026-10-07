"""Estimate camera position relative to the participant's INITIAL body orientation.

Labels left/right mean the participant's anatomical sides, not image sides.
Confidence is a geometric heuristic, not a calibrated probability.
"""
from __future__ import annotations

import numpy as np

KNOWN_LABELS = ('front', 'left', 'right')
LABEL_VI = {'front': 'chính diện', 'left': 'trái', 'right': 'phải',
            'unidentify_angle': 'không xác định góc quay'}
ALIASES = {'front': 'front', 'chinh_dien': 'front', 'chính diện': 'front',
           'left': 'left', 'side_left': 'left', 'oblique_left': 'left', 'trai': 'left', 'trái': 'left',
           'right': 'right', 'side_right': 'right', 'oblique_right': 'right', 'phai': 'right', 'phải': 'right',
           'unidentify_angle': 'unidentify_angle'}


def classify_camera_view(pose, *, action_start_frame=None, window_s=0.5,
                         min_frames=5, visibility_threshold=0.5, min_confidence=0.7):
    """Use shoulder/hip depth from the initial window, before the action if known.

    Image/world x points right, z grows away from the camera. In a frontal view,
    anatomical left is image-right. A nearer LEFT shoulder has smaller z and
    indicates a camera on the participant's LEFT. Mirrored inputs need review.
    """
    result = dict(label='unidentify_angle', label_vi=LABEL_VI['unidentify_angle'], confidence=0.0,
                  reliable=False, method='initial_world_torso_v2',
                  reason='', initial_frame_range=[0, 0], valid_frames=0,
                  yaw_deg=None, votes={'front': 0, 'left': 0, 'right': 0})
    world, visibility = pose.get('world'), pose.get('vis')
    if world is None or visibility is None:
        result['reason'] = 'Thiếu world 3D/visibility để nhận diện góc quay.'
        return result
    w, v = np.asarray(world), np.asarray(visibility)
    if w.ndim != 3 or w.shape[1:] != (13, 3) or v.shape != w.shape[:2]:
        result['reason'] = 'World/visibility sai kích thước.'
        return result
    fps = float(pose.get('fps', 0))
    if not np.isfinite(fps) or fps <= 0:
        result['reason'] = 'FPS không hợp lệ khi chọn cửa sổ tư thế đầu.'
        return result
    end = min(len(w), max(1, round(window_s * fps)))
    if action_start_frame is not None:
        end = min(end, max(0, action_start_frame))
    result['initial_frame_range'] = [0, end]
    # Shoulder and hip axes provide complementary estimates of the torso view.
    axes, masks = [], []
    for l, r in ((1, 2), (7, 8)):
        axis = w[:end, l] - w[:end, r]
        horizontal = np.hypot(axis[:, 0], axis[:, 2])
        valid = np.isfinite(axis).all(axis=1) & (horizontal > 1e-4)
        valid &= (v[:end, l] >= visibility_threshold) & (v[:end, r] >= visibility_threshold)
        axes.append(axis)
        masks.append(valid)
    angles = []
    rear = 0
    disagreement = 0
    for i in range(end):
        yaw = [float(np.degrees(np.arctan2(-a[i, 2], a[i, 0])))
               for a, mask in zip(axes, masks) if mask[i]]
        if not yaw:
            continue
        # Different torso axes can indicate a twisted guard stance or bad world.
        if len(yaw) == 2 and abs((yaw[0] - yaw[1] + 180) % 360 - 180) > 45:
            disagreement += 1
            continue
        if any(abs(a) > 135 for a in yaw):
            rear += 1
            continue
        angles.append(float(np.median(yaw)))
    result['valid_frames'] = len(angles)
    result['rear_frames'] = rear
    result['torso_disagreement_frames'] = disagreement
    if not angles:
        result['reason'] = 'Không đủ tư thế đầu hợp lệ; có thể quay từ sau, ảnh lật gương hoặc mất pose.'
        return result
    angles = np.asarray(angles)
    labels = np.where(angles > 45, 'left', np.where(angles < -45, 'right', 'front'))
    votes = {label: int(np.sum(labels == label)) for label in KNOWN_LABELS}
    winner = max(votes, key=votes.get)
    selected = angles[labels == winner]
    # Confidence falls near 45-degree boundaries, on unstable clips and missing frames.
    distance = np.minimum(np.abs(selected - 45), np.abs(selected + 45))
    distance = np.minimum(distance, 135 - np.abs(selected))
    strength = float(np.median(0.5 + 0.5 * np.clip(distance / 45, 0, 1)))
    confidence = strength * votes[winner] / max(end, 1)
    reliable = len(angles) >= min_frames and confidence >= min_confidence
    label = winner if reliable else 'unidentify_angle'
    result.update(label=label, label_vi=LABEL_VI[label], confidence=confidence,
                  candidate_label=winner,
                  yaw_deg=float(np.median(angles)), votes=votes,
                  reliable=reliable,
                  reason=('Ước lượng theo trục vai/hông 3D ở tư thế đầu.' if reliable else
                          'Góc ngoài nhóm chuẩn hoặc không đủ frame/độ tin cậy để xác định.'))
    return result
