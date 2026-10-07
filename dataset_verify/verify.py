"""Local batch verifier. Run: python -m dataset_verify --help."""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import importlib
import json
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

if __package__:
    from .camera_view import ALIASES as VIEW_ALIASES, classify_camera_view
    from .recognition import RecognitionBackend
else:
    from camera_view import ALIASES as VIEW_ALIASES, classify_camera_view
    from recognition import RecognitionBackend

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_SCRIPTS = ROOT / 'TKD-Kick3_Zenodo_Materials_20260526_153619/code/scripts'
DEFAULT_TASK = DEFAULT_SCRIPTS.parent.parent / 'models/pose_landmarker_full.task'
EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}
KICKS = {'front', 'roundhouse', 'axe', 'side', 'back', 'hook_kick', 'push', 'side_low', 'roundhouse_low'}
REQUIRED = ('clip_id', 'participant_id', 'session_id', 'source_video_id', 'technique',
            'side', 'camera_view', 'target_zone', 'start_stance', 'end_stance',
            'dataset_role', 'execution_variant', 'contact_mode', 'camera_setup_id',
            'take_number', 'protocol_version')
HUMAN_CHECKS = ('single_action_verified', 'technique_verified', 'side_verified',
                'full_body_verified', 'phases_verified', 'camera_fixed_verified',
                'realtime_verified', 'view_assessable_verified')
PHASES = ('action_start_frame', 'chamber_end_frame', 'apex_frame', 'strike_frame',
          'recovery_start_frame', 'action_end_frame')
FIELDS = ['video_path', *REQUIRED, 'split', *PHASES, 'chamber_na_reason',
          'guard', *HUMAN_CHECKS, 'reviewer_id', 'review_notes', 'capture_group_id',
          'coach_approved', 'rubric_version']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def clean(value):
    """Strict JSON: never emit NaN/Infinity from scorer or pose."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if hasattr(value, 'tolist'):
        return clean(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def issue(report, code, message, severity='review', **evidence):
    report['issues'].append(dict(code=code, severity=severity, message=message, evidence=clean(evidence)))


def finalize(report):
    severities = {it['severity'] for it in report['issues']}
    report['quality_status'] = 'rejected' if 'rejected' in severities else 'review' if 'review' in severities else 'accepted'
    report['approved'] = report['quality_status'] == 'accepted'
    report['scorable'] = 'yes' if report['approved'] else 'no' if report['quality_status'] == 'rejected' else 'review'
    return report


def discover(source, output):
    if source.is_file():
        if source.suffix.lower() not in EXTENSIONS:
            raise ValueError('Đầu vào phải là video hoặc thư mục video.')
        return [source]
    return sorted(p for p in source.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS
                  and not p.is_relative_to(output))


def load_manifest(path, source_root):
    if path is None:
        return {}
    records = {}
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or 'video_path' not in reader.fieldnames:
            raise ValueError('Manifest cần cột video_path.')
        for line, row in enumerate(reader, 2):
            row = {k: (v or '').strip() for k, v in row.items() if k is not None}
            if row.get('split') == 'validation':
                row['split'] = 'val'
            raw = row.get('video_path', '')
            if not raw:
                raise ValueError(f'Manifest dòng {line}: thiếu video_path.')
            p = Path(raw).expanduser()
            key = p.resolve() if p.is_absolute() else (source_root / p).resolve()
            if key in records:
                raise ValueError(f'Manifest trùng video_path tại dòng {line}: {raw}')
            records[key] = row
    return records


def leakage_issues(records):
    """Only use actual metadata; do not infer participant identity from filename."""
    errors = defaultdict(list)
    for field in ('participant_id', 'source_video_id', 'capture_group_id'):
        groups = defaultdict(list)
        for path, row in records.items():
            if row.get(field) and row.get('split'):
                groups[row[field]].append((path, row['split']))
        for identity, rows in groups.items():
            splits = sorted({s for _, s in rows})
            if len(splits) > 1:
                for path, _ in rows:
                    errors[path].append(f'{field}={identity} xuất hiện ở nhiều split: {splits}')
    ids = defaultdict(list)
    for path, row in records.items():
        if row.get('clip_id'):
            ids[row['clip_id']].append(path)
    for clip_id, paths in ids.items():
        if len(paths) > 1:
            for path in paths:
                errors[path].append(f'clip_id bị trùng: {clip_id}')
    return errors


def validate_metadata(row, report, config, frame_count=None, fps=None):
    for field in REQUIRED:
        if not row.get(field):
            issue(report, 'METADATA_MISSING', f'Thiếu metadata: {field}', field=field)
    if row.get('side') not in {'left', 'right', 'L', 'R', 'both', 'none', ''}:
        issue(report, 'SIDE_INVALID', 'side cần left/right/L/R/both/none.')
    if row.get('target_zone') not in {'low', 'mid', 'high', 'custom', ''}:
        issue(report, 'TARGET_INVALID', 'target_zone cần low/mid/high/custom.')
    if row.get('split', '') not in {'', 'train', 'val', 'validation', 'test'}:
        issue(report, 'SPLIT_INVALID', 'split cần train/val/validation/test hoặc để trống.')
    if row.get('guard', '') not in {'', 'chin', 'waist'}:
        issue(report, 'GUARD_INVALID', 'guard cần chin/waist.')
    if row.get('contact_mode') not in {'', 'air', 'target', 'contact', 'non_contact'}:
        issue(report, 'CONTACT_MODE_UNKNOWN', 'Cần kiểm tra contact_mode; dùng air/target/contact/non_contact.')
    if config['require_human_checks']:
        for field in HUMAN_CHECKS:
            if row.get(field, '').lower() not in {'yes', 'true', '1'}:
                issue(report, 'HUMAN_CHECK_PENDING', f'Chưa xác nhận: {field}', field=field)
        if not row.get('reviewer_id'):
            issue(report, 'REVIEWER_MISSING', 'Thiếu reviewer_id cho checklist xác nhận.')
    role = row.get('dataset_role', '')
    if role == 'demonstration_slow':
        issue(report, 'SLOW_DEMONSTRATION', 'Clip hướng dẫn chậm cần tập riêng; không dùng scoring tốc độ thực.')
    if role in {'reference', 'standard', 'technical_reference'}:
        if row.get('coach_approved', '').lower() not in {'yes', 'true', '1'}:
            issue(report, 'COACH_APPROVAL_MISSING', 'Mẫu chuẩn cần coach_approved=yes từ HLV.')
        if not row.get('rubric_version'):
            issue(report, 'RUBRIC_MISSING', 'Mẫu chuẩn cần rubric_version đã được HLV duyệt.')
    if frame_count is None:
        return {}
    marks = {}
    required_phases = {'action_start_frame', 'strike_frame', 'recovery_start_frame', 'action_end_frame'}
    if row.get('technique') in {'front', 'roundhouse', 'side'}:
        required_phases.add('chamber_end_frame')
    if row.get('technique') == 'axe':
        required_phases.add('apex_frame')
    for field in PHASES:
        raw = row.get(field, '')
        if not raw:
            if field in required_phases:
                issue(report, 'PHASE_MISSING', f'Thiếu mốc pha: {field}', field=field)
            continue
        try:
            v = int(raw)
            if not 0 <= v < frame_count:
                raise ValueError()
            marks[field] = v
        except ValueError:
            issue(report, 'PHASE_INVALID', f'{field} phải là frame nguyên 0..{frame_count-1}.', value=raw)
    ordered = [marks[f] for f in PHASES if f in marks]
    if ordered != sorted(ordered):
        issue(report, 'PHASE_ORDER', 'Mốc pha không đúng thứ tự chuẩn bị → đánh → thu → kết thúc.', marks=marks)
    a, b, c, d = [marks.get(f) for f in ('action_start_frame', 'strike_frame', 'recovery_start_frame', 'action_end_frame')]
    if all(v is not None for v in (a, b, c, d)) and not (a < b <= c < d):
        issue(report, 'PHASE_DURATION', 'Cần action_start < strike ≤ recovery_start < action_end.', marks=marks)
    if row.get('technique') == 'axe' and 'apex_frame' in marks and b is not None and marks['apex_frame'] >= b:
        issue(report, 'AXE_APEX', 'Đá chẻ cần apex_frame trước strike_frame; không gộp đỉnh nâng và pha chẻ.')
    if row.get('technique') == 'axe' and not row.get('chamber_end_frame') and not row.get('chamber_na_reason'):
        issue(report, 'CHAMBER_NA_REASON', 'Đá chẻ không có chamber riêng: ghi chamber_na_reason.')
    if fps and a is not None and d is not None:
        before, after = a / fps, (frame_count - 1 - d) / fps
        report['metrics']['hold_before_s'], report['metrics']['hold_after_s'] = before, after
        if min(before, after) + 1e-6 < config['min_hold_s']:
            issue(report, 'HOLD_TOO_SHORT', 'Thiếu khoảng giữ tư thế trước/sau động tác.', before_s=before, after_s=after,
                  minimum_s=config['min_hold_s'])
    return marks


def inspect_video(video, config, report):
    """Decode EVERY frame; quality measurements run on every frame at 360 px tall."""
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        cap.release()
        issue(report, 'VIDEO_OPEN_FAILED', 'Không mở/giải mã được video.', 'rejected')
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    blur, luma = [], []
    width = height = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if width and (w != width or h != height):
                issue(report, 'RESOLUTION_CHANGED', 'Độ phân giải thay đổi giữa clip.', 'rejected')
            width, height = w, h
            small = cv2.resize(frame, (max(1, round(w * 360 / h)), 360))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            blur.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
            luma.append(float(gray.mean()))
    finally:
        cap.release()
    count = len(blur)
    if count < 2:
        issue(report, 'VIDEO_EMPTY', 'Video không có đủ frame giải mã được.', 'rejected', frames=count)
        return None
    if not math.isfinite(fps) or fps <= 0:
        issue(report, 'FPS_INVALID', 'Không truy được FPS; không tự thay bằng 30 FPS.', 'rejected')
        return None
    if expected > 0 and count != expected:
        issue(report, 'DECODE_INCOMPLETE', 'Số frame giải mã khác header; kiểm tra file hỏng/cắt cụt.', 'rejected', expected=expected, decoded=count)
    blurry = float(np.mean(np.array(blur) < config['blur_laplacian_min']))
    dark = float(np.mean((np.array(luma) < config['min_luma']) | (np.array(luma) > config['max_luma'])))
    metrics = dict(fps=fps, width=width, height=height, frame_count=count, duration_s=count / fps,
                   capture_profile=f'{width}x{height}@{fps:.3f}', blurry_ratio=blurry,
                   bad_exposure_ratio=dark, blur_median=float(np.median(blur)))
    report['metrics'].update(metrics)
    if fps < config['min_fps']:
        issue(report, 'LOW_FPS', 'FPS dưới profile QC pilot.', actual=fps, minimum=config['min_fps'])
    if width <= height:
        issue(report, 'PORTRAIT_VIDEO', 'Khung hình chưa ở hướng ngang theo checklist quay.')
    if min(width, height) < config['min_short_edge_px']:
        issue(report, 'LOW_RESOLUTION', 'Độ phân giải dưới profile QC pilot.', width=width, height=height)
    if blurry > config['max_blurry_ratio']:
        issue(report, 'BLUR_REVIEW', 'Nhiều frame có dấu hiệu nhòe/ít chi tiết; cần xem người tập.', ratio=blurry)
    if dark > config['max_bad_exposure_ratio']:
        issue(report, 'EXPOSURE_REVIEW', 'Nhiều frame quá tối/sáng theo ngưỡng pilot.', ratio=dark)
    return metrics


def inspect_timestamps(video, metrics, config, report):
    ffprobe = shutil.which('ffprobe')
    if not ffprobe:
        issue(report, 'TIMEBASE_UNVERIFIED', 'Chưa có ffprobe để xác minh timestamp và FPS hằng định.',
              'review' if config['require_ffprobe'] else 'warning')
        return
    try:
        run = subprocess.run([ffprobe, '-v', 'error', '-select_streams', 'v:0', '-show_frames',
                              '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', str(video)],
                             capture_output=True, text=True, timeout=300, check=True)
        frames = json.loads(run.stdout).get('frames', [])
        times = [float(f['best_effort_timestamp_time']) for f in frames]
        if len(times) != metrics['frame_count'] or any(not math.isfinite(t) for t in times):
            raise ValueError('Thiếu timestamp hoặc số timestamp khác số frame giải mã.')
        intervals = [b - a for a, b in zip(times, times[1:])]
        nominal = 1 / metrics['fps']
        tolerance = max(config['timestamp_absolute_tolerance_s'], nominal * config['timestamp_relative_tolerance'])
        bad = [i for i, dt in enumerate(intervals) if dt <= 0 or abs(dt - nominal) > tolerance]
        report['metrics']['timestamps_verified'] = not bad
        report['metrics']['timestamp_irregular_intervals'] = len(bad)
        if bad:
            issue(report, 'VARIABLE_TIMEBASE', 'Timestamp không đều/mất frame; scorer dùng FPS cố định nên cần xử lý riêng.', frames=bad[:20])
    except (subprocess.SubprocessError, ValueError, KeyError) as exc:
        issue(report, 'TIMESTAMP_FAILED', f'Không xác minh được timestamp: {exc}', 'rejected')


def inspect_pose(pose, metrics, marks, action, config, report):
    import numpy as np
    k = np.asarray(pose['kpts'])
    v = np.asarray(pose['vis'])
    n = metrics['frame_count']
    if k.shape[0] == 0:
        issue(report, 'NO_PERSON', 'Không phát hiện được người tập trong video.', 'rejected')
        return False
    if k.shape != (n, 13, 2) or v.shape != (n, 13):
        issue(report, 'POSE_SHAPE', 'Pose phải giữ toàn bộ frame gốc: kpts(T,13,2), vis(T,13).', 'rejected', kpts=k.shape, vis=v.shape, decoded=n)
        return False
    if np.any(~np.isfinite(v)) or np.any((v < 0) | (v > 1)):
        issue(report, 'POSE_VIS_INVALID', 'Visibility chứa giá trị không hợp lệ.', 'rejected')
        return False
    if not np.isclose(float(pose['fps']), metrics['fps'], rtol=0.001):
        issue(report, 'POSE_FPS_MISMATCH', 'FPS pose không khớp video gốc.', 'rejected')
        return False
    if int(pose.get('invalid_for_scoring', 0)):
        issue(report, 'PERSON_AMBIGUOUS', 'Extractor không xác định chắc người chính (invalid_for_scoring=1).')
    if int(pose.get('multi_person', 0)):
        issue(report, 'MULTI_PERSON', 'Có nhiều người; cần kiểm tra người được theo dõi và che khuất.')
    finite = np.isfinite(k).all(axis=2)
    visible = finite & (v >= config['visibility_threshold'])
    margin = config['frame_margin_ratio']
    inside = finite & (k[:, :, 0] >= metrics['width'] * margin) & (k[:, :, 0] < metrics['width'] * (1-margin))
    inside &= (k[:, :, 1] >= metrics['height'] * margin) & (k[:, :, 1] < metrics['height'] * (1-margin))
    full_ratio = float(np.mean((inside & visible).all(axis=1)))
    ratios = (visible & inside).mean(axis=0)
    report['metrics']['joint_visible_ratios'] = ratios.tolist()
    report['metrics']['full_body_ratio'] = full_ratio
    if full_ratio < config['min_full_body_ratio']:
        issue(report, 'BODY_CROPPED_OR_OCCLUDED', 'Toàn thân chưa quan sát đủ trong clip (pose 13 khớp).', ratio=full_ratio,
              frames=np.flatnonzero(~(inside & visible).all(axis=1))[:20].tolist())
    missing = np.flatnonzero(ratios < config['min_joint_visible_ratio']).tolist()
    if missing:
        issue(report, 'JOINT_VISIBILITY', 'Khớp mất/che/ra biên quá nhiều theo ngưỡng pilot.', joint_indices=missing)
    if 'strike_frame' in marks:
        center = marks['strike_frame']
        radius = max(1, round(metrics['fps'] * config['strike_radius_s']))
        s, e = max(0, center-radius), min(n, center+radius+1)
        ratio = float(np.mean((visible & inside)[s:e]))
        report['metrics']['strike_visible_ratio'] = ratio
        if ratio < config['min_strike_visible_ratio'] or not (visible & inside)[center].all():
            issue(report, 'STRIKE_NOT_CLEAR', 'Khớp quan trọng chưa rõ ở frame/vùng pha đánh.', frame=center, ratio=ratio)
    world = pose.get('world')
    has_world = world is not None and np.asarray(world).shape == (n, 13, 3)
    report['metrics']['pose_has_world'] = bool(has_world)
    if action not in KICKS:
        if not has_world:
            issue(report, 'WORLD_REQUIRED', 'Luật này cần world(T,13,3); không thể dùng pose 2D thay thế.', 'rejected')
            return False
        world_ok = np.isfinite(world).all(axis=2)
        if not world_ok[visible].all():
            issue(report, 'WORLD_MISSING', 'World 3D thiếu ở khớp đang thấy rõ; điểm 3D chưa tin cậy.')
    return True


class ScoringBackend:
    def __init__(self, scripts, task):
        if not (scripts / 'action_scorers.py').is_file():
            raise ValueError(f'Không tìm được code scorer v0.2: {scripts}')
        sys.path.insert(0, str(scripts))
        self.pose_module = importlib.import_module('action_pose')
        self.scorers = importlib.import_module('action_scorers')
        for plugin in sorted(scripts.glob('scorers_*.py')):
            importlib.import_module(plugin.stem)
        self.task = task
        self.fingerprint = hashlib.sha256(''.join(sha256(p) for p in sorted(scripts.glob('*.py'))).encode()).hexdigest()

    def extract(self, video):
        if not self.task.is_file():
            raise FileNotFoundError(f'Thiếu model MediaPipe: {self.task}')
        return self.pose_module.extract_pose(video, self.task, num_poses=4, select='limbs')

    def score(self, action, pose, guard):
        # Do not force side: compare detected side with the supplied label.
        result = self.scorers.score_action(action, pose, guard=guard, kick_window='none', fix_leg=True)
        return dataclasses.asdict(result)


def inspect_score(result, row, report):
    report['scoring'] = result
    if not result['valid']:
        issue(report, 'SCORER_INVALID', f"Scorer không chấm được: {result.get('reason', '')}")
    for warning in result.get('warnings', []):
        issue(report, 'SCORER_WARNING', warning)
    raw = result.get('raw', {})
    detected = raw.get('kick_side') or raw.get('side')
    report['metrics']['detected_side'] = detected
    expected = {'left': 'L', 'right': 'R', 'L': 'L', 'R': 'R'}.get(row.get('side'))
    if expected and detected is None:
        issue(report, 'SIDE_UNVERIFIED', 'Scorer không trả detected_side; cần kiểm tra nhãn bên.')
    elif expected and detected != expected:
        issue(report, 'SIDE_MISMATCH', 'Bên scorer phát hiện khác nhãn metadata.', expected=expected, detected=detected)
    items = result.get('acc_items', []) + result.get('expr_items', [])
    na = [it.get('code') for it in items if not it.get('assessable', True) or it.get('got') is None]
    if na:
        issue(report, 'CRITERION_NA', 'Có tiêu chí không đủ quan sát; cần review/NA.', criteria=na)
    if result.get('valid') and (result.get('total') is None or not math.isfinite(float(result['total']))):
        issue(report, 'SCORE_NONFINITE', 'Scorer trả tổng điểm không hữu hạn.', 'rejected')
    # No minimum technique score: clear mistakes are useful quality-training samples.


def route(video, relative, output, report):
    """Immutable run destinations prevent stale copies when approval changes."""
    folder = output / ('approved' if report['approved'] else 'unapproved') / relative.parent
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / relative.name
    if destination.exists():
        raise FileExistsError(f'Không ghi đè video đã tồn tại: {destination}')
    shutil.copy2(video, destination)
    report['output_video'] = str(destination)
    write_json(destination.with_name(destination.name + '.qc.json'), report)
    if not report['approved']:
        lines = [f'VIDEO: {relative}', f"TRẠNG THÁI: {report['quality_status']}",
                 f"GÓC QUAY TỰ ĐỘNG: {report['metadata'].get('camera_view', 'chưa xác định')}",
                 f"ĐỘNG TÁC: {report['metadata'].get('technique', 'chưa xác định')}",
                 'KẾT LUẬN: chưa approved; giữ video gốc.', 'LÝ DO:']
        for it in report['issues']:
            if it['severity'] != 'warning':
                lines.append(f"- [{it['code']}] {it['message']}")
                if it['evidence']:
                    lines.append('  ' + json.dumps(it['evidence'], ensure_ascii=False))
        destination.with_name(destination.name + '.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def new_report(video, row, config):
    return dict(video=str(video), metadata=row, policy=config, issues=[], metrics={}, scoring=None)


def assign_camera_view(pose, row, config, report):
    declared = row.get('camera_view', '')
    try:
        start = int(row['action_start_frame']) if row.get('action_start_frame') else None
    except (ValueError, TypeError):
        start = None
    view = classify_camera_view(pose, action_start_frame=start,
                                window_s=config['camera_view_window_s'],
                                min_frames=config['camera_view_min_frames'],
                                visibility_threshold=config['visibility_threshold'],
                                min_confidence=config['camera_view_min_confidence'])
    view['declared_view'] = declared
    view['reliable'] = view['reliable'] and view['confidence'] >= config['camera_view_min_confidence']
    report['camera_view_prediction'] = view
    row['camera_view'] = view['label']
    row['camera_view_confidence'] = view['confidence']
    row['camera_view_source'] = 'auto'
    if not view['reliable']:
        issue(report, 'CAMERA_VIEW_UNIDENTIFIED', 'Gán unidentify_angle; tiếp tục verify chất lượng theo các điều kiện còn lại.',
              severity='warning', prediction=view)
    canonical_declared = VIEW_ALIASES.get(declared.strip().lower())
    if view['reliable'] and declared and canonical_declared != view['label']:
        issue(report, 'CAMERA_VIEW_CONFLICT', 'Góc quay dự đoán khác/không khớp nhãn đã khai báo.',
              declared=declared, predicted=view['label'])
    return view


def assign_action(pose, row, recognizer, canonical, config, report):
    declared = canonical(row.get('technique', ''))
    report['technique_declared'] = declared
    row['technique_source'] = 'declared' if declared else 'auto'
    if recognizer is None:
        report['recognition'] = dict(used=False, reason='Nhận diện tự động đã tắt.')
        row['technique'] = declared
        return declared
    supported = {canonical(c) for c in recognizer.classes}
    if declared and declared not in supported:
        report['recognition'] = dict(used=False, classes=recognizer.classes,
                                    reason='Động tác khai báo nằm ngoài các lớp của checkpoint; dùng nhãn khai báo để scoring.')
        row['technique'] = declared
        return declared
    try:
        prediction = recognizer.predict(pose)
        candidate = canonical(prediction['predicted_action'])
        confident = (prediction['confidence'] >= config['action_min_confidence'] and
                     prediction['margin'] >= config['action_min_margin'])
        prediction['reliable'] = confident
        report['recognition'] = prediction
        row['technique_confidence'] = prediction['confidence']
        row['technique_predicted'] = candidate
        if not confident:
            issue(report, 'ACTION_RECOGNITION_UNCERTAIN', 'Model chưa đủ chắc chắn về động tác; cần xem lại nhãn.',
                  predicted=candidate, confidence=prediction['confidence'], margin=prediction['margin'])
        if declared and confident and declared != candidate:
            issue(report, 'ACTION_LABEL_MISMATCH', 'Nhãn model khác technique đã khai báo; giữ nhãn khai báo và yêu cầu review.',
                  declared=declared, predicted=candidate)
        row['technique'] = declared or (candidate if confident else 'unidentify_action')
    except Exception as exc:
        report['recognition'] = dict(used=False, reason=f'{type(exc).__name__}: {exc}')
        issue(report, 'ACTION_RECOGNITION_FAILED', f'Không nhận diện được động tác: {exc}')
        row['technique'] = declared or 'unidentify_action'
    return row['technique']


def run(args):
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.exists():
        raise ValueError(f'Đầu vào không tồn tại: {source}')
    base = source if source.is_dir() else source.parent
    if base.is_relative_to(output) or source == output:
        raise ValueError('Thư mục output không được bao chứa nguồn hoặc trùng nguồn.')
    videos = discover(source, output)
    if not videos:
        raise ValueError('Không tìm thấy video đầu vào.')
    if args.init_manifest:
        path = args.init_manifest.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            for i, video in enumerate(videos, 1):
                writer.writerow({'video_path': video.relative_to(base).as_posix(), 'clip_id': f'CLIP_{i:05d}',
                                 'technique': args.action or ''})
        print(f'Đã tạo manifest {path} ({len(videos)} clip); điền metadata và checklist trước khi duyệt.')
        return 0
    config = json.loads(args.config.read_text(encoding='utf-8'))
    validate_config(config)
    records = load_manifest(args.manifest, base)
    leaks = leakage_issues(records)
    # Preflight environment failures abort before creating misleading QC rejections.
    backend = ScoringBackend(args.scripts.resolve(), args.task.resolve())
    if not backend.task.is_file():
        raise FileNotFoundError(f'Thiếu model MediaPipe: {backend.task}')
    if args.action:
        args.action = backend.scorers.canonical(args.action)
        if args.action not in backend.scorers.SCORERS:
            raise ValueError(f'Không có luật scoring cho {args.action}')
    recognizer = None if args.no_recognition else RecognitionBackend(args.scripts.resolve(), args.ckpt, args.device)
    checkpoint_info = ([dict(path=str(p.resolve()), sha256=sha256(p)) for p in recognizer.paths]
                       if recognizer else [])
    for folder in ('approved', 'unapproved', 'reports', 'pose_cache'):
        target = output / folder
        if target.exists() and any(p.name != '.gitkeep' for p in target.iterdir()):
            raise ValueError(f'Output đã có kết quả: {target}. Dùng --output mới cho lần chạy tiếp để tránh bản approved cũ.')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'approved').mkdir(exist_ok=True)
    (output / 'unapproved').mkdir(exist_ok=True)
    cache = output / 'pose_cache'
    cache.mkdir(exist_ok=True)
    reports = []
    hashes = defaultdict(list)
    for i, video in enumerate(videos, 1):
        print(f'[{i}/{len(videos)}] {video.relative_to(base)}', flush=True)
        row = dict(records.get(video, {}))
        row.setdefault('video_path', video.relative_to(base).as_posix())
        if args.action and not row.get('technique'):
            row['technique'] = args.action
        action = backend.scorers.canonical(row.get('technique', ''))
        row['technique'] = action
        report = new_report(video, row, config)
        report['source_sha256'] = sha256(video)
        report['scorer_source_sha256'] = backend.fingerprint
        report['model_sha256'] = sha256(backend.task) if backend.task.is_file() else None
        report['recognition_checkpoints'] = checkpoint_info
        hashes[report['source_sha256']].append((video, row))
        for message in leaks.get(video, []):
            issue(report, 'DATASET_LEAKAGE', message, 'rejected')
        # Extract once and label the initial camera view BEFORE quality verification.
        pose = None
        try:
            pose = backend.extract(video)
            assign_camera_view(pose, row, config, report)
        except (FileNotFoundError, ImportError):
            raise
        except Exception as exc:
            assign_camera_view({}, row, config, report)
            issue(report, 'POSE_EXTRACTION_FAILED', f'{type(exc).__name__}: {exc}', 'rejected')
        action = assign_action(pose or {}, row, recognizer, backend.scorers.canonical, config, report)
        metrics = inspect_video(video, config, report)
        marks = validate_metadata(row, report, config,
                                  metrics['frame_count'] if metrics else None, metrics['fps'] if metrics else None)
        if args.action and action != args.action:
            issue(report, 'ACTION_CONFLICT', '--action khác technique trong manifest; không tự đổi nhãn.', 'rejected')
        if action not in backend.scorers.SCORERS:
            issue(report, 'ACTION_UNKNOWN', f'Chưa xác định động tác có luật scoring: {action!r}.',
                  'review' if action in {'', 'unidentify_action'} else 'rejected')
        elif metrics and pose is not None:
            inspect_timestamps(video, metrics, config, report)
            try:
                if inspect_pose(pose, metrics, marks, action, config, report):
                    import numpy as np
                    pose_path = cache / (report['source_sha256'] + '.npz')
                    np.savez_compressed(pose_path, **pose)
                    report['pose_path'] = str(pose_path)
                    inspect_score(backend.score(action, pose, row.get('guard') or 'chin'), row, report)
            except (FileNotFoundError, ImportError):
                raise  # setup errors should not classify every input as bad data
            except Exception as exc:
                issue(report, 'POSE_OR_SCORING_FAILED', f'{type(exc).__name__}: {exc}', 'rejected')
        reports.append((video, report))
    # Mark every duplicate, including the first occurrence, before routing.
    for digest, group in hashes.items():
        if len(group) > 1:
            paths = {p for p, _ in group}
            splits = {r.get('split') for _, r in group if r.get('split')}
            for video, report in reports:
                if video in paths:
                    issue(report, 'DUPLICATE_VIDEO', 'Video trùng nội dung SHA256; chọn một bản trước khi nhập dataset.',
                          'rejected' if len(splits) > 1 else 'review', paths=[str(p) for p in sorted(paths)])
    for video, report in reports:
        finalize(report)
        route(video, video.relative_to(base), output, report)
    report_dir = output / 'reports'
    report_dir.mkdir(exist_ok=True)
    write_json(report_dir / 'results.json', [r for _, r in reports])
    with (report_dir / 'labeled_manifest.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        fields = list(dict.fromkeys([*FIELDS, *[k for _, r in reports for k in r['metadata']],
                                     'camera_view_declared', 'technique_declared']))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for _, r in reports:
            writer.writerow({**r['metadata'], 'camera_view_declared': r['camera_view_prediction']['declared_view'],
                             'technique_declared': r['technique_declared']})
    with (report_dir / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['video', 'output_video', 'technique', 'technique_predicted', 'technique_confidence', 'camera_view', 'camera_view_confidence', 'approved', 'quality_status', 'scorable', 'total10', 'reason_codes'])
        writer.writeheader()
        for _, r in reports:
            writer.writerow({**{k: r[k] for k in ('video', 'output_video', 'approved', 'quality_status', 'scorable')},
                             'total10': (r['scoring'] or {}).get('total'),
                             'camera_view': r['metadata'].get('camera_view'),
                             'camera_view_confidence': r['metadata'].get('camera_view_confidence'),
                             **{k: r['metadata'].get(k) for k in ('technique', 'technique_predicted', 'technique_confidence')},
                             'reason_codes': ';'.join(it['code'] for it in r['issues'] if it['severity'] != 'warning')})
    approved = sum(r['approved'] for _, r in reports)
    print(f'Hoàn tất: approved={approved}, unapproved={len(reports)-approved}. Kết quả: {output}')
    return 0


def validate_config(config):
    default = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
    if set(config) != set(default):
        raise ValueError(f'Config phải có đúng các khóa: {sorted(default)}')
    for key, value in config.items():
        if key == 'policy_version':
            if not isinstance(value, str) or not value.strip():
                raise ValueError('policy_version không được trống.')
        elif key.startswith('require_'):
            if not isinstance(value, bool):
                raise ValueError(f'{key} phải là boolean.')
        elif not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError(f'{key} phải là số hữu hạn không âm.')
        elif ('ratio' in key or key in {'visibility_threshold', 'camera_view_min_confidence', 'action_min_confidence', 'action_min_margin'}) and value > 1:
            raise ValueError(f'{key} phải trong [0,1].')
    if config['frame_margin_ratio'] >= 0.5 or config['min_fps'] <= 0 or config['min_luma'] >= config['max_luma']:
        raise ValueError('Config có margin/FPS/độ sáng không hợp lệ.')
    if config['camera_view_window_s'] <= 0 or config['camera_view_min_frames'] < 1 or not float(config['camera_view_min_frames']).is_integer():
        raise ValueError('Cửa sổ nhận diện góc phải dương; số frame tối thiểu phải là số nguyên ≥1.')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Kiểm tra video dataset theo workbook scoring v0.2; copy vào approved/unapproved và ghi lý do tiếng Việt.')
    parser.add_argument('--input', type=Path, required=True, help='Video hoặc thư mục video (quét đệ quy).')
    parser.add_argument('--output', type=Path, default=HERE, help='Mặc định dataset_verify/; các thư mục kết quả phải mới/rỗng. Nguồn luôn được giữ nguyên.')
    parser.add_argument('--manifest', type=Path, help='CSV UTF-8; video_path tương đối với --input (hoặc thư mục chứa video đơn).')
    parser.add_argument('--init-manifest', type=Path, help='Chỉ tạo CSV mẫu từ video, không chạy QC.')
    parser.add_argument('--action', help='Nhãn khai báo khi CSV thiếu technique; model đối chiếu nếu lớp được hỗ trợ.')
    parser.add_argument('--ckpt', help='Checkpoint .pt, danh sách phân cách dấu phẩy, glob hoặc thư mục; mặc định ensemble LSTM 3 lớp.')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--no-recognition', action='store_true', help='Tắt model nhận diện, chỉ dùng technique/--action đã khai báo.')
    parser.add_argument('--config', type=Path, default=HERE / 'config.json')
    parser.add_argument('--scripts', type=Path, default=DEFAULT_SCRIPTS, help='Thư mục code/scripts chứa action_scorers.py v0.2.')
    parser.add_argument('--task', type=Path, default=DEFAULT_TASK, help='Model pose_landmarker_full.task.')
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (ValueError, OSError, ImportError, KeyError, RuntimeError) as exc:
        print(f'LỖI: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
