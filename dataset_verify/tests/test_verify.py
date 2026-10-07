import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from dataset_verify import verify as q


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((q.HERE / 'config.json').read_text())
        self.row = {f: 'recorded' for f in q.REQUIRED}
        self.row.update(technique='front', camera_view='front', side='left', target_zone='mid', contact_mode='air',
                        dataset_role='recognition', guard='chin', reviewer_id='HLV01',
                        action_start_frame='15', chamber_end_frame='25', strike_frame='35',
                        recovery_start_frame='40', action_end_frame='74')
        self.row.update({f: 'yes' for f in q.HUMAN_CHECKS})
        self.metrics = dict(frame_count=90, fps=30., width=1280, height=720)
        self.pose = dict(kpts=np.full((90, 13, 2), 200., dtype=np.float32),
                         vis=np.ones((90, 13), dtype=np.float32), fps=30.,
                         world=np.zeros((90, 13, 3), dtype=np.float32),
                         invalid_for_scoring=0, multi_person=0)
        self.pose['world'][:, [1, 7], 0] = 0.2
        self.pose['world'][:, [2, 8], 0] = -0.2
        self.report = q.new_report(Path('video.mp4'), self.row, self.config)

    def codes(self):
        return {i['code'] for i in self.report['issues']}

    def test_complete_metadata(self):
        marks = q.validate_metadata(self.row, self.report, self.config, 90, 30.)
        self.assertEqual(marks['strike_frame'], 35)
        self.assertTrue(q.finalize(self.report)['approved'])

    def test_no_manifest_cannot_approve(self):
        q.validate_metadata({}, self.report, self.config, 90, 30.)
        self.assertIn('METADATA_MISSING', self.codes())
        self.assertIn('HUMAN_CHECK_PENDING', self.codes())
        self.assertFalse(q.finalize(self.report)['approved'])

    def test_impossible_phase_order_and_missing_hold(self):
        self.row.update(strike_frame='10', action_start_frame='0')
        q.validate_metadata(self.row, self.report, self.config, 90, 30.)
        self.assertIn('PHASE_ORDER', self.codes())
        self.assertIn('HOLD_TOO_SHORT', self.codes())

    def test_axe_apex_is_not_strike(self):
        self.row.update(technique='axe', apex_frame='35', chamber_end_frame='', chamber_na_reason='Không có chamber riêng')
        q.validate_metadata(self.row, self.report, self.config, 90, 30.)
        self.assertIn('AXE_APEX', self.codes())

    def test_world_required_for_hook(self):
        del self.pose['world']
        self.assertFalse(q.inspect_pose(self.pose, self.metrics, {}, 'hook', self.config, self.report))
        self.assertIn('WORLD_REQUIRED', self.codes())

    def test_pose_keeps_original_frame_count(self):
        self.pose['kpts'] = self.pose['kpts'][:70]
        self.assertFalse(q.inspect_pose(self.pose, self.metrics, {}, 'front', self.config, self.report))
        self.assertIn('POSE_SHAPE', self.codes())

    def test_no_person_rejected(self):
        self.pose['kpts'] = np.zeros((0, 13, 2))
        self.pose['vis'] = np.zeros((0, 13))
        self.assertFalse(q.inspect_pose(self.pose, self.metrics, {}, 'front', self.config, self.report))
        self.assertIn('NO_PERSON', self.codes())

    def test_nan_and_cropped_strike(self):
        self.pose['kpts'][35, 11] = np.nan
        self.pose['kpts'][35, 5] = [1279, 400]
        q.inspect_pose(self.pose, self.metrics, {'strike_frame': 35}, 'front', self.config, self.report)
        self.assertIn('STRIKE_NOT_CLEAR', self.codes())

    def test_ambiguous_person(self):
        self.pose['invalid_for_scoring'] = 1
        q.inspect_pose(self.pose, self.metrics, {}, 'front', self.config, self.report)
        self.assertIn('PERSON_AMBIGUOUS', self.codes())
        self.assertFalse(q.finalize(self.report)['approved'])

    def test_bad_technique_is_valid_training_data(self):
        q.inspect_score(dict(valid=True, total=0.2, raw={'kick_side': 'L'}, warnings=[],
                             acc_items=[{'code': 'FR_A1', 'assessable': True, 'got': 0.}], expr_items=[]),
                        self.row, self.report)
        self.assertTrue(q.finalize(self.report)['approved'])

    def test_side_and_na_are_review(self):
        q.inspect_score(dict(valid=True, total=8., raw={'kick_side': 'R'}, warnings=[],
                             acc_items=[{'code': 'FR_A1', 'assessable': False, 'got': None}], expr_items=[]),
                        self.row, self.report)
        self.assertEqual({'SIDE_MISMATCH', 'CRITERION_NA'}, self.codes())

    def test_split_leakage_marks_every_clip(self):
        paths = [Path('a.mp4'), Path('b.mp4')]
        rows = {p: dict(participant_id='P1', split=s) for p, s in zip(paths, ['train', 'test'])}
        self.assertEqual(set(q.leakage_issues(rows)), set(paths))

    def test_nonfinite_score_never_approved(self):
        q.inspect_score(dict(valid=True, total=float('nan'), raw={'side': 'L'}), self.row, self.report)
        self.assertEqual(q.finalize(self.report)['quality_status'], 'rejected')

    def test_route_preserves_sources_and_disambiguates_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for suffix in ('.mp4', '.mov'):
                src = root / ('same' + suffix)
                src.write_bytes(b'original')
                report = q.new_report(src, {}, self.config)
                q.issue(report, 'TEST', 'Lý do tiếng Việt')
                q.route(src, Path('front') / src.name, root / 'out', q.finalize(report))
                dst = root / 'out/unapproved/front' / src.name
                self.assertEqual(src.read_bytes(), dst.read_bytes())
                self.assertIn('Lý do tiếng Việt', dst.with_name(dst.name + '.txt').read_text())
                self.assertTrue(dst.with_name(dst.name + '.qc.json').exists())

    def test_irregular_timestamps_fail_review(self):
        completed = type('Process', (), {'stdout': json.dumps({'frames': [
            {'best_effort_timestamp_time': str(t)} for t in [0, .033333, .2]]})})()
        with patch.object(q.shutil, 'which', return_value='/bin/ffprobe'), patch.object(q.subprocess, 'run', return_value=completed):
            q.inspect_timestamps(Path('video.mp4'), dict(frame_count=3, fps=30), self.config, self.report)
        self.assertIn('VARIABLE_TIMEBASE', self.codes())

    def test_corrupt_video_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / 'bad.mp4'
            video.write_bytes(b'not a video')
            self.assertIsNone(q.inspect_video(video, self.config, self.report))
        self.assertEqual(q.finalize(self.report)['quality_status'], 'rejected')

    def test_batch_routes_approved_and_rejected_and_no_stale_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'input'
            source.mkdir()
            rng = np.random.default_rng(10)
            good = source / 'good.mp4'
            writer = cv2.VideoWriter(str(good), cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280, 720))
            self.assertTrue(writer.isOpened())
            for _ in range(90):
                writer.write(rng.integers(40, 215, (720, 1280, 3), dtype=np.uint8))
            writer.release()
            (source / 'corrupt.mp4').write_bytes(b'corrupt')
            (source / 'corrupt_copy.mp4').write_bytes(b'corrupt')
            manifest = root / 'manifest.csv'
            with manifest.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=q.FIELDS)
                writer.writeheader()
                writer.writerow(dict(self.row, video_path='good.mp4', technique=''))
            backend = type('Backend', (), {})()
            backend.scorers = type('Scorers', (), {'canonical': staticmethod(lambda x: x), 'SCORERS': {'front': None}})()
            backend.fingerprint, backend.task = 'test-backend', root / 'model.task'
            backend.task.write_bytes(b'test model fixture')
            backend.extract = lambda _: self.pose
            backend.score = lambda *_: dict(valid=True, total=.2, raw={'kick_side': 'L'}, warnings=[], acc_items=[], expr_items=[])
            recognizer = type('Recognizer', (), {'classes': ['front', 'roundhouse', 'axe'], 'paths': []})()
            recognizer.predict = lambda _: dict(used=True, model='test model', classes=recognizer.classes,
                                               predicted_action='front', confidence=.95, margin=.9,
                                               probabilities={'front': .95, 'roundhouse': .03, 'axe': .02}, window=[0, 89, 'ankle'])
            output = root / 'out'
            args = ['--input', str(source), '--output', str(output), '--manifest', str(manifest)]
            real_inspect = q.inspect_video
            def inspect_after_label(video, config, report):
                self.assertIn('camera_view_prediction', report)
                self.assertIn('recognition', report)
                self.assertIn(report['metadata']['camera_view'], {'front', 'left', 'right', 'unidentify_angle'})
                return real_inspect(video, config, report)
            with patch.object(q, 'ScoringBackend', return_value=backend), patch.object(q, 'RecognitionBackend', return_value=recognizer), patch.object(q, 'inspect_timestamps'), patch.object(q, 'inspect_video', side_effect=inspect_after_label):
                self.assertEqual(q.main(args), 0)
                self.assertEqual(q.main(args), 2)
            self.assertTrue((output / 'approved/good.mp4').is_file())
            self.assertTrue((output / 'unapproved/corrupt.mp4.txt').is_file())
            data = json.loads((output / 'reports/results.json').read_text())
            self.assertEqual(sum(r['approved'] for r in data), 1)
            good_result = next(r for r in data if r['approved'])
            self.assertEqual(good_result['metadata']['camera_view'], 'front')
            self.assertEqual(good_result['metadata']['technique'], 'front')
            self.assertEqual(good_result['metadata']['technique_source'], 'auto')
            with (output / 'reports/labeled_manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
                labeled = list(csv.DictReader(stream))
            self.assertEqual(len(labeled), 3)
            self.assertEqual(labeled[-1]['camera_view_source'], 'auto')
            duplicates = [r for r in data if any(it['code'] == 'DUPLICATE_VIDEO' for it in r['issues'])]
            self.assertEqual(len(duplicates), 2)
            self.assertTrue(good.exists())
            # A rear view gets the fallback label and still reaches quality scoring/routing.
            self.pose['world'] *= -1
            unknown_output = root / 'unknown_out'
            with patch.object(q, 'ScoringBackend', return_value=backend), patch.object(q, 'RecognitionBackend', return_value=recognizer), patch.object(q, 'inspect_timestamps'):
                self.assertEqual(q.main(['--input', str(good), '--output', str(unknown_output),
                                         '--manifest', str(manifest), '--action', 'front']), 0)
            unknown_results = json.loads((unknown_output / 'reports/results.json').read_text())
            self.assertEqual(unknown_results[0]['metadata']['camera_view'], 'unidentify_angle')
            self.assertTrue(unknown_results[0]['approved'])
            self.assertIsNotNone(unknown_results[0]['scoring'])
            self.assertTrue((unknown_output / 'approved/good.mp4').exists())


if __name__ == '__main__':
    unittest.main()
