import json
import unittest
from pathlib import Path

import numpy as np

from dataset_verify.camera_view import classify_camera_view
from dataset_verify import verify as q


def make_pose(yaw, count=90):
    world = np.zeros((count, 13, 3))
    yaw = np.broadcast_to(np.asarray(yaw), (count,))
    axis = np.stack([.4 * np.cos(np.radians(yaw)), np.zeros(count),
                     -.4 * np.sin(np.radians(yaw))], axis=1)
    for left, right in ((1, 2), (7, 8)):
        world[:, left] = axis / 2
        world[:, right] = -axis / 2
    return {'world': world, 'vis': np.ones((count, 13)), 'fps': 30.}


class CameraViewTests(unittest.TestCase):
    def test_three_views_and_anatomical_sign(self):
        for yaw, label in [(0, 'front'), (90, 'left'), (-90, 'right')]:
            with self.subTest(label=label):
                result = classify_camera_view(make_pose(yaw))
                self.assertEqual(result['label'], label)
                self.assertTrue(result['reliable'])
                self.assertAlmostEqual(result['confidence'], 1)

    def test_action_rotation_does_not_change_initial_label(self):
        yaw = np.r_[np.zeros(15), np.full(75, -90)]
        result = classify_camera_view(make_pose(yaw), action_start_frame=15)
        self.assertEqual(result['label'], 'front')
        self.assertEqual(result['initial_frame_range'], [0, 15])

    def test_no_world_angle_does_not_block_quality_approval(self):
        config = json.loads((q.HERE / 'config.json').read_text())
        row = {}
        report = q.new_report(Path('a.mp4'), row, config)
        q.assign_camera_view({}, row, config, report)
        self.assertEqual(row['camera_view'], 'unidentify_angle')
        self.assertFalse(report['camera_view_prediction']['reliable'])
        self.assertTrue(q.finalize(report)['approved'])

    def test_boundary_view_has_low_confidence(self):
        for yaw in [44, 46, -44, -46]:
            self.assertLess(classify_camera_view(make_pose(yaw))['confidence'], .7)
            self.assertEqual(classify_camera_view(make_pose(yaw))['label'], 'unidentify_angle')

    def test_rear_not_mislabeled_with_confidence(self):
        result = classify_camera_view(make_pose(180))
        self.assertFalse(result['reliable'])
        self.assertEqual(result['confidence'], 0)
        self.assertEqual(result['label'], 'unidentify_angle')

    def test_unknown_does_not_conflict_with_declared_angle(self):
        config = json.loads((q.HERE / 'config.json').read_text())
        row = {'camera_view': 'front'}
        report = q.new_report(Path('a.mp4'), row, config)
        q.assign_camera_view(make_pose(180), row, config, report)
        self.assertEqual(row['camera_view'], 'unidentify_angle')
        self.assertEqual(report['camera_view_prediction']['declared_view'], 'front')
        self.assertTrue(q.finalize(report)['approved'])

    def test_near_rear_boundary_is_unidentified(self):
        for yaw in (134, -134, 150, -150):
            self.assertEqual(classify_camera_view(make_pose(yaw))['label'], 'unidentify_angle')

    def test_visibility_and_short_initial_segment(self):
        pose = make_pose(0)
        pose['vis'][:15] = 0
        self.assertFalse(classify_camera_view(pose)['reliable'])
        self.assertFalse(classify_camera_view(make_pose(0), action_start_frame=2)['reliable'])

    def test_declared_view_conflict_and_autofill(self):
        config = json.loads((q.HERE / 'config.json').read_text())
        row = {'camera_view': 'side_left'}
        report = q.new_report(Path('a.mp4'), row, config)
        q.assign_camera_view(make_pose(-90), row, config, report)
        self.assertEqual(row['camera_view'], 'right')
        self.assertEqual(report['camera_view_prediction']['declared_view'], 'side_left')
        self.assertIn('CAMERA_VIEW_CONFLICT', {i['code'] for i in report['issues']})

    def test_torso_twist_needs_review(self):
        pose = make_pose(90)
        pose['world'][:, 7:9] = make_pose(-90)['world'][:, 7:9]
        result = classify_camera_view(pose)
        self.assertFalse(result['reliable'])
        self.assertGreater(result['torso_disagreement_frames'], 0)


if __name__ == '__main__':
    unittest.main()
