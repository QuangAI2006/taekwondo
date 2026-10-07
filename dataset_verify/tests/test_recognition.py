import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from dataset_verify import verify as q
from dataset_verify.recognition import RecognitionBackend


class RecognitionTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((q.HERE / 'config.json').read_text())
        self.recognizer = SimpleNamespace(classes=['front', 'roundhouse', 'axe'])
        self.recognizer.predict = Mock(return_value=dict(used=True, predicted_action='axe', confidence=.9,
                                                        margin=.8, probabilities={'front': .05, 'roundhouse': .05, 'axe': .9}))

    def assign(self, declared=''):
        row = {'technique': declared}
        report = q.new_report(Path('video.mp4'), row, self.config)
        action = q.assign_action({}, row, self.recognizer, lambda a: a, self.config, report)
        return action, row, report

    def test_auto_selects_scorer_from_prediction(self):
        action, row, report = self.assign()
        self.assertEqual(action, 'axe')
        self.assertEqual(row['technique_source'], 'auto')
        self.assertEqual(report['technique_declared'], '')
        self.assertTrue(q.finalize(report)['approved'])

    def test_mismatch_keeps_declared_label_and_requires_review(self):
        action, row, report = self.assign('front')
        self.assertEqual(action, 'front')
        self.assertEqual(row['technique_predicted'], 'axe')
        self.assertFalse(q.finalize(report)['approved'])

    def test_low_confidence_or_small_margin_does_not_autolabel(self):
        for confidence, margin in ((.5, .3), (.9, .01)):
            self.recognizer.predict.return_value.update(confidence=confidence, margin=margin)
            action, _, report = self.assign()
            self.assertEqual(action, 'unidentify_action')
            self.assertFalse(q.finalize(report)['approved'])

    def test_out_of_scope_declared_action_is_preserved(self):
        action, row, report = self.assign('hook')
        self.assertEqual(action, 'hook')
        self.assertFalse(report['recognition']['used'])
        self.recognizer.predict.assert_not_called()
        self.assertTrue(q.finalize(report)['approved'])

    def test_inference_error_does_not_stop_batch(self):
        self.recognizer.predict.side_effect = ValueError('Không đủ pose')
        action, _, report = self.assign()
        self.assertEqual(action, 'unidentify_action')
        self.assertFalse(q.finalize(report)['approved'])

    def test_disable_recognition(self):
        row = {'technique': 'side'}
        report = q.new_report(Path('video.mp4'), row, self.config)
        self.assertEqual(q.assign_action({}, row, None, lambda a: a, self.config, report), 'side')
        self.assertFalse(report['recognition']['used'])

    def test_adapter_validates_pose_and_model_probabilities(self):
        backend = RecognitionBackend.__new__(RecognitionBackend)
        backend.classes = ['front', 'roundhouse', 'axe']
        backend.classifier = SimpleNamespace(name='Fake', predict=Mock(return_value=(np.array([.8, .1, .1]), (0, 29, 'ankle'))))
        pose = dict(kpts=np.ones((30, 13, 2)), vis=np.ones((30, 13)), fps=30.)
        original = pose['kpts'].copy()
        result = backend.predict(pose)
        self.assertEqual(result['predicted_action'], 'front')
        self.assertAlmostEqual(result['margin'], .7)
        np.testing.assert_array_equal(pose['kpts'], original)
        backend.classifier.predict.return_value = (np.array([float('nan'), .1, .1]), (0, 29, 'ankle'))
        with self.assertRaises(ValueError):
            backend.predict(pose)
        pose['vis'][:] = 0
        with self.assertRaises(ValueError):
            backend.predict(pose)


if __name__ == '__main__':
    unittest.main()
