import sys
import unittest
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from filename_action import detect_filename_action


class FilenameActionTests(unittest.TestCase):
    actions = [
        ("Đá", "axe", "Naeryo chagi (đá bổ)"),
        ("Đá", "front", "Ap chagi (đá tống trước)"),
        ("Đá", "roundhouse", "Dollyo chagi (đá vòng cầu)"),
        ("Đá", "roundhouse_low", "Dollyo chagi thấp (đá vòng cầu thấp)"),
        ("Tay", "low_block", "Gạt thấp"),
        ("Tay", "straight", "Đấm thẳng"),
        ("Phối hợp", "block_punch", "Gạt thấp – đấm thẳng"),
    ]
    aliases = {"da_truoc": "front", "gat_thap": "low_block"}

    def detect(self, filename):
        return detect_filename_action(filename, self.actions, self.aliases)

    def test_requested_examples_and_spelling_variants(self):
        cases = {
            "đá chẻ - trái.mp4": "axe",
            "DA_CHE_PHAI_01.MOV": "axe",
            "đá bổ - phải.mp4": "axe",
            unicodedata.normalize("NFD", "ĐÁ CHẺ – phải.mp4"): "axe",
            "đá trước - trái.mp4": "front",
            "gạt thấp - đấm thẳng - gạt phải.mp4": "block_punch",
            "DA_TRUOC_PHAI_01.MOV": "front",
            "001_GAT_THAP__DAM_THANG__TRAI.avi": "block_punch",
            "Ap chagi - trái.mp4": "front",
            "đá tống trước - phải.mp4": "front",
            "đá vòng cầu thấp - trái.mp4": "roundhouse_low",
            "front - L.mp4": "front",
            unicodedata.normalize("NFD", "gạt thấp – đấm thẳng.mp4"): "block_punch",
        }
        for filename, expected in cases.items():
            with self.subTest(filename=filename):
                self.assertEqual(self.detect(filename), expected)

    def test_unknown_ambiguous_and_partial_words(self):
        for filename in ("video001.mp4", "confront.mp4", "", "đá trước - đấm thẳng.mp4"):
            with self.subTest(filename=filename):
                self.assertIsNone(self.detect(filename))

    def test_only_filename_is_used(self):
        self.assertIsNone(self.detect(r"C:\đá chẻ\video001.mp4"))
        self.assertIsNone(self.detect(r"C:\đá trước\video001.mp4"))
        self.assertEqual(self.detect('"C:\\đấm thẳng\\đá trước - trái.mp4"'), "front")


if __name__ == "__main__":
    unittest.main()
