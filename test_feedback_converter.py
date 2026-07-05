import unittest

from feedback_converter import find_text_column, normalize_column_name


class FeedbackConverterTests(unittest.TestCase):
    def test_normalizes_columns_with_spaces_and_underscores(self):
        self.assertEqual(normalize_column_name(" Tweet_Text "), "tweet text")

    def test_finds_xquik_tweet_text_column(self):
        self.assertEqual(
            find_text_column(["id", "Tweet Text", "likes"]),
            "Tweet Text",
        )

    def test_returns_none_for_missing_text_column(self):
        self.assertIsNone(find_text_column(["id", "created_at"]))


if __name__ == "__main__":
    unittest.main()
