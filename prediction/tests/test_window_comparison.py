"""Tests for moving-average window comparison with synthetic CSV data."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from prediction.src.compare_windows import compare_window_sizes


SYNTHETIC_COMPARISON_CSV = (
	"timestamp,request_rate_rps\n"
	"2026-10-12T10:00:00Z,20\n"
	"2026-10-12T10:00:10Z,25\n"
	"2026-10-12T10:00:20Z,23\n"
	"2026-10-12T10:00:30Z,30\n"
	"2026-10-12T10:00:40Z,35\n"
)


class WindowComparisonTests(unittest.TestCase):
	def setUp(self) -> None:
		self.temporary_directory = tempfile.TemporaryDirectory()
		self.root = Path(self.temporary_directory.name)
		self.input_csv = self.root / "synthetic_comparison.csv"
		self.input_csv.write_text(SYNTHETIC_COMPARISON_CSV, encoding="utf-8")

	def tearDown(self) -> None:
		self.temporary_directory.cleanup()

	def test_comparison_csv_has_calculated_results_and_explicit_skip(self) -> None:
		output_csv = self.root / "outputs" / "comparison.csv"
		comparison = compare_window_sizes(
			self.input_csv,
			window_sizes=(2, 3, 4),
			output_csv=output_csv,
			dataset_label="synthetic unit-test data",
		)
		saved = pd.read_csv(output_csv)

		self.assertTrue(output_csv.is_file())
		self.assertEqual(saved["window_size"].tolist(), [2, 3, 4])
		self.assertEqual(
			saved["evaluation_status"].tolist(), ["evaluated", "evaluated", "skipped"]
		)
		self.assertEqual(saved["number_of_evaluated_predictions"].tolist()[:2], [3, 2])
		self.assertTrue(pd.isna(saved.loc[2, "mae_rps"]))
		self.assertIn("at least 6 rows", saved.loc[2, "skip_reason"])
		self.assertEqual(saved.loc[0, "dataset_label"], "synthetic unit-test data")
		pd.testing.assert_frame_equal(
			comparison.fillna({"skip_reason": ""}),
			saved.fillna({"skip_reason": ""}),
		)

	def test_comparison_is_reproducible(self) -> None:
		first = compare_window_sizes(
			self.input_csv,
			window_sizes=(2, 3, 4),
			output_csv=self.root / "first.csv",
		)
		second = compare_window_sizes(
			self.input_csv,
			window_sizes=(2, 3, 4),
			output_csv=self.root / "second.csv",
		)

		pd.testing.assert_frame_equal(first, second)

	def test_invalid_or_duplicate_window_sizes_are_rejected(self) -> None:
		for sizes in ((0,), (-1,), (2.5,), (True,), (2, 2)):
			with self.subTest(window_sizes=sizes):
				with self.assertRaises(ValueError):
					compare_window_sizes(
						self.input_csv,
						window_sizes=sizes,
						output_csv=self.root / "invalid.csv",
					)

	def test_output_cannot_overwrite_raw_input_csv(self) -> None:
		original_content = self.input_csv.read_text(encoding="utf-8")
		with self.assertRaisesRegex(ValueError, "must not overwrite the input CSV"):
			compare_window_sizes(
				self.input_csv,
				window_sizes=(2, 3),
				output_csv=self.input_csv,
			)
		self.assertEqual(self.input_csv.read_text(encoding="utf-8"), original_content)


if __name__ == "__main__":
	unittest.main()