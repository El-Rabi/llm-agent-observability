"""Tests for synthetic traces and observability features."""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from features import extract_session_features  # noqa: E402
from generate_traces import generate_trace_data  # noqa: E402


class ObservabilityFeatureTests(unittest.TestCase):
    def test_feature_rows_match_session_count(self) -> None:
        events = generate_trace_data(n_sessions=180, seed=12)
        features = extract_session_features(events, fraction=1.0)
        self.assertEqual(len(features), 180)
        self.assertIn("suspicious_tool_count", features.columns)

    def test_partial_trace_uses_less_evidence(self) -> None:
        events = generate_trace_data(n_sessions=180, seed=13)
        full = extract_session_features(events, fraction=1.0).set_index("session_id")
        early = extract_session_features(events, fraction=0.40).set_index("session_id")
        self.assertTrue((early["event_count"] < full["event_count"]).all())
        self.assertTrue((early["total_data_volume_kb"] <= full["total_data_volume_kb"]).all())

    def test_invalid_fraction_is_rejected(self) -> None:
        events = generate_trace_data(n_sessions=100, seed=14)
        with self.assertRaises(ValueError):
            extract_session_features(events, fraction=0)


if __name__ == "__main__":
    unittest.main()
