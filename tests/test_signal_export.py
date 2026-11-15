import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from signal_export import write_snapshot


class SnapshotTests(unittest.TestCase):
    def test_same_session_price_and_public_fields(self):
        history = pd.DataFrame(
            [[10.123456], [11.0]],
            columns=pd.MultiIndex.from_tuples([("Close", "SOXL")]),
            index=pd.to_datetime(["2026-11-12", "2026-11-13"]),
        )
        row = dict(ticker="SOXL", strategy="逆張り", signal_date="2026-11-12",
                   composite_signal="HOLD", rsi=float("nan"), deviation=0.01,
                   score=0, adx=20, ret_5d=-0.01, quantity=123)
        with tempfile.TemporaryDirectory() as directory, patch(
            "signal_export.yf.download", return_value=history
        ), patch.dict("os.environ", {"GITHUB_RUN_ID": "42"}):
            path = Path(directory) / "latest.json"
            write_snapshot(path, [row], ["SOXL"], ["逆張り"], [])
            snapshot = json.loads(path.read_text())
            self.assertEqual(snapshot["run_id"], "42")
            self.assertEqual(snapshot["signals"][0]["close_usd"], "10.123456")
            self.assertIsNone(snapshot["signals"][0]["rsi"])
            self.assertNotIn("quantity", snapshot["signals"][0])
            with self.assertRaises(ValueError):
                write_snapshot(path, [], ["SOXL"], ["逆張り"], ["SOXL"])
            self.assertEqual(json.loads(path.read_text()), snapshot)
