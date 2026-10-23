"""Verify the CLI without downloading the dataset."""

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock, patch

from steer.cli import DATASET_ID, main
from test_contrastive_pairs import make_dialogue


class CliTests(unittest.TestCase):
    def test_rejects_nonpositive_limit(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(["--limit", "0"])
        self.assertEqual(error.exception.code, 2)

    def test_writes_json_lines_with_requested_split(self):
        dataset = Mock()
        dataset.__len__ = Mock(return_value=1)
        dataset.select.return_value = [make_dialogue()]
        loader = Mock(return_value=dataset)
        output = io.StringIO()
        with patch.dict("sys.modules", {"datasets": SimpleNamespace(load_dataset=loader)}):
            with redirect_stdout(output):
                main(["--split", "validation", "--limit", "10"])
        loader.assert_called_once_with(DATASET_ID, split="validation")
        dataset.select.assert_called_once_with(range(1))
        pair = json.loads(output.getvalue())
        self.assertEqual(pair["turn_id"], "1")
        self.assertEqual(pair["positive_response"], "Try the north.")
