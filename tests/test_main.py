"""Check the complete pipeline without downloading datasets or model weights."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from steer.main import DATASET_ID, main
from test_contrastive_pairs import make_dialogue


class PipelineTests(unittest.TestCase):
    def test_invalid_arguments(self):
        for args in [["--limit", "0"], ["--layers", "-1"], ["--layers", "1", "1"]]:
            with self.subTest(args=args), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main(args)
                self.assertEqual(error.exception.code, 2)

    def test_pairs_are_passed_directly_to_model_and_saved(self):
        dataset = Mock()
        dataset.__len__ = Mock(return_value=1)
        dataset.select.return_value = [make_dialogue()]
        loader = Mock(return_value=dataset)
        model = Mock()
        model.config._commit_hash = "revision"
        model_factory = Mock()
        model_factory.from_pretrained.return_value = model
        tokenizer_factory = Mock()
        torch = Mock(float32="float32", __version__="test")
        with tempfile.TemporaryDirectory() as directory:
            with patch("steer.main.load_dataset", loader), \
                 patch("steer.main.AutoModelForCausalLM", model_factory), \
                 patch("steer.main.AutoTokenizer", tokenizer_factory), \
                 patch("steer.main.torch", torch), redirect_stdout(io.StringIO()):
                with patch("steer.main.make_vector", return_value={}) as capture:
                    main(["--split", "validation", "--limit", "10", "--layers", "0", "1",
                          "--output-dir", directory])
            loader.assert_called_once_with(DATASET_ID, split="validation")
            dataset.select.assert_called_once_with(range(1))
            pairs = capture.call_args.args[2]
            self.assertEqual(capture.call_args.args[3], [0, 1])
            self.assertEqual(pairs[0]["positive_response"], "Try the north.")
            saved_pair = json.loads((Path(directory) / "pairs.jsonl").read_text())
            self.assertEqual(saved_pair, pairs[0])
            result, output = torch.save.call_args.args
            self.assertEqual(result["split"], "validation")
            self.assertEqual(output, Path(directory) / "activations.pt")
