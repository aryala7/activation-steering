"""Verify validation selection and paired human-review aggregation offline."""

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from steer.main import MODELS, rank_main, sweep_main, validation_turns
from test_contrastive_pairs import make_dialogue


class SweepTests(unittest.TestCase):
    def test_validation_selection_uses_latest_user_state(self):
        row = make_dialogue()
        row["turns"]["utterance"][1] = "A hotel is available."
        selected = list(validation_turns([row], 1))
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["constraints"], {"hotel-area": ["north"]})
        self.assertEqual(selected[0]["messages"][-1]["role"], "user")
        self.assertNotIn("A hotel is available.", str(selected[0]))

    def test_rank_reports_joint_success_and_baseline(self):
        records = [
            {"id": 0, "layer": None, "strength": 0, "dialogue_id": "a", "turn_id": "1"},
            {"id": 1, "layer": 12, "strength": 1, "dialogue_id": "a", "turn_id": "1"},
            {"id": 2, "layer": None, "strength": 0, "dialogue_id": "b", "turn_id": "1"},
            {"id": 3, "layer": 12, "strength": 1, "dialogue_id": "b", "turn_id": "1"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            predictions = Path(directory) / "predictions.jsonl"
            review = Path(directory) / "review.csv"
            predictions.write_text("".join(json.dumps(r) + "\n" for r in records))
            with review.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["id", "constraint_ok", "useful"])
                writer.writeheader()
                writer.writerows([
                    {"id": 0, "constraint_ok": 0, "useful": 1},
                    {"id": 1, "constraint_ok": 1, "useful": 1},
                    {"id": 2, "constraint_ok": 1, "useful": 0},
                    {"id": 3, "constraint_ok": 1, "useful": 1},
                ])
            output = io.StringIO()
            with redirect_stdout(output):
                rank_main(["--predictions", str(predictions), "--review", str(review)])
            self.assertIn("12,1,2,1.000,1.000,1.000", output.getvalue())
            self.assertIn("baseline,0,2,0.500,0.500,0.000", output.getvalue())

    def test_sweep_generates_baseline_and_layer_from_training_vectors(self):
        import torch

        stored = {
            "split": "train", "model_id": MODELS["mistral"],
            "model_revision": "same", "layers": [12],
            "mean_difference": torch.ones(1, 4),
        }
        model = Mock()
        model.config._commit_hash = "same"
        model.to.return_value = model
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            fake_generate = Mock(side_effect=["Baseline answer", "Steered answer"])
            with patch("steer.main.torch.load", return_value=stored), \
                 patch("steer.main.load_dataset", return_value=[make_dialogue()]) as load, \
                 patch("steer.main.AutoTokenizer.from_pretrained"), \
                 patch("steer.main.AutoModelForCausalLM.from_pretrained", return_value=model), \
                 patch("steer.main.generate_response", fake_generate), \
                 redirect_stdout(io.StringIO()):
                sweep_main(["--activations", "train.pt", "--strengths", "1",
                            "--limit", "1", "--output-dir", str(output_dir)])
            load.assert_called_once_with("tuetschek/multi_woz_v22", split="validation")
            records = [json.loads(line) for line in (output_dir / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual([(r["layer"], r["response"]) for r in records],
                             [(None, "Baseline answer"), (12, "Steered answer")])
            self.assertIsNone(fake_generate.call_args_list[0].kwargs["vector"])
            torch.testing.assert_close(fake_generate.call_args_list[1].kwargs["vector"],
                                       torch.ones(4))
            with (output_dir / "review.csv").open(newline="") as file:
                review = list(csv.DictReader(file))
            self.assertEqual(len(review), 2)
            self.assertEqual({r["constraint_ok"] for r in review}, {""})


if __name__ == "__main__":
    unittest.main()
