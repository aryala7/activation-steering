"""Exercise MLX wrapper behavior with a small NumPy stand-in on Intel CI."""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from steer.main import choose_backend
from steer.models.model import make_vector, mlx_capture_pair, mlx_generate_response


class Block:
    use_sliding = False

    def __call__(self, hidden):
        return hidden + 1


class Model:
    def __init__(self):
        self.model = SimpleNamespace(layers=[Block()])
        self.args = SimpleNamespace(max_position_embeddings=32)

    def __call__(self, ids):
        hidden = np.repeat(np.asarray(ids, dtype=np.float32)[:, :, None], 4, axis=-1)
        for layer in self.model.layers:
            # mlx-lm reads this on the block before calling it.
            self.asserted_sliding = layer.use_sliding
            hidden = layer(hidden)
        return hidden


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        if kwargs.get("continue_final_message"):
            return [1, 2] if messages[-1]["content"] == "good" else [1, 3]
        return [1, 2, 3]


class MLXBackendTests(unittest.TestCase):
    def setUp(self):
        nn = types.ModuleType("mlx.nn")
        nn.Module = type("Module", (), {})
        mx = types.ModuleType("mlx.core")
        mx.array = np.asarray
        mx.concatenate = np.concatenate
        mx.eval = lambda *args: None
        mlx = types.ModuleType("mlx")
        mlx.nn, mlx.core = nn, mx
        self.modules = patch.dict(sys.modules, {"mlx": mlx, "mlx.nn": nn, "mlx.core": mx})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_capture_reads_content_token_and_restores_block(self):
        model = Model()
        original = model.model.layers[0]
        pair = {
            "messages": [{"role": "user", "content": "Find a restaurant"}],
            "positive_response": "good", "negative_response": "bad",
        }
        positive, negative = mlx_capture_pair(model, Tokenizer(), pair, [0])
        torch.testing.assert_close(positive - negative, -torch.ones(1, 4))
        self.assertIs(model.asserted_sliding, False)
        self.assertIs(model.model.layers[0], original)

    def test_chunked_prefill_is_untouched_until_final_prompt_position(self):
        model = Model()
        original = model.model.layers[0]
        outputs = []

        def generate(model, tokenizer, prompt, **kwargs):
            self.assertEqual(prompt, [1, 2, 3])
            for ids in ([[1, 2]], [[3]], [[4]]):
                outputs.append(model(np.asarray(ids)))
            return "steered"

        backend = types.ModuleType("mlx_lm")
        backend.generate = generate
        with patch.dict(sys.modules, {"mlx_lm": backend}):
            response = mlx_generate_response(
                model, Tokenizer(), [{"role": "user", "content": "Hi"}],
                5, layer=0, vector=torch.ones(4), strength=2,
            )
        self.assertEqual(response, "steered")
        np.testing.assert_allclose(outputs[0][0, -1], np.full(4, 3))
        np.testing.assert_allclose(outputs[1][0, -1], np.full(4, 6))
        np.testing.assert_allclose(outputs[2][0, -1], np.full(4, 7))
        self.assertIs(model.asserted_sliding, False)
        self.assertIs(model.model.layers[0], original)

    def test_running_vector_does_not_retain_pair_tensors(self):
        def capture(pair):
            return (torch.full((1, 4), pair), torch.zeros(1, 4))

        result = make_vector(None, None, [2, 4], [0], capture_pair=capture,
                             retain_pairs=False)
        torch.testing.assert_close(result["mean_difference"], torch.full((1, 4), 3.0))
        self.assertNotIn("positive_activations", result)

    def test_auto_backend_is_mlx_for_native_apple_silicon(self):
        with patch("steer.main.sys.platform", "darwin"), \
             patch("steer.main.platform.machine", return_value="arm64"):
            self.assertEqual(choose_backend("auto"), "mlx")
        self.assertEqual(choose_backend("auto"), "torch")


if __name__ == "__main__":
    unittest.main()
