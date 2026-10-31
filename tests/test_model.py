"""Context formatting and activation extraction regressions (no model download)."""

import importlib.util
import unittest
from unittest.mock import Mock

from steer.models.model import (capture_activations, encode, make_vector,
                                response_messages, steering_hook)


def example_pair():
    return {
        "prompt": "User: North please.\nAssistant: A hotel?\nUser: Yes.\nAssistant:",
        "positive_response": "Try the north.",
        "negative_response": "Try the south.",
    }


class ContextTests(unittest.TestCase):
    def test_context_is_shared_and_candidate_is_assistant(self):
        pair = example_pair()
        positive = response_messages(pair, "positive_response")
        negative = response_messages(pair, "negative_response")
        self.assertEqual(positive[:-1], negative[:-1])
        self.assertEqual([m["role"] for m in positive],
                         ["user", "assistant", "user", "assistant"])
        self.assertEqual(positive[0]["content"], "North please.")
        self.assertEqual(positive[-1]["content"], pair["positive_response"])

    def test_structured_history_preserves_multiline_content(self):
        pair = example_pair()
        pair["messages"] = [{"role": "user", "content": "Quote:\nAssistant: hello"}]
        messages = response_messages(pair, "positive_response")
        self.assertEqual(messages[0], pair["messages"][0])
        self.assertEqual(len(pair["messages"]), 1)

    def test_tokenizer_returns_mapping_without_extra_assistant_turn(self):
        tokenizer = Mock()
        encode(tokenizer, example_pair(), "positive_response")
        kwargs = tokenizer.apply_chat_template.call_args.kwargs
        self.assertTrue(kwargs["return_dict"])
        self.assertTrue(kwargs["continue_final_message"])
        self.assertFalse(kwargs["add_generation_prompt"])
        self.assertFalse(kwargs["truncation"])

    def test_invalid_history_and_empty_response_fail(self):
        for changes in [{"prompt": "no history"}, {"positive_response": ""},
                        {"messages": []}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                response_messages(example_pair() | changes, "positive_response")


@unittest.skipUnless(importlib.util.find_spec("torch"), "Install the models extra for tensor tests")
class ActivationTests(unittest.TestCase):
    def setUp(self):
        import torch
        from types import SimpleNamespace
        self.torch = torch

        class Block(torch.nn.Module):
            def forward(self, hidden):
                return hidden.cumsum(dim=1)

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(32, 4)
                self.model = torch.nn.Module()
                self.model.layers = torch.nn.ModuleList([Block(), Block()])
                self.config = SimpleNamespace(max_position_embeddings=20)

            def get_input_embeddings(self):
                return self.embedding

            def forward(self, input_ids, attention_mask=None, use_cache=False):
                hidden = self.embedding(input_ids)
                for block in self.model.layers:
                    hidden = block(hidden)
                return hidden

        self.model = TinyModel()
        self.inputs = {"input_ids": torch.tensor([[1, 2, 3]])}

    def test_last_token_all_layers_and_cleanup(self):
        torch = self.torch
        output = capture_activations(self.model, self.inputs, [0, 1])
        expected = self.model.embedding(self.inputs["input_ids"]).cumsum(1)
        torch.testing.assert_close(output[0], expected[0, -1])
        torch.testing.assert_close(output[1], expected.cumsum(1)[0, -1])
        self.assertEqual(output.shape, (2, 4))
        self.assertTrue(self.model.training)
        self.assertFalse(any(block._forward_hooks for block in self.model.model.layers))

    def test_cleanup_on_forward_error(self):
        with unittest.mock.patch.object(self.model, "forward", side_effect=RuntimeError("test")):
            with self.assertRaises(RuntimeError):
                capture_activations(self.model, self.inputs, [0])
        self.assertFalse(self.model.model.layers[0]._forward_hooks)
        self.assertTrue(self.model.training)

    def test_invalid_layers_and_empty_pairs(self):
        for layers in [[], [-1], [2], [0, 0]]:
            with self.assertRaises(ValueError):
                capture_activations(self.model, self.inputs, layers)
        with self.assertRaises(ValueError):
            make_vector(self.model, None, [], [0])

    def test_generation_hook_only_changes_predictive_position(self):
        torch = self.torch
        plain = self.model(self.inputs["input_ids"])
        with steering_hook(self.model, 0, torch.ones(4), strength=2):
            steered = self.model(self.inputs["input_ids"])
        torch.testing.assert_close(plain[:, :-1], steered[:, :-1])
        torch.testing.assert_close(steered[:, -1] - plain[:, -1], torch.full((1, 4), 2.0))
        self.assertFalse(self.model.model.layers[0]._forward_hooks)

    def test_generation_hook_cleanup_on_error(self):
        with self.assertRaisesRegex(RuntimeError, "failed"):
            with steering_hook(self.model, 0, self.torch.ones(4), 1):
                raise RuntimeError("failed")
        self.assertFalse(self.model.model.layers[0]._forward_hooks)
