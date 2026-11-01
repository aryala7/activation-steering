"""Capture context-conditioned response activations from Qwen/Llama-style models."""

from functools import partial
from contextlib import contextmanager


def parse_history(prompt: str) -> list[dict[str, str]]:
    """Read older pairs that store history as User:/Assistant: text."""
    if not prompt.endswith("\nAssistant:"):
        raise ValueError("Expected a prompt ending in a blank Assistant: turn")

    history = prompt.removesuffix("\nAssistant:")
    messages = []
    roles = {"User": "user", "Assistant": "assistant"}

    for line in history.split("\n"):
        speaker, separator, content = line.partition(": ")
        if separator and speaker in roles:
            messages.append({"role": roles[speaker], "content": content})
        elif messages:
            # Continue the previous message when an utterance spans several lines.
            messages[-1]["content"] += "\n" + line
        else:
            raise ValueError("Cannot parse dialogue history")

    return messages


def response_messages(pair: dict, response_key: str) -> list[dict[str, str]]:
    """Combine the shared history with one candidate assistant response."""
    if "messages" in pair:
        history = pair["messages"]
    else:
        history = parse_history(pair["prompt"])

    if not history or history[-1].get("role") != "user":
        raise ValueError("History must end with the user turn before the response")
    for message in history:
        if message.get("role") not in {"system", "user", "assistant"}:
            raise ValueError("Unsupported message role")
        if not isinstance(message.get("content"), str):
            raise ValueError("Message content must be text")

    response = pair[response_key]
    if not isinstance(response, str) or not response.strip():
        raise ValueError("Candidate response must be nonempty text")

    return history + [{"role": "assistant", "content": response}]


def encode(tokenizer, pair: dict, response_key: str):
    """Tokenize history + response, ending at the response's last content token."""
    messages = response_messages(pair, response_key)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
        truncation=False,
    )


def capture_activations(model, inputs, layers: list[int]):
    """Return CPU float32 [layers, hidden_size] post-block residual states.

    Layer indices are zero-based. Requires a single unpadded example and a
    model exposing model.layers (Qwen2/Llama); unsupported layouts fail clearly.
    Hooks are removed even when the forward pass raises an error.
    """
    import torch

    blocks = getattr(getattr(model, "model", None), "layers", None)
    if blocks is None:
        raise ValueError("Expected transformer blocks at model.model.layers")
    if not layers or len(set(layers)) != len(layers):
        raise ValueError("Choose at least one layer, without duplicates")
    if any(layer < 0 or layer >= len(blocks) for layer in layers):
        raise ValueError(f"Layer indices must be between 0 and {len(blocks) - 1}")
    ids = inputs["input_ids"]
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
        raise ValueError("Expected a single nonempty token sequence")
    if "attention_mask" in inputs and not inputs["attention_mask"].bool().all():
        raise ValueError("Padded inputs are not supported; process one response at a time")
    max_length = getattr(model.config, "max_position_embeddings", None)
    if max_length and ids.shape[1] > max_length:
        raise ValueError(f"Input has {ids.shape[1]} tokens; model limit is {max_length}")

    captured = {}
    handles = []

    def save_activation(layer, _module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        # Shape: [batch, tokens, hidden_size]. Keep the last response token.
        last_token = hidden[0, -1, :]
        captured[layer] = last_token.detach().float().cpu().clone()

    was_training = model.training
    try:
        model.eval()
        for layer in layers:
            hook = partial(save_activation, layer)
            handle = blocks[layer].register_forward_hook(hook)
            handles.append(handle)
        device = model.get_input_embeddings().weight.device
        model_inputs = {}
        for name, tensor in inputs.items():
            model_inputs[name] = tensor.to(device)

        with torch.inference_mode():
            model(**model_inputs, use_cache=False)
        return torch.stack([captured[layer] for layer in layers])
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)


def make_vector(model, tokenizer, pairs: list[dict], layers: list[int],
                capture_pair=None, retain_pairs=True) -> dict:
    """Average paired activations, optionally retaining examples for plots."""
    import torch

    if not pairs:
        raise ValueError("At least one contrastive pair is required")
    positives = []
    negatives = []
    running_sum = None
    if capture_pair is None:
        def capture_pair(pair):
            return (
                capture_activations(model, encode(tokenizer, pair, "positive_response"), layers),
                capture_activations(model, encode(tokenizer, pair, "negative_response"), layers),
            )

    for pair in pairs:
        positive, negative = capture_pair(pair)
        difference = positive.float() - negative.float()
        running_sum = difference.clone() if running_sum is None else running_sum.add_(difference)
        if retain_pairs:
            positives.append(positive)
            negatives.append(negative)

    mean_difference = running_sum / len(pairs)

    # Keep both the raw vector and a unit-length version for each layer.
    lengths = mean_difference.norm(dim=-1, keepdim=True)
    unit_direction = mean_difference / lengths.clamp_min(1e-8)
    result = {
        "mean_difference": mean_difference,
        "unit_direction": unit_direction,
        "layers": layers,
        "pairs": pairs,
        "token_position": "last_response_content_token",
        "activation_location": "transformer_block_output_before_final_norm",
    }
    if retain_pairs:
        positive = torch.stack(positives)
        negative = torch.stack(negatives)
        result.update({
            "positive_activations": positive,
            "negative_activations": negative,
            "differences": positive - negative,
        })
    return result


@contextmanager
def mlx_layer_taps(model, callbacks):
    """Temporarily wrap MLX transformer blocks; restore them on every exit."""
    import mlx.nn as nn

    blocks = getattr(getattr(model, "model", None), "layers", None)
    if blocks is None or any(i < 0 or i >= len(blocks) for i in callbacks):
        raise ValueError("Invalid MLX transformer layer")

    class Tap(nn.Module):
        def __init__(self, block, callback):
            super().__init__()
            self.block = block
            self.callback = callback

        def __call__(self, *args, **kwargs):
            return self.callback(self.block(*args, **kwargs))

    originals = {i: blocks[i] for i in callbacks}
    try:
        for i, callback in callbacks.items():
            blocks[i] = Tap(originals[i], callback)
        yield
    finally:
        for i, block in originals.items():
            blocks[i] = block


def mlx_capture_pair(model, tokenizer, pair, layers):
    """Read final content-token residual states from a quantized MLX model."""
    import mlx.core as mx
    import numpy as np
    import torch

    def capture(response_key):
        messages = response_messages(pair, response_key)
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            continue_final_message=True,
        )
        if not ids:
            raise ValueError("Empty MLX input")
        max_length = getattr(model.args, "max_position_embeddings", None)
        if max_length and len(ids) > max_length:
            raise ValueError("Dialogue exceeds the MLX model context window")
        captured = {}
        callbacks = {
            layer: (lambda output, index=layer: captured.setdefault(index, output[0, -1, :]))
            for layer in layers
        }
        with mlx_layer_taps(model, callbacks):
            logits = model(mx.array(ids)[None, :])
            mx.eval(*(captured[i] for i in layers))
            del logits
        return torch.stack([
            torch.from_numpy(np.asarray(captured[i]).astype(np.float32, copy=True))
            for i in layers
        ])

    return capture("positive_response"), capture("negative_response")


def mlx_generate_response(model, tokenizer, messages, max_new_tokens,
                          layer=None, vector=None, strength=0.0):
    """Generate greedily on MLX, steering only the prediction positions."""
    import mlx.core as mx
    from mlx_lm import generate

    if not messages or messages[-1]["role"] != "user":
        raise ValueError("Dialogue history must end in a user message")
    if (layer is None) != (vector is None):
        raise ValueError("Provide both layer and vector, or neither")
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
    )
    max_length = getattr(model.args, "max_position_embeddings", None)
    if max_length and len(prompt_ids) + max_new_tokens > max_length:
        raise ValueError("Dialogue and generation exceed the MLX context window")

    callbacks = {}
    if layer is not None and strength:
        direction = mx.array(vector.detach().cpu().numpy()) * strength
        processed = 0

        def steer(output):
            nonlocal processed
            if output.shape[-1] != direction.size:
                raise ValueError("Steering vector does not match MLX hidden states")
            processed += output.shape[1]
            # MLX prefill can split a long prompt into chunks. Do not change
            # intermediate prompt chunks; the final position predicts token 1.
            if processed < len(prompt_ids):
                return output
            return mx.concatenate(
                [output[:, :-1, :], output[:, -1:, :] + direction.astype(output.dtype)[None, None, :]],
                axis=1,
            )

        callbacks[layer] = steer
    with mlx_layer_taps(model, callbacks):
        return generate(model, tokenizer, prompt=prompt_ids, max_tokens=max_new_tokens,
                        prefill_step_size=512, verbose=False).strip()


@contextmanager
def steering_hook(model, layer: int, vector, strength: float):
    """Steer the next-token prediction and subsequent decode steps at one block.

    During prefill the last prompt position predicts the first answer token;
    earlier prompt positions stay untouched. With a KV cache, the block sees
    just the new token at later steps. Hook removal is guaranteed on failure.
    """
    import torch

    blocks = getattr(getattr(model, "model", None), "layers", None)
    if blocks is None or not 0 <= layer < len(blocks):
        raise ValueError("Invalid model layer for steering")
    if vector.ndim != 1:
        raise ValueError("Steering vector must have shape [hidden_size]")

    def add_direction(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.ndim != 3 or hidden.shape[-1] != vector.numel():
            raise ValueError("Steering vector does not match hidden states")
        delta = vector.to(device=hidden.device, dtype=hidden.dtype) * strength
        changed = hidden.clone()
        changed[:, -1, :] += delta
        if isinstance(output, tuple):
            return (changed, *output[1:])
        return changed

    handle = blocks[layer].register_forward_hook(add_direction)
    try:
        yield
    finally:
        handle.remove()


def generate_response(model, tokenizer, messages, max_new_tokens: int,
                      layer: int | None = None, vector=None, strength: float = 0.0) -> str:
    """Generate one deterministic response with optional single-layer steering."""
    import torch

    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("Dialogue history must end in a user message")
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", truncation=False,
    )
    input_length = inputs["input_ids"].shape[-1]
    max_positions = getattr(model.config, "max_position_embeddings", None)
    if max_positions and input_length + max_new_tokens > max_positions:
        raise ValueError("Dialogue and generation exceed the model context window")
    device = model.get_input_embeddings().weight.device
    model_inputs = {key: value.to(device) for key, value in inputs.items()}
    if (layer is None) != (vector is None):
        raise ValueError("Provide both layer and vector, or neither")
    if layer is None and strength:
        raise ValueError("A nonzero strength requires a layer and vector")

    from contextlib import nullcontext
    intervention = (steering_hook(model, layer, vector, strength)
                    if layer is not None and strength else nullcontext())
    model.eval()
    with torch.inference_mode(), intervention:
        output = model.generate(
            **model_inputs, max_new_tokens=max_new_tokens, do_sample=False,
            use_cache=True, pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0, input_length:], skip_special_tokens=True).strip()
