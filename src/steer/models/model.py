"""Capture context-conditioned response activations from Qwen/Llama-style models."""

from functools import partial


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


def make_vector(model, tokenizer, pairs: list[dict], layers: list[int]) -> dict:
    """Preserve pair activations and raw mean differences for later analysis."""
    import torch

    if not pairs:
        raise ValueError("At least one contrastive pair is required")
    positives = []
    negatives = []

    for pair in pairs:
        # 1. Encode the same history with each candidate response.
        positive_input = encode(tokenizer, pair, "positive_response")
        negative_input = encode(tokenizer, pair, "negative_response")

        # 2. Run both inputs and record the selected layers.
        positive = capture_activations(model, positive_input, layers)
        negative = capture_activations(model, negative_input, layers)
        positives.append(positive)
        negatives.append(negative)

    # 3. Average positive-minus-negative differences across pairs.
    positive = torch.stack(positives)
    negative = torch.stack(negatives)
    differences = positive - negative
    mean_difference = differences.mean(dim=0)

    # Keep both the raw vector and a unit-length version for each layer.
    lengths = mean_difference.norm(dim=-1, keepdim=True)
    unit_direction = mean_difference / lengths.clamp_min(1e-8)
    return {
        "positive_activations": positive,
        "negative_activations": negative,
        "differences": differences,
        "mean_difference": mean_difference,
        "unit_direction": unit_direction,
        "layers": layers,
        "pairs": pairs,
        "token_position": "last_response_content_token",
        "activation_location": "transformer_block_output_before_final_norm",
    }
