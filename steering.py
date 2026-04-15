import threading
from typing import Generator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

SUPPORTED_MODELS = {
    "Llama-3.2-1B-Instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "Llama-3.2-3B-Instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen2.5-0.5B-Instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B-Instruct": "Qwen/Qwen2.5-3B-Instruct",
}

HARMFUL_CSV_URL = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"


def detect_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(
    model_name: str, hf_token: str, device: str
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    model_id = SUPPORTED_MODELS[model_name]
    dtype = torch.float16 if device == "mps" else torch.bfloat16

    token = hf_token.strip() or None
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, token=token, torch_dtype=dtype
    ).to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_harmful_prompts() -> tuple[list[str], list[str]]:
    df = pd.read_csv(HARMFUL_CSV_URL)
    goals = df["goal"].tolist()
    mid = len(goals) // 2
    return goals[:mid], goals[mid:]


def load_harmless_prompts(n_samples: int = 256) -> list[str]:
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    instructions = [row["instruction"] for row in ds if row["instruction"].strip()]
    return instructions[:n_samples]


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------


@torch.no_grad()
def collect_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    device: str,
    progress_cb=None,
) -> dict[int, torch.Tensor]:
    num_layers = model.config.num_hidden_layers
    all_acts: dict[int, list[torch.Tensor]] = {l: [] for l in range(num_layers)}

    for i, prompt in enumerate(prompts):
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(device)

        outputs = model(input_ids, output_hidden_states=True)
        # outputs.hidden_states: tuple of (num_layers+1) tensors, each (1, seq_len, hidden)
        # index 0 is embedding output; layers 1..num_layers correspond to decoder layers 0..num_layers-1
        for l in range(num_layers):
            h = outputs.hidden_states[l + 1][0, -1, :].cpu()
            all_acts[l].append(h)

        if progress_cb and (i + 1) % 16 == 0:
            progress_cb(i + 1, len(prompts))

    return {l: torch.stack(vecs) for l, vecs in all_acts.items()}


# ---------------------------------------------------------------------------
# Refusal direction
# ---------------------------------------------------------------------------


def compute_refusal_direction(
    harmful_acts: dict[int, torch.Tensor],
    harmless_acts: dict[int, torch.Tensor],
) -> dict[int, torch.Tensor]:
    directions = {}
    for l in harmful_acts:
        mu = harmful_acts[l].mean(dim=0)
        nu = harmless_acts[l].mean(dim=0)
        directions[l] = mu - nu
    return directions


def select_best_refusal_direction(
    directions: dict[int, torch.Tensor],
) -> tuple[torch.Tensor, int]:
    best_layer = max(directions, key=lambda l: directions[l].norm().item())
    r = directions[best_layer]
    r = r / r.norm()  # normalize for numerical stability
    return r, best_layer


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------


def compute_pca_plot(
    harmful_acts: torch.Tensor,
    harmless_acts: torch.Tensor,
    layer: int,
) -> plt.Figure:
    H = harmful_acts.float()
    N = harmless_acts.float()
    X = torch.cat([H, N], dim=0)
    X_centered = X - X.mean(dim=0, keepdim=True)
    _, _, V = torch.pca_lowrank(X_centered, q=2)
    coords = (X_centered @ V[:, :2]).numpy()
    n_h = H.shape[0]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        coords[:n_h, 0], coords[:n_h, 1],
        c="#d62728", alpha=0.5, s=20, label="harmful",
    )
    ax.scatter(
        coords[n_h:, 0], coords[n_h:, 1],
        c="#1f77b4", alpha=0.5, s=20, label="harmless",
    )
    h_c = coords[:n_h].mean(axis=0)
    n_c = coords[n_h:].mean(axis=0)
    ax.scatter(h_c[0], h_c[1], c="#8b0000", marker="X", s=220,
               edgecolors="white", linewidths=1.5, label="harmful centroid")
    ax.scatter(n_c[0], n_c[1], c="#08306b", marker="X", s=220,
               edgecolors="white", linewidths=1.5, label="harmless centroid")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"PCA of post-instruction activations (layer {layer})")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    return fig


def _aggregate_tokens_to_words(
    tokenizer: AutoTokenizer,
    input_ids: list[int],
    message: str,
) -> tuple[list[str], list[list[int]]]:
    """Map token indices in input_ids to whitespace-separated words of message."""
    # Tokenize each word separately (without special tokens), then locate in input_ids
    words = message.split()
    if not words:
        return [], []

    # Convert input_ids -> token string list (preserves word-boundary markers)
    token_strs = tokenizer.convert_ids_to_tokens(input_ids)
    # Strip BPE boundary markers so equality on decoded pieces works
    def _clean(t: str) -> str:
        return t.replace("Ġ", "").replace("▁", "")

    cleaned = [_clean(t) for t in token_strs]

    word_ranges: list[list[int]] = []
    cursor = 0
    for word in words:
        remaining = word
        idxs: list[int] = []
        while remaining and cursor < len(cleaned):
            piece = cleaned[cursor]
            if piece and remaining.startswith(piece):
                idxs.append(cursor)
                remaining = remaining[len(piece):]
                cursor += 1
            else:
                cursor += 1
                if not idxs:
                    # Haven't started matching this word yet; keep scanning
                    continue
                # Mismatch mid-word — fall back: include this token and stop
                break
        if not idxs:
            # Could not locate this word; skip it
            word_ranges.append([])
        else:
            word_ranges.append(idxs)
    return words, word_ranges


@torch.no_grad()
def compute_token_cosine_heatmap(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    directions: dict[int, torch.Tensor],
    message: str,
    device: str,
) -> plt.Figure:
    enc = tokenizer(message, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(device)

    outputs = model(input_ids, output_hidden_states=True)

    words, word_ranges = _aggregate_tokens_to_words(
        tokenizer, input_ids[0].tolist(), message
    )
    if not words:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "(empty message)", ha="center", va="center")
        ax.axis("off")
        return fig

    num_layers = len(directions)
    matrix = np.zeros((len(words), num_layers), dtype=np.float32)

    for l in range(num_layers):
        h = outputs.hidden_states[l + 1][0].float().cpu()  # (seq_len, hidden)
        d = directions[l].float().cpu()
        d_norm = d / (d.norm() + 1e-8)
        h_norm = h / (h.norm(dim=-1, keepdim=True) + 1e-8)
        sims = (h_norm @ d_norm).numpy()  # (seq_len,)
        for w_idx, tok_idxs in enumerate(word_ranges):
            if tok_idxs:
                matrix[w_idx, l] = float(np.mean(sims[tok_idxs]))

    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(words) + 1)))
    vmax = float(np.max(np.abs(matrix))) or 1.0
    im = ax.imshow(
        matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Word")
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words)
    ax.set_xticks(range(num_layers))
    ax.set_title("Cosine similarity: token hidden state vs. layer-wise Δmean")
    fig.colorbar(im, ax=ax, shrink=0.8, label="cosine")
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    return fig


# ---------------------------------------------------------------------------
# Steering context (forward hooks)
# ---------------------------------------------------------------------------


class SteeringContext:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        refusal_dir: torch.Tensor,
        beta: float,
        layer_start: int,
        layer_end: int,
    ):
        self.model = model
        self.refusal_dir = refusal_dir
        self.beta = beta
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.hooks: list = []

    def _make_hook(self):
        r = self.refusal_dir
        beta = self.beta

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output

            # Compute in float32 to avoid float16 overflow on MPS
            orig_dtype = h.dtype
            h_f32 = h.float()
            proj_coeff = torch.einsum("d, ...d -> ...", r, h_f32)
            h_steered = (h_f32 - beta * proj_coeff.unsqueeze(-1) * r).to(orig_dtype)

            if isinstance(output, tuple):
                return (h_steered,) + output[1:]
            return h_steered

        return hook_fn

    def attach_hooks(self):
        hook_fn = self._make_hook()
        for idx in range(self.layer_start, self.layer_end + 1):
            layer = self.model.model.layers[idx]
            handle = layer.register_forward_hook(hook_fn)
            self.hooks.append(handle)

    def remove_hooks(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()

    def __enter__(self):
        self.attach_hooks()
        return self

    def __exit__(self, *args):
        self.remove_hooks()


# ---------------------------------------------------------------------------
# Generation with steering
# ---------------------------------------------------------------------------


def generate_with_steering(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: list[dict],
    refusal_dir: torch.Tensor,
    beta: float,
    layer_start: int,
    layer_end: int,
    device: str,
    max_new_tokens: int = 512,
) -> Generator[str, None, None]:
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    # Move refusal_dir to model device (use float32 to avoid float16 NaN on MPS)
    r = refusal_dir.to(device=device, dtype=torch.float32)

    ctx = SteeringContext(model, r, beta, layer_start, layer_end)
    ctx.attach_hooks()

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        streamer=streamer,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    try:
        for text in streamer:
            yield text
    finally:
        thread.join()
        ctx.remove_hooks()
