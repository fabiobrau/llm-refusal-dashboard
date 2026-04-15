import queue
import threading

import gradio as gr

from steering import (
    SUPPORTED_MODELS,
    collect_activations,
    compute_pca_plot,
    compute_refusal_direction,
    compute_token_cosine_heatmap,
    detect_device,
    generate_with_steering,
    load_harmful_prompts,
    load_harmless_prompts,
    load_model,
)

# ---------------------------------------------------------------------------
# Global state (model objects are too large for gr.State)
# ---------------------------------------------------------------------------
state = {
    "model": None,
    "tokenizer": None,
    "refusal_dir": None,
    "directions": None,
    "centroid_layer": None,
    "num_layers": None,
    "test_prompts": None,
    "device": None,
    "harmful_acts": None,
    "harmless_acts": None,
}


# ---------------------------------------------------------------------------
# Phase 1: setup
# ---------------------------------------------------------------------------
GATED_MODELS = {"Llama-3.2-1B-Instruct", "Llama-3.2-3B-Instruct"}


def setup_model(model_name, hf_token, device):
    # Outputs: status, centroid_layer, layer_start, layer_end, prompt_dropdown, pca_plot, phase2
    n_outputs = 7
    no_update = (gr.update(),) * (n_outputs - 1)

    if not hf_token.strip() and model_name in GATED_MODELS:
        yield ("Error: please enter your HuggingFace token for Llama models.",) + no_update
        return

    # --- Load model ---
    yield ("Loading model… (this may take a minute)",) + no_update
    try:
        model, tokenizer = load_model(model_name, hf_token, device)
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield (f"Error loading model: {e}",) + no_update
        return

    state["model"] = model
    state["tokenizer"] = tokenizer
    state["device"] = device
    num_layers = model.config.num_hidden_layers
    state["num_layers"] = num_layers

    # --- Load datasets ---
    yield ("Downloading harmful prompts…",) + no_update
    train_harmful, test_harmful = load_harmful_prompts()
    state["test_prompts"] = test_harmful

    yield ("Downloading harmless prompts (tatsu-lab/alpaca)…",) + no_update
    harmless = load_harmless_prompts(n_samples=256)

    # --- Collect activations (thread + queue so the generator can yield progress) ---
    n_train = min(256, len(train_harmful))
    train_subset = train_harmful[:n_train]
    n_harmless = min(256, len(harmless))

    # Harmful
    harmful_q: queue.Queue = queue.Queue()
    harmful_result: list = [None]

    def _run_harmful():
        harmful_result[0] = collect_activations(
            model, tokenizer, train_subset, device,
            progress_cb=lambda done, tot: harmful_q.put((done, tot)),
        )
        harmful_q.put(None)

    threading.Thread(target=_run_harmful, daemon=True).start()
    yield (f"Collecting harmful activations (0/{n_train})…",) + no_update
    while True:
        item = harmful_q.get()
        if item is None:
            break
        done, tot = item
        yield (f"Collecting harmful activations ({done}/{tot})…",) + no_update
    harmful_acts = harmful_result[0]

    # Harmless
    harmless_q: queue.Queue = queue.Queue()
    harmless_result: list = [None]

    def _run_harmless():
        harmless_result[0] = collect_activations(
            model, tokenizer, harmless[:n_harmless], device,
            progress_cb=lambda done, tot: harmless_q.put((done, tot)),
        )
        harmless_q.put(None)

    threading.Thread(target=_run_harmless, daemon=True).start()
    yield (f"Collecting harmless activations (0/{n_harmless})…",) + no_update
    while True:
        item = harmless_q.get()
        if item is None:
            break
        done, tot = item
        yield (f"Collecting harmless activations ({done}/{tot})…",) + no_update
    harmless_acts = harmless_result[0]

    # --- Compute refusal direction for every layer ---
    yield ("Computing per-layer refusal directions…",) + no_update
    directions = compute_refusal_direction(harmful_acts, harmless_acts)
    default_layer = num_layers // 2
    r = directions[default_layer]
    r = r / r.norm()
    state["refusal_dir"] = r
    state["directions"] = directions
    state["centroid_layer"] = default_layer
    state["harmful_acts"] = harmful_acts
    state["harmless_acts"] = harmless_acts

    # --- PCA plot for the default centroid layer ---
    yield ("Rendering PCA…",) + no_update
    pca_fig = compute_pca_plot(
        harmful_acts[default_layer], harmless_acts[default_layer], default_layer
    )

    # --- Done ---
    layer_max = num_layers - 1
    summary = (
        f"Setup complete!\n"
        f"  Model: {model_name}\n"
        f"  Device: {device}\n"
        f"  Layers: {num_layers}\n"
        f"  Centroid layer (adjustable): {default_layer}\n"
        f"  Harmful train: {n_train}, Harmless: {n_harmless}\n"
        f"  Test prompts available: {len(test_harmful)}"
    )

    yield (
        summary,
        gr.update(maximum=layer_max, value=default_layer),
        gr.update(maximum=layer_max, value=0),
        gr.update(maximum=layer_max, value=layer_max),
        gr.update(choices=test_harmful[:50]),
        pca_fig,
        gr.update(visible=True),
    )


def set_centroid_layer(layer: int):
    if state["directions"] is None:
        return gr.update()
    layer = int(layer)
    r = state["directions"][layer]
    r = r / r.norm()
    state["refusal_dir"] = r
    state["centroid_layer"] = layer
    return compute_pca_plot(
        state["harmful_acts"][layer], state["harmless_acts"][layer], layer
    )


# ---------------------------------------------------------------------------
# Phase 2: chat
# ---------------------------------------------------------------------------
def chat_fn(message, history, layer_start, layer_end, beta_val):
    if not message or not message.strip():
        yield history, gr.update()
        return

    if state["model"] is None:
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "Please complete Phase 1 setup first."},
        ]
        yield history, gr.update()
        return

    # Compute cosine-similarity heatmap for the most recent user sentence
    try:
        heatmap_fig = compute_token_cosine_heatmap(
            state["model"], state["tokenizer"], state["directions"],
            message, state["device"],
        )
    except Exception as e:
        print(f"[heatmap] failed: {e}")
        heatmap_fig = gr.update()

    history = history + [{"role": "user", "content": message}]
    yield history, heatmap_fig

    def _as_text(c):
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            parts = []
            for p in c:
                if isinstance(p, str):
                    parts.append(p)
                elif isinstance(p, dict) and p.get("type") == "text":
                    parts.append(p.get("text", ""))
            return "".join(parts)
        return str(c)

    messages_for_model = [
        {"role": m["role"], "content": _as_text(m["content"])} for m in history
    ]

    partial = ""
    history = history + [{"role": "assistant", "content": ""}]

    for token in generate_with_steering(
        model=state["model"],
        tokenizer=state["tokenizer"],
        messages=messages_for_model,
        refusal_dir=state["refusal_dir"],
        beta=beta_val,
        layer_start=int(layer_start),
        layer_end=int(layer_end),
        device=state["device"],
    ):
        partial += token
        history[-1]["content"] = partial
        yield history, gr.update()


def use_selected_prompt(prompt):
    return prompt


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Refusal Suppression Dashboard") as demo:
    gr.Markdown("# Refusal Suppression Dashboard")
    gr.Markdown(
        "Demonstrates the Single Direction method for suppressing refusal "
        "in open-source LLMs by steering internal activations."
    )

    # --- Phase 1 (always visible) ---
    gr.Markdown("## Phase 1: Setup")
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=list(SUPPORTED_MODELS.keys()),
                    label="Model",
                    value="Llama-3.2-1B-Instruct",
                )
                device_dropdown = gr.Dropdown(
                    choices=["mps", "cuda", "cpu"],
                    label="Device",
                    value=detect_device(),
                )
            hf_token = gr.Textbox(label="HuggingFace Token", type="password")
            setup_btn = gr.Button(
                "Load Model & Compute Refusal Direction", variant="primary"
            )
            setup_status = gr.Textbox(label="Status", interactive=False, lines=8)
        with gr.Column(scale=1):
            pca_plot = gr.Plot(label="PCA of training activations (post-instruction token)")
            centroid_layer_slider = gr.Slider(
                minimum=0, maximum=15, value=0, step=1,
                label="Centroid Layer (refusal direction source)",
            )

    # --- Phase 2 (revealed after Phase 1 completes) ---
    phase2 = gr.Column(visible=False)
    with phase2:
        gr.Markdown("## Phase 2: Chat")
        with gr.Row():
            layer_start_slider = gr.Slider(
                minimum=0, maximum=15, value=0, step=1,
                label="Steering Start Layer",
            )
            layer_end_slider = gr.Slider(
                minimum=0, maximum=15, value=15, step=1,
                label="Steering End Layer",
            )
            beta_slider = gr.Slider(
                minimum=0.0, maximum=10.0, value=1.0, step=0.1,
                label="Steering Strength (β)",
            )

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(label="Chat")
                msg = gr.Textbox(label="Your message", placeholder="Type a message…")
                prompt_dropdown = gr.Dropdown(
                    choices=[],
                    label="Select a harmful test prompt (click to insert)",
                    allow_custom_value=True,
                )
                with gr.Row():
                    send_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear Chat")
            with gr.Column(scale=1):
                heatmap_plot = gr.Plot(
                    label="Cosine similarity of your latest message vs. Δmean per layer"
                )

    # --- Event wiring ---
    setup_btn.click(
        fn=setup_model,
        inputs=[model_dropdown, hf_token, device_dropdown],
        outputs=[
            setup_status, centroid_layer_slider,
            layer_start_slider, layer_end_slider,
            prompt_dropdown, pca_plot, phase2,
        ],
    )

    centroid_layer_slider.release(
        fn=set_centroid_layer,
        inputs=[centroid_layer_slider],
        outputs=[pca_plot],
    )

    prompt_dropdown.change(
        fn=use_selected_prompt,
        inputs=[prompt_dropdown],
        outputs=[msg],
    )

    send_btn.click(
        fn=chat_fn,
        inputs=[msg, chatbot, layer_start_slider, layer_end_slider, beta_slider],
        outputs=[chatbot, heatmap_plot],
    ).then(fn=lambda: "", outputs=[msg])

    msg.submit(
        fn=chat_fn,
        inputs=[msg, chatbot, layer_start_slider, layer_end_slider, beta_slider],
        outputs=[chatbot, heatmap_plot],
    ).then(fn=lambda: "", outputs=[msg])

    clear_btn.click(fn=lambda: [], outputs=[chatbot])


if __name__ == "__main__":
    demo.launch()
