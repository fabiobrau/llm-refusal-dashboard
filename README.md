# LLM Refusal Dashboard

An interactive Gradio dashboard that demonstrates the **Single Direction** refusal-suppression method from Arditi et al. on small open-source instruction-tuned LLMs (Llama-3.2, Qwen-2.5). The app computes a per-layer *refusal direction* from harmful/harmless prompt activations, visualises it, and lets you chat with the model while the direction is ablated from the residual stream at user-chosen layers and strength.

> **Educational use only.** This project exists to illustrate how activation steering works on a safety-relevant behaviour. It is not a tool for producing harmful content, and it is not intended for deployment.

---

## What it does

1. **Phase 1 — Setup.** Pick a supported model, load it on MPS / CUDA / CPU, download harmful prompts (AdvBench) and harmless prompts (Alpaca), and collect per-layer hidden states at the post-instruction token. The difference of class means `Δ = μ_harmful − μ_harmless` at each layer is cached. A PCA scatter of the two classes' activations is rendered; a slider lets you pick the **centroid layer** whose Δ becomes the steering direction `r`.

2. **Phase 2 — Chat.** Talk to the model with steering active. Three controls:
   - **β (strength)** — scales the ablation.
   - **Steering start / end layer** — the residual-stream projection `h ← h − β · (r·h) · r` is applied inside that layer range at every forward pass.

   Alongside the chat, a heatmap visualises `cos(h_l(token), Δ_l)` for each word of your latest message across all layers — a quick read on which tokens / layers the base model already represents in a "refusal-like" way.

See [docs/refusal.md](docs/refusal.md) for the method and [docs/init.md](docs/init.md) for the original design notes.

## Screenshots

**Phase 1 — PCA of post-instruction activations, centroid layer picker:**

![Phase 1](docs/screenshots/phase1.png)

**Phase 2 — Chat with steering, per-token cosine-similarity heatmap:**

![Phase 2](docs/screenshots/phase2.png)

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Gradio prints a `http://127.0.0.1:7860` URL. For a public share link, replace the last line with `demo.launch(share=True)` in [app.py](app.py).

### Model access

- **Qwen-2.5** models are open — no token needed.
- **Llama-3.2** models are gated. Request access on the model's HuggingFace page, generate a read token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), and paste it into the "HuggingFace Token" field in Phase 1.

## Project layout

```
.
├── app.py              # Gradio UI + orchestration
├── steering.py         # Activation collection, refusal direction, steering hooks, plots
├── requirements.txt
└── docs/
    ├── refusal.md          # The Single Direction method
    ├── init.md             # Original design spec
    ├── harmful_prompts.md  # Dataset sources
    └── screenshots/        # README images
```

## Credits & disclaimer

- Method: Arditi et al., *"Refusal in Language Models Is Mediated by a Single Direction"* (2024).
- Harmful prompts: [llm-attacks/AdvBench](https://github.com/llm-attacks/llm-attacks).
- Harmless prompts: [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca).
- Models: `meta-llama/Llama-3.2-*-Instruct`, `Qwen/Qwen2.5-*-Instruct`.

**This repository is released strictly for educational and research purposes** — to help people understand how modern safety training is represented in model activations and how fragile a *single* linear direction can be. Do not use it to generate content that you would not be willing to sign your name to. The author takes no responsibility for misuse.
