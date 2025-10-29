# ZeptoGPT

A tiny **GPT‑style** transformer that learns the Python-like `sorted()` behavior on `{a,b,c}` — autoregressively — in **~318 trainable parameters**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshbickett/zeptoGPT/blob/main/notebook.ipynb)

---

## TL;DR
- **Task:** sort 3 letters from `{a,b,c}` (duplicates allowed), e.g. `caa → aac`
- **Architecture:** decoder‑only Transformer (masked self‑attention + MLP), 1 block, 2 heads, embed **6** (→ head_dim **3**), **tied LM head**
- **Context:** 7 tokens — `[x1, x2, x3, <sep>, y1, y2, y3]`
- **Training:** next‑token language modeling (teacher forcing) on synthetic data
- **Decoding:** greedy, autoregressive; `<sep>` is forbidden after the separator
- **Params:** **318 trainable** with positional embeddings (or **276** without)
- **Result:** **27/27** correct on all triples over `{a,b,c}` with duplicates

---

## “GPT‑style”?  
Architecturally, this is a **decoder‑only** Transformer trained with a **causal next‑token** objective and used **autoregressively** at inference. It’s “GPT‑style” rather than a pretrained language model (the “P” in GPT).

---

## Autoregressive trace (what the logs mean)

At inference we start from `[x1, x2, x3, <sep>]` and generate **one token at a time**:

