# ZeptoGPT

A tiny GPT‑style transformer that learns the Python‑like `sorted()` behavior on `{a,b,c}` — autoregressively — in **~318 trainable parameters**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshbickett/zeptoGPT/blob/main/notebook.ipynb)

---

## TL;DR
- Task: sort 3 letters from `{a,b,c}` (duplicates allowed), e.g. `caa → aac`
- Architecture: decoder‑only Transformer (masked self‑attention + MLP), 1 block, 2 heads, embed **6** (→ head_dim **3**), tied LM head
- Context: 7 tokens — `[x1, x2, x3, <sep>, y1, y2, y3]`
- Training: next‑token language modeling (teacher forcing) on synthetic data
- Decoding: greedy, autoregressive; `<sep>` is forbidden after the separator
- Params: **318 trainable** with positional embeddings (or **276** without)
- Result: **27/27** correct on all triples over `{a,b,c}` with duplicates

---

## “GPT‑style”?
Architecturally, this is a decoder‑only Transformer trained with a causal next‑token objective and used autoregressively at inference. It’s “GPT‑style” rather than a pretrained language model (the “P” in GPT).

---

## Autoregressive trace (what the logs mean)

At inference we start from `[x1, x2, x3, <sep>]` and generate one token at a time:

start [c, a, a, <sep>]  (heads=2, head_dim=3)
  step1: p(a)=0.999, p(b)=0.000, p(c)=0.000 -> 'a'
  step2: p(a)=0.998, p(b)=0.000, p(c)=0.002 -> 'a'
  step3: p(a)=0.000, p(b)=0.000, p(c)=1.000 -> 'c'
  full [c, a, a, <sep>, a, a, c] -> outputs [a, a, c]

Each step shows the next‑token distribution and the greedy choice. The printed attention rows per head show how the last position attends to the inputs and previous outputs (useful to see “counting” behavior emerge).

---

## Parameter count (E=6, H=2, MLP×1)
- token embedding: `4 × 6 = 24`
- positional embedding: `7 × 6 = 42`  (toggle off to save 42 params)
- attention (q, k, v, proj): `4 × (6 × 6) = 144`
- MLP (×1): `6 × 6 + 6 × 6 = 72`
- LayerNorms (ln1, ln2, final): `3 × (2 × 6) = 36`
- LM head: weight‑tied → `0`

**Total:** 318 (or 276 without positional embeddings)

---

## Run it

**Colab (one click):**
https://colab.research.google.com/github/joshbickett/zeptoGPT/blob/main/notebook.ipynb

**Local**
~~~bash
python micro_abc_sorter_e6.py
~~~

The script trains, evaluates on all 27 triples, prints AR traces, and exposes a simple text box (in notebooks) to try your own inputs.

---

## Variants

| Variant                                | Params | Notes                                  |
|----------------------------------------|-------:|----------------------------------------|
| E=6, H=2, MLP×1, pos=ON                |   318  | Default. Robust 27/27.                 |
| E=6, H=2, MLP×1, pos=OFF               |   276  | Usually 27/27; may need more steps.    |
| E=4, H=2, MLP×1, pos=OFF               |   136  | Finicky; can fail on duplicates.       |
| E=8, H=2, MLP×2, pos=ON                |   648  | Very reliable, converges fast.         |

---

## Inspiration
- LLM visualization — https://bbycroft.net/llm

---

## License
MIT (or your choice)
