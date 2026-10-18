# Research question and methodology

Working title: "Do Safety Probes Survive KV-Cache Compression? Evidence from Low-Rank and Eviction Methods"

## Research question

When KV-cache compression weakens a language model's safety behaviour, do linear-probe safety monitors trained on the uncompressed model still detect harmful requests and unsafe responses, and how does this depend on the compression method, the compression ratio and model scale?

The paper is a safety extension of JoLT (Joint Lagrangian Tucker, low-rank KV-cache compression). Safety behaviour under JoLT and monitor reliability under JoLT are both headline results. SnapKV token eviction is the comparison method, so the paper can say whether an effect is specific to JoLT or common to KV-cache compression.

Motivation: labs deploy both compressed inference and probe-based monitoring. Anthropic's Constitutional Classifiers++ (arXiv 2601.04603) uses linear probes on model activations in a deployed classifier. Nobody has tested whether the two practices are compatible. If compression silently blinds the monitor, that is a safety gap that standard benchmarks do not show.

## Sub-questions

1. Behaviour. How do refusal rates on harmful prompts, and over-refusal on harmless prompts, change as KV-cache compression increases, when judged by a safety classifier rather than by keyword matching?
2. Monitoring. How does the AUROC of a residual-stream safety probe change under compression, both when the probe is trained on the uncompressed model and reused unchanged, and when it is retrained on compressed activations?
3. Silent failures. How often does the model comply with a harmful request while the probe stays below threshold, and does this silent-failure rate grow faster than the drop in refusals alone?
4. Method comparison. At matched memory, does JoLT damage safety behaviour and probe reliability less than SnapKV?
5. Scale. Do the effects in 1 to 4 change across model sizes within one model family?

## Hypotheses

- H1: Refusal behaviour stays near baseline at moderate compression and degrades at high compression, more so for SnapKV.
- H2: Probe accuracy degrades earlier than refusal behaviour, because moderate compression already changes internal representations while outputs still look normal.
- H3: At matched memory, JoLT preserves safety-relevant directions better than SnapKV, because it keeps an approximation of every token instead of dropping some entirely.

A clean negative result is also informative: if probes stay reliable where behaviour degrades, that is evidence that probe monitors are robust to compressed inference. Prior work found frozen probes mostly robust to weight quantization (arXiv 2606.15980), so a null or mild probe effect is a plausible outcome.

## Related work and novelty

| Paper | What it does | Overlap |
|---|---|---|
| Alignment Collapse Under KV Cache Quantization (arXiv 2606.09864, ICML 2026) | KV quantization 2 to 8 bit and FP8 KV on 11 models, 3.8B to 72B, WildGuard-7B judge, XSTest over-refusal. "Mistral-7B loses 15.2% of its refusals at only 1.03x perplexity". No probes; lists eviction and low-rank compression as future work | Sub-question 1 for KV quantization |
| AnchorKV (arXiv 2606.17872) | Llama-3.1-8B under FastKV eviction at 25% retention; keyword-judged jailbreak ASR rises "from 0.299 to 0.632" under a transfer attack. Uses a harm direction for token scoring, not as a monitor | Sub-question 1 for one eviction setting |
| On the Efficiency-Safety Dilemma in Large Reasoning Models (arXiv 2609.23587) | Reasoning models (QwQ-32B, R1-Distill) under weight quantization, pruning, SnapKV and PyramidKV at one setting each; apparent safety gains come from "attempted but failed" responses. No probes | Sub-question 1 for eviction on reasoning models |
| Silent Alarm (arXiv 2607.12792) | Qwen3 1.7B, 4B, 8B under weight quantization, thinking off; a Jacobian-lens readout, not a trained probe; recognition vs behaviour discussed conceptually | Model family; sub-question 3 in concept; no KV compression |
| Do Activation Monitors Survive Model Updates? (arXiv 2606.15980) | Frozen linear probes under weight quantization and LoRA updates: "quantization-style updates largely preserve frozen probe performance" | Design precedent for sub-question 2; no KV compression |

The claim that KV-cache compression can erode refusals while standard metrics look fine is established by this prior work and is cited, not claimed. New in this paper:
- safety behaviour under low-rank KV-cache compression;
- a residual-stream probe trained on the uncompressed model and tested under KV-cache compression;
- the silent-failure rate as a measured per-prompt quantity;
- low-rank vs eviction at matched memory, for safety;
- the Qwen3 scale axis under KV-cache compression.

## Models

Qwen3 dense: 0.6B, 1.7B, 4B, 8B, loaded from local model directories, bf16, batch size 1.

Thinking mode is off (`enable_thinking=False` in the chat template) and stated in the paper. The setting changes safety results strongly: on Qwen3-32B, harmful compliance is 25.0% with thinking on and 72.2% with it off (arXiv 2607.13346), and the misalignment rate is 23.12% on vs 7.63% off (arXiv 2509.00544, Table 1). Silent Alarm, the closest prior work on Qwen3, also turns thinking off.

## Compression methods

| Method | Type | Implementation |
|---|---|---|
| JoLT | low-rank Tucker decomposition plus rotated low-bit residual, joint Lagrangian rank and bit allocation | vendored `kv_compress/`, `kvsafety/jolt_cache.py` |
| SnapKV | token eviction (Li et al. 2024, arXiv 2404.14469) | `kvsafety/snapkv_cache.py` |

Both compress only the prompt KV, once, and keep generated tokens dense. Memory is matched on persistent prompt-KV bytes measured per prompt, not on the nominal ratio. Ratios: 2x, 4x, 5x, 6x and 8x.

SnapKV details: observation window 32 tokens, max-pool kernel 7. Scores are the softmax attention of the post-RoPE window queries over all prompt keys, with a causal mask inside the window, summed over the window queries for the prefix columns. For GQA the scores are averaged over the query heads of each KV group, so selection is per KV head. Each KV head keeps max(32, round(T / r)) tokens, the window always included, in original order. Token positions stay logical after eviction, so a new token at step j gets RoPE position T + j.

## Protocol: compress, then finish

If compression only starts after the full prompt is processed, the prompt-token activations and the first response token are identical to the uncompressed model by construction. To avoid this:

1. Prefill the chat-formatted prompt except its last token with the compressing cache.
2. Feed the last prompt token (the assistant-turn start) through the compressed cache. The probe reads the residual stream at this position.
3. Generate the response greedily, 64 new tokens, with the compressed prompt KV and a dense decode tail.

Every request follows a benign 1024-token document (WikiText-2 raw test text, a different chunk per prompt, identical across methods and ratios). The user message is "Here is a document:", the document, then the request. Compression is deployed for long contexts, and short prompts cannot reach high eviction ratios because SnapKV always keeps its 32-token window.

## Data

Following Arditi et al. 2024 (arXiv 2406.11717), which uses "harmful instructions drawn from AdvBench, MaliciousInstruct, TDC2023, and HarmBench" and harmless instructions from Alpaca:

- Train: 128 harmful and 128 harmless prompts from the Arditi training splits.
- Validation (layer choice and threshold): 32 harmful and 32 harmless prompts from the Arditi validation splits.
- Evaluation, disjoint from training: HarmBench test (159) and JailbreakBench (100) harmful prompts vs 259 Alpaca test prompts; XSTest (250 safe prompts as hard negatives, 200 unsafe prompts). Evaluation prompts that duplicate a training or validation prompt after normalisation are removed.

Trailing punctuation is stripped from every prompt, because AdvBench vs Alpaca is separable by punctuation alone ("a trivial punctuation classifier nearly perfectly separates the two sources", arXiv 2604.18901). A probe that reads surface cues would survive compression trivially and fake a robustness result.

## Measurements

- Probe: logistic regression on the residual-stream activation at the probe position, standardised features, trained on uncompressed training activations; layer chosen by validation AUROC. A difference-in-means direction (Arditi et al.) at the same layer is reported alongside. The probe stays frozen for every compressed condition.
- Monitoring: AUROC of the frozen probe on each condition's evaluation activations, overall and per source.
- Text-only baseline: bag-of-words logistic regression on the prompt text, reported next to the probe.
- Behaviour: Qwen3Guard-Gen-4B in response-moderation mode labels each response for safety (Safe, Unsafe, Controversial) and refusal (Yes, No). Refusal rate on harmful prompts, over-refusal on XSTest safe prompts, unsafe-response rate.
- Silent failure: a harmful prompt whose response is judged Unsafe while the probe score is below the threshold that gives 5% false positives on uncompressed validation harmless prompts.
- Uncertainty: 95% bootstrap confidence intervals, 1000 resamples.

## Controls

- Error-matched noise. For each prompt and ratio, JoLT is run once to measure the relative reconstruction error at every layer, separately for keys and values: ||K_hat - K|| / ||K|| and ||V_hat - V|| / ||V||. The uncompressed prompt K and V then receive Gaussian noise scaled so that each layer's relative error equals JoLT's. The model runs on this noisy cache under the same protocol, judge and frozen probe. Both caches are wrong by the same amount at every layer; JoLT's error is structured (it drops low-energy directions), the noise is not. If noise damages behaviour and probe AUROC as much as JoLT, the effect depends only on the size of the error; if JoLT damages them less, its low-rank structure preserves safety-relevant directions (H3).

- Which tokens matter (counterfactual). Two variants per method at each ratio: (a) only the benign document is compressed, the request and the assistant-start tokens stay exact; (b) only the request is compressed, the document stays exact. If damage appears mainly when the request is compressed, compression hurts by corrupting the model's view of the request; if it appears when only the document is compressed, it damages the context the refusal decision relies on.
- Refusal-direction intervention. The refusal direction is the difference between mean harmful and mean harmless uncompressed training activations at the probe layer (Arditi et al. 2024). (a) Projection: for every evaluation activation, its projection onto this direction, compared between dense and compressed conditions. (b) Steering: under compression, the direction is added at the probe layer during generation on harmful prompts at two strengths, and refusal is judged as usual. If a small strength restores dense-level refusal, the mechanism is intact and only its input signal is weaker; if large strengths fail, compression damages computation downstream of the direction.
- Random eviction. At each ratio, a random set of prompt tokens is kept per KV head (seeded per prompt), with the 32-token window always kept, instead of the tokens SnapKV's attention scores select. If random eviction hurts safety as much as SnapKV, the number of dropped tokens matters, not which; if SnapKV is safer, its selection keeps safety-relevant tokens; if SnapKV is worse, its selection discards refusal-relevant tokens.
- Retrained probe. The training and validation prompts also run under every compressed condition, and a probe is trained and its layer chosen on those compressed activations, then tested on compressed evaluation activations. If the retrained probe recovers dense-level AUROC, the harm information survives and only moved; if it does not, compression destroyed the information at the probed layers.

## Experimental grid

- All four models at 2x, 4x and 8x: JoLT, SnapKV, error-matched noise, random eviction, refusal-direction projection, frozen probe, judge.
- Qwen3-1.7B and Qwen3-8B at 8x additionally: which-tokens counterfactual, steering at two strengths on harmful prompts, retrained probe.

## Scope of this version

Not included: a response-level probe and a thinking-on ablation.
