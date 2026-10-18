# Do Safety Probes Survive KV-Cache Compression?

Code and aggregate results for a hobby AI-safety project by Rahul Krishnan. It tests whether a linear-probe safety monitor, trained on an uncompressed model, still detects harmful requests when the model's KV cache is compressed. The project extends JoLT, a low-rank KV-cache compression method ([arXiv:2607.12550](https://arxiv.org/abs/2607.12550)), and compares it with SnapKV token eviction.

Full write-up: [blog post](BLOG_URL_TBD)

## Setup

- Models: Qwen3 0.6B, 1.7B, 4B and 8B, thinking off.
- Compression: JoLT (low-rank) and SnapKV (eviction) at 2x, 4x, 5x, 6x and 8x. Each request follows a 1024-token benign document, and the prompt is compressed before the last prompt token is processed.
- Monitor: a logistic-regression probe on the residual stream at the assistant-start position, trained on the uncompressed model and then frozen.
- Data: training and validation splits from Arditi et al. (2024). Evaluation uses HarmBench and JailbreakBench (258 harmful), Alpaca (142 harmless) and XSTest (100 safe).
- Behaviour: judged by Qwen3Guard-Gen-4B. A blind audit found that its "unsafe" labels overcount harmful compliance, so the reported unsafe rates are upper bounds.
- Controls:
  - Gaussian noise matched to JoLT's per-layer error;
  - random eviction;
  - a which-tokens counterfactual (compress only the document, or only the request);
  - refusal-direction projection and steering;
  - a probe retrained on compressed activations.

## Main results

- JoLT at 2x, and SnapKV at every ratio up to 8x, match the uncompressed model on probe AUROC and behaviour for all four models.
- JoLT breaks generation between 4x and 6x, and later for larger models. Matched-error noise does much less damage, so the structure of JoLT's error matters, not its size.
- Most of JoLT's damage comes from compressing the long document, not the request.
- The frozen probe keeps detecting harmful requests while outputs stay fluent (Qwen3-8B at JoLT 5x: 96.9% of harmful prompts flagged). It fails mainly when the request itself is lost from the cache.

Tables for every method, ratio and model are in [analysis/sweep.md](analysis/sweep.md), with per-model summaries in `analysis/Qwen3-*/summary.md`.

## Layout

- `kvsafety/`: SnapKV cache, JoLT cache wrapper and the experiment pipeline (data, run, controls driver, judge, analysis).
- `kv_compress/`: the author's JoLT code, vendored with a Qwen3 fix (see `kv_compress/VENDORED.md`).
- `scripts/`: job runner, results merge and supplementary analysis.
- `tests/`: CPU tests on tiny random models.
- `data/raw/`: public prompt sets and the WikiText-2 filler text.
- `research_question_and_methodology.md`: the full design.

## Running

```bash
pip install -r requirements.txt
python -m kvsafety.pipeline.data
python -m kvsafety.pipeline.run_controls --model-dir <Qwen3 dir> --set eval --ratio 5 --variants jolt,snapkv,noise,random
python -m kvsafety.pipeline.judge --judge-dir <Qwen3Guard-Gen-4B dir> results/<model>/eval_*_f1024.jsonl
python -m kvsafety.pipeline.analyze --results-dir results/<model>
pytest tests
```
