# Implementation Spec: EfficientVLA for the RECAP/SnapFlow Policy

Status: draft
Paper: EfficientVLA — Training-Free Acceleration and Compression for VLA Models
(arXiv:2506.10100). Reported on CogACT: 1.93× speedup, FLOPs → 28.9%, −0.6% success.
Related docs: `docs/analyze_module_spec.md` (provides the measurement tooling),
`research/snapflow_analysis.md` (1-NFE distillation).

## 1. What the paper does

Three training-free components:

1. **Language-module layer pruning.** Importance of layer ℓ =
   `1 − mean cosine similarity(hidden_in, hidden_out)` over a calibration set;
   remove the n lowest-importance layers non-contiguously (paper: 6 of 28).
2. **Task-aware visual token pruning** (from the 2nd transformer layer on).
   Relevance rᵢ = text→vision attention mass onto visual token i, averaged over
   heads. Keep: (a) top `K_key` by relevance (4–8), (b) `⌊α·K_aug⌋` more by
   relevance (α = 0.5), (c) the rest by diversity — greedily maximize
   `1 − max cos(vⱼ, V_selected)`. Paper: 256 → 56–112 tokens per image.
3. **Static temporal caching in the diffusion action head.** Cache self-attn and
   MLP outputs per DiT block; recompute only when `t mod N = 0` (N = 5 over 10
   denoising steps), reuse otherwise.

Paper ablation: tokens-only 1.25×, cache-only 1.23×, layers-only 1.46× (−5.6%
success — the risky one), combined 1.93×.

## 2. Architecture delta: CogACT vs. our stack

The paper's numbers do not transfer directly; our compute profile is different.

| | CogACT (paper) | Ours (SmolVLA RECAP) |
|---|---|---|
| Backbone | 28-layer ~7B LLM | SmolVLM2, **already truncated to first 16 layers** (`num_vlm_layers=16`) |
| Visual tokens | 256/image | **64/camera × 2 cameras = 128** |
| Action head | Separate DiT, 10 diffusion steps | Interleaved action expert (16 layers, cross-attn every 2), 10-step flow matching — **or 1 step with SnapFlow** |
| Prefix reuse | recomputed | prefix computed once, KV-cached; denoise steps only run the expert against it (`sample_actions`, `fill_kv_cache=True`) |
| Extras | — | CFG (2× batch via cond/uncond duplication), KI (VLM frozen in FM path), AR/FAST head (training-only) |

Consequences:

- **Component 3 (denoise caching) is superseded by SnapFlow and must not be
  built.** At 1 NFE there are no steps to cache across; it only ever applies to
  non-distilled `num_steps=10` checkpoints and is TRT-hostile. Explicitly out of
  scope — see §3.3.
- **Components 1–2 compound with SnapFlow** (they shrink the prefix pass and the
  per-step expert cost, SnapFlow shrinks the step count) — this is where our
  headroom is.
- Our token budget is already 2–4.5× smaller than CogACT's, so pruning ratios
  must be re-calibrated, not copied (56 of 256 ≠ 56 of 128).

## 3. Component specs

All inference-time switches live in `SmolVLARECAPConfig` and default to off.
Model-side changes go into `modeling_smolvla_recap.py` / a patched
`smolvlm_with_expert` wrapper — same subclass-and-override pattern the repo
already uses.

### 3.1 C1 — Layer pruning (VLM + expert, paired)

Unlike CogACT, we cannot prune VLM layers independently: expert layer k
consumes the KV of VLM layer k (interleaved, `self_attn_every_n_layers=2`).
Pruning must remove **(VLM layer, expert layer) pairs** and preserve the
self-attn/cross-attn interleaving pattern of the survivors.

```python
# configuration_smolvla_recap.py
pruned_layers: list[int] | None = None   # indices into the 16 kept layers, e.g. [5, 9, 12]
```

- **Importance measurement** is an `analyze` module command
  (`analyze layer-redundancy --checkpoint p --dataset-repo-id d`), computing the
  paper's cosine-similarity score per layer over a calibration set (≥256
  frames), separately for the VLM stream and the expert stream; combined score =
  min of the two normalized scores (a layer is only redundant if it is redundant
  in *both* streams). Output: ranked table → the user picks `pruned_layers`.
- **Mechanism:** drop the pruned layers from the execution list in the
  `VLMWithExpert` forward (both prefix pass and denoise steps). Weights stay in
  the checkpoint (config-driven skip), so one checkpoint serves all ablations.
- **Constraint check at load:** pruning pattern must keep at least one
  cross-attn layer per contiguous run, and never prune layer 0 or the final layer.
- **Contiguous KV re-index (hard requirement for TRT export).** The surviving
  layers must produce a `past_key_values` keyed by **contiguous integers
  0..N−1**, not the original layer numbers with holes. The TRT exporter walks the
  cache as `for i in range(len(cache)): cache[i]...` and the suffix wrapper
  rebuilds `cache[i] for i in range(num_layers)`; a non-contiguous key (e.g. layer
  2 skipped, leaving keys {0,1,3,...}) raises `KeyError` at export. The layer-skip
  implementation must therefore renumber survivors densely and consistently across
  the prefix pass and the denoise step. Add a test asserting
  `list(cache.keys()) == list(range(len(cache)))` under a pruned config. See §6.
- **Caveats:** (a) KI froze the VLM during FM training — the expert was trained
  against *exact* layer-k features, so expect more sensitivity than the paper's
  −5.6%; start with n = 2–3 of 16, not 6 of 28 proportionally. (b) The AR/FAST
  head is not used at inference; its degradation is irrelevant here, but a
  pruned config must never be used for RECAP co-training.
- **Recovery option (non-training-free extension):** if success drops, a short
  phase-2 finetune (expert only, VLM frozen) with the pruned config restores
  most of it; gate this behind measured need.

### 3.2 C2 — Task-aware visual token pruning

Prunes each camera's 64 tokens down to `K` inside `embed_prefix`, *before* the
prefix pass — this shrinks the one-time prefix forward AND the prefix KV that
every denoise step cross-attends to.

```python
# configuration_smolvla_recap.py
visual_tokens_keep: int | None = None    # K per camera, e.g. 24; None = off
token_prune_alpha: float = 0.5           # relevance vs diversity split
token_prune_k_key: int = 4
```

**Scoring pass.** The paper scores tokens with layer-2 text→vision attention.
We get this almost for free: run the first 2 VLM layers on the full prefix,
extract text-query → image-key attention (eager attention, capturable with the
`AttentionRecorder` from `analyze/attribution/attention.py`), compute rᵢ per
image token, select per camera with the paper's three-stage
key/relevance/diversity procedure, then run the **full** prefix pass on the
pruned token set. Overhead: 2 extra shallow layers on the full prefix, repaid by
14 layers on the pruned prefix + all denoise-step cross-attn.

**Selection is per camera** (K tokens from each camera's 64), not global —
guarantees no camera is silently dropped entirely; whole-camera dropping is a
separate decision made from `analyze ablate` results (see analyze spec §5.2).

**Integration points and invariants:**

- Slice `img_emb` + the corresponding `pad_masks`/`att_masks` entries in
  `embed_prefix`; `PrefixLayout` (analyze spec §5.1) must reflect the pruned
  spans so attribution tooling keeps working on pruned models.
- **CFG:** conditioned/unconditioned branches are batch-duplicated
  (`torch.cat([img, img])`). Token selection must be computed once (on the
  conditioned half) and applied to both halves, or the two branches see
  different tokens and the guidance direction is corrupted.
- `prefix_length` padding: pruned prefix is shorter; keep padding logic intact
  (it already pads to fixed length — with pruning, set `prefix_length` to the
  pruned size for real savings, else tokens are pruned and then padded back).
- **Fixed K, always** (no per-frame adaptive count) — keeps shapes static for
  batching, `torch.compile`, and any future TRT export.
- **Temporal amortization (rover extension, off by default):** consecutive
  chunks see near-identical frames; option `token_prune_refresh: int = 1` — 
  recompute the selection every R chunks, reuse indices in between. Cuts the
  scoring pass to 1/R. Validate with the `analyze` method-agreement tooling
  before enabling on the rover.

**Calibration:** `K` is chosen from data, not guessed: run
`analyze ablate`-style offline sweeps (action-MSE vs. K ∈ {16, 24, 32, 48} per
camera) plus the attribution saliency maps to sanity-check *what* is kept.
Acceptance: pick the smallest K whose action MSE vs. the unpruned policy is
within noise of the K=64 baseline on the eval set.

### 3.3 C3 — Denoise-step feature caching — DO NOT IMPLEMENT

**This component is explicitly out of scope and must not be built.** It is listed
only to record the decision so it is not "rediscovered" later as an optimization.

The paper caches self-attn/MLP outputs across diffusion steps and recomputes only
every N-th step. In our stack this is **strictly dominated by SnapFlow**: SnapFlow
runs the suffix at 1 NFE, so there are no steps to cache across (caching 4 of 5
steps of a 10-step loop cannot beat a single step). It only ever applies to a
non-distilled `num_steps > 1` checkpoint — and the plan is that production ships
SnapFlow-distilled.

It is also actively harmful to the deployment target: step-dependent recompute is
**TRT-hostile** (§6) — it forces per-step engine calls with cache tensors threaded
as graph I/O and step-conditional control flow, breaking the clean two-static-pass
shape that SnapFlow + C1 + C2 achieves.

Enforcement: **do not add** an `action_cache_interval` config field, a cache path
in `denoise_step`, or a stub. If a 10-step checkpoint ever genuinely must ship
without distillation, reopen this decision in review first — do not implement it
speculatively.

## 4. File-level plan

```
src/lerobot_policy_smolvla_rl/
├── configuration_smolvla_recap.py   # + pruned_layers, visual_tokens_keep,
│                                    #   token_prune_{alpha,k_key,refresh}
│                                    #   (NO action_cache_interval — see §3.3)
├── efficient_inference.py           # NEW: token scorer/selector, layer-skip
│                                    #   wrapper (dense KV re-index). NO denoise cache.
├── modeling_smolvla_recap.py        # hook points: embed_prefix override,
│                                    #   layer execution list, CFG-consistent selection
└── analyze/
    ├── layer_redundancy.py          # NEW: importance scores (§3.1) + CLI
    └── attribution/…                # existing spec — reused for K calibration
```

Tests (`tests/test_efficient_inference.py`, CPU, mock-policy style of
`test_recap_stages.py`):

- Selection: on a synthetic attention pattern, the three-stage selection returns
  key ∪ relevance ∪ diversity sets with correct sizes and no duplicates; per-camera
  guarantee holds.
- CFG invariant: selection identical across cond/uncond halves.
- Layer skip: forward with `pruned_layers=[k]` ≡ forward of a hand-built model
  without layer k (tiny random-weight config).
- `PrefixLayout` correctness under pruning.
- Dense KV re-index: under a pruned config, `list(cache.keys()) == list(range(N))`
  in both the prefix pass and the denoise step (guards the TRT exporter — §3.1/§6).
- Config validation: illegal `pruned_layers` patterns rejected at load.

## 5. Evaluation protocol

Offline first, robot last:

1. **Action-space metrics** (cheap, per config): action MSE / per-dim MSE vs.
   ground truth and vs. the unmodified policy, per task — this is exactly the
   `analyze ablate` machinery with the efficient-inference config toggled.
2. **Latency/FLOPs:** wall-clock `select_action` on target hardware (rover
   Jetson if applicable, else dev GPU), reported per config as a table:
   baseline / +C2 / +C1 / +C1+C2 / ×SnapFlow. FLOPs via `torch.profiler`.
3. **Closed-loop:** LIBERO suites via the existing eval scripts
   (`run_array_task.py` grid gains an `efficient` axis), then
   `scripts/record_eval.py` on the rover for the final config.
4. **Report:** extend `analyze eval-results` to pivot on the new config axis.

Success criteria (mirrors paper): combined config reaches ≥1.5× wall-clock on
top of SnapFlow with closed-loop success within 2% of baseline; every component
individually toggleable and measured.

## 6. Deployment / TensorRT compatibility

The TRT export/runtime lives in a **separate repo**:
`ros-fhnw-autonomy/.../lerobot_ros/lerobot_ros/trt/` — `exporter.py` (ONNX
export + engine build), `engine.py` (`TRTEngineRunner`), `policy.py`
(`RECAPTRTPolicy` / `RECAPSnapflowTRTPolicy` runtime), `validate.py` (per-config
accuracy check vs. PyTorch). It splits SmolVLA into a **prefix engine**
(images + language + state → KV cache) and a **suffix engine** (one denoise/
velocity step against the cached KV); the denoise loop and CFG blend run on the
host in Python. This structure maps cleanly onto EfficientVLA. Verified against
the code (2026-07):

- **C1 layer pruning — near-free, one hard constraint.** The exporter derives
  `num_layers` from the prefix's KV-cache output count
  (`num_layers = num_cache_tensors // 2`) and the suffix wrapper takes it
  explicitly, so removing layers propagates to both engines with **no TRT edits —
  just a re-export**. The constraint is the **contiguous KV re-index** in §3.1:
  the exporter walks the cache as `cache[i] for i in range(len(cache))`, so the
  surviving layers must be renumbered `0..N−1` with no holes or export raises
  `KeyError`.
- **C2 token pruning — the one real integration, and it belongs in-graph.**
  `torch.onnx.export` is called with **no `dynamic_axes`**, so batch (=1), prefix
  length and KV seq-length are all frozen at export — which is exactly what
  fixed-K pruning wants. Put the layer-2 scoring + TopK + Gather selection inside
  `SmolVLAPrefixWrapper.forward` (TRT supports TopK/Gather natively); the suffix
  engine then re-exports automatically against the shorter KV length. The
  `token_prune_refresh > 1` variant instead computes indices host-side and needs
  image preprocessing split into its own engine — more engines, only worth it if
  the scorer shows up in latency.
- **Because there are no `dynamic_axes`, every EfficientVLA config variant (each K,
  each `pruned_layers` set) requires a fresh export + engine rebuild.** Build the
  sweep tooling around that, and **add a TRT timing cache to
  `build_trt_engine`** — you will rebuild engines many times across configs and it
  cuts rebuild time substantially.
- **C3 caching — must not be built (see §3.3).** It is step-dependent control flow
  (TRT-hostile: per-step engine calls with cache tensors as I/O) and strictly
  dominated by SnapFlow, which the host loop already runs at 1 step.

**Target end state: SnapFlow (1 NFE) + C1 + C2 collapses inference to two
static-graph passes (prefix, suffix) — the ideal TRT shape.** The existing TRT
code was checked for correctness (CFG gating, SnapFlow clamp/step, export hooks
all match the reference model) and needs **no corrective changes**; the
EfficientVLA work is purely additive on top of it. Re-run `validate.py` per
config (FP16 is on by default; harmless at 1 NFE, but every shipped config must
pass its max-abs-error check against the PyTorch reference).

## 7. Rollout phases

| Phase | Work | Gate |
|---|---|---|
| 1 | C2 token pruning + `analyze layer-redundancy` + tests | offline action-MSE sweep picks K |
| 2 | C1 layer pruning (config-driven skip) | redundancy table + offline MSE; decide n |
| 3 | Combined C1+C2 on SnapFlow checkpoint; latency table; LIBERO grid | ≥1.5×, ≤2% success drop |
| 4 | Rover deployment (`record_eval.py`), optional `token_prune_refresh` | field validation |
| — | C3 denoise caching — **do not implement** (§3.3); reopen in review only if a non-distilled 10-step checkpoint must ship | — |

## 8. Risks

- **KI-trained expert sensitivity to layer pruning** (features it was trained
  on disappear) — mitigated by paired pruning, small n, and the finetune escape hatch (§3.1).
- **Token pruning under distribution shift** (rover scenes ≠ calibration data):
  diversity component helps, but validate per-task; the per-camera K floor
  prevents catastrophic single-camera blindness.
- **CFG-selection mismatch** is a silent correctness bug — covered by a dedicated test.
- **Interaction with SnapFlow distillation:** distillation was trained on
  unpruned prefixes. Measure C2 on the distilled checkpoint (expected fine —
  prefix is input conditioning, not part of the distilled dynamics), but if MSE
  jumps, re-distill with pruning enabled (cheap: ~12h, single GPU).
