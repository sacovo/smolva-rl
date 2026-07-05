# EfficientVLA for the RECAP/SnapFlow Policy

Status: **implemented** (C1 layer pruning, C2 visual-token pruning; C3 deliberately dropped in favor of SnapFlow — see `docs/snapflow.md`).

Paper: EfficientVLA — Training-Free Acceleration and Compression for VLA Models (arXiv:2506.10100). Reported on CogACT: 1.93× speedup, FLOPs → 28.9%, −0.6% success.

Related docs: `docs/analyze_module.md` (measurement/calibration tooling), `docs/snapflow.md` (1-NFE distillation — supersedes the paper's C3).

This document describes the two training-free acceleration components that are implemented and tested in the `SmolVLARECAP` policy — **C1 paired layer pruning** and **C2 task-aware visual-token pruning** — plus the deliberate decision *not* to build the paper's third component (denoise-step feature caching). Both C1 and C2 are inference-time switches that live in `SmolVLARECAPConfig`, default to off, and compound on top of SnapFlow's 1-NFE suffix.

## 1. What the paper does, and what we took from it

The paper defines three training-free components:

1. **Language-module layer pruning.** Importance of layer ℓ = `1 − mean cosine similarity(hidden_in, hidden_out)` over a calibration set; remove the n lowest-importance layers non-contiguously (paper: 6 of 28). — **We implemented this as C1**, adapted to paired (VLM, expert) removal.
2. **Task-aware visual token pruning** (from the 2nd transformer layer on). Relevance rᵢ = text→vision attention mass onto visual token i, averaged over heads. Keep: (a) top `K_key` by relevance, (b) `⌊α·K_aug⌋` more by relevance (α = 0.5), (c) the rest by diversity — greedily maximize `1 − max cos(vⱼ, V_selected)`. Paper: 256 → 56–112 tokens per image. — **We implemented this as C2**, per camera.
3. **Static temporal caching in the diffusion action head.** Cache self-attn and MLP outputs per DiT block; recompute only every N-th step. — **Deliberately dropped** (see §5); SnapFlow runs the suffix at 1 NFE, so there are no steps to cache across.

Paper ablation (context only, does not transfer directly): tokens-only 1.25×, cache-only 1.23×, layers-only 1.46× (−5.6% success — the risky one), combined 1.93×.

## 2. Architecture delta: CogACT (paper) vs. our stack

The paper's numbers do not transfer directly; our compute profile is different.

| | CogACT (paper) | Ours (SmolVLA RECAP) |
|---|---|---|
| Backbone | 28-layer ~7B LLM | SmolVLM2, **already truncated to first 16 layers** (`num_vlm_layers=16`) |
| Visual tokens | 256/image | **`SIGLIP_TOKENS_PER_CAMERA = 64`/camera × 2 cameras = 128** (SigLIP 8×8 grid after pixel-shuffle) |
| Action head | Separate DiT, 10 diffusion steps | Interleaved action expert (16 layers, cross-attn every 2), 10-step flow matching — **or 1 step with SnapFlow** |
| Prefix reuse | recomputed | prefix computed once, KV-cached; denoise steps only run the expert against it (`sample_actions`, `fill_kv_cache=True`) |
| Extras | — | CFG (2× batch via cond/uncond duplication), KI (VLM frozen in FM path), AR/FAST head (training-only) |

Consequences that shaped the implementation:

- **The paper's C3 (denoise caching) is superseded by SnapFlow and was not built.** At 1 NFE there are no steps to cache across; it only ever applies to non-distilled `num_steps=10` checkpoints and is TRT-hostile. Out of scope — see §5.
- **C1–C2 compound with SnapFlow** (they shrink the prefix pass and the per-step expert cost, SnapFlow shrinks the step count) — this is where the headroom is.
- Our token budget is already 2–4.5× smaller than CogACT's, so pruning ratios are re-calibrated from data, not copied (56 of 256 ≠ 56 of 128).

## 3. C1 — Paired layer pruning (VLM + expert)

Unlike CogACT, VLM layers cannot be pruned independently: expert layer k consumes the KV of VLM layer k (interleaved, `self_attn_every_n_layers=2`). Pruning removes **(VLM layer, expert layer) pairs** and preserves the self-attn/cross-attn interleaving pattern of the survivors.

Config field (in `configuration_smolvla_recap.py`):

```python
pruned_layers: list[int] | None = None   # Paired layers to skip (e.g. [5, 9, 12]); None = off
```

**Mechanism.** `SmolVLARECAP.__init__` monkeypatches the wrapper's decoder loop with `make_patched_forward` (in `efficient_inference.py`), which iterates the surviving layers only (`if layer_idx in pruned_layers: continue`) while preserving every o_proj, layernorm, MLP, and residual connection. Weights stay in the checkpoint (config-driven skip), so one checkpoint serves all ablations.

**Constraint check at load** — `check_pruning_constraints(pruned_layers, num_layers, self_attn_every_n_layers=2)`:
- Indices must be in range `[0, num_layers − 1]`.
- Layer 0 and the final layer (`num_layers − 1`) may never be pruned (preserve input embedding and final projection dynamics).
- Every contiguous run of *kept* layers must contain at least one cross-attention layer (an odd index, since `self_attn_every_n_layers=2`), so the expert can still attend to VLM context.

`SmolVLARECAPPolicy.__init__` threads `config.pruned_layers` onto `vlm_with_expert` and calls this check at construction; illegal patterns raise `ValueError` at load.

**Contiguous dense KV re-index (hard requirement for TRT export).** Surviving layers must produce a `past_key_values` keyed by **contiguous integers `0..N−1`**, not the original layer numbers with holes. The exporter walks the cache as `cache[i] for i in range(len(cache))`, so a hole raises `KeyError`. Implemented via `_layer_idx_to_cache_idx` (built in `__init__` from the kept-layer list) plus `make_kv_reindex_wrapper`, which wraps `forward_attn_layer`/`forward_cross_attn_layer`: the original layer index is passed unchanged for weight lookup, but the cache read/write is translated to the dense `cache_idx`. This holds across both the prefix pass and the denoise step and is asserted by tests (see §9).

**Patch ordering.** The KV-reindex wrappers are applied *after* the KI patch, so they wrap the KI-patched attention functions rather than being clobbered by them.

**Caveats:**
- KI froze the VLM during FM training — the expert was trained against *exact* layer-k features, so expect more sensitivity than the paper's −5.6%; start with n = 2–3 of 16, not 6 of 28 proportionally.
- The AR/FAST head is not used at inference; its degradation is irrelevant here, but a pruned config must never be used for RECAP co-training (enforced — see §7).
- **Recovery option (non-training-free extension):** if success drops, a short phase-2 finetune (expert only, VLM frozen) with the pruned config restores most of it; gate this behind measured need.

**Choosing `pruned_layers`.** Importance is measured offline by the `analyze layer-redundancy` command (`src/lerobot_policy_smolvla_rl/analyze/layer_redundancy.py`):

```bash
analyze layer-redundancy --checkpoint <checkpoint_path> --dataset-repo-id <dataset_name>
```

It records each layer's `hidden_in`/`hidden_out` over a calibration set (≥256 frames), computes the paper's per-layer cosine-similarity score separately for the VLM and expert streams, normalizes each to `[0, 1]`, and combines them as `min(R_vlm, R_exp)` (a layer is only redundant if it is redundant in *both* streams). It prints a ranked table (highest redundancy first) from which the user picks `pruned_layers`.

## 4. C2 — Task-aware visual token pruning

C2 prunes each camera's `SIGLIP_TOKENS_PER_CAMERA = 64` tokens down to `visual_tokens_keep` (K per camera) inside `embed_prefix`, *before* the prefix pass — this shrinks the one-time prefix forward AND the prefix KV that every denoise step cross-attends to. It drops `SIGLIP_TOKENS_PER_CAMERA − visual_tokens_keep` tokens from each camera.

Config fields (in `configuration_smolvla_recap.py`):

```python
visual_tokens_keep: int | None = None    # K visual tokens to keep per camera; None = off
token_prune_alpha: float = 0.5           # relevance vs. diversity split
token_prune_k_key: int = 4               # key visual tokens count
token_prune_refresh: int = 1             # recompute selection every R chunks (temporal amortization)
```

**Scoring pass.** The paper scores tokens with layer-2 text→vision attention; we get this almost for free. `select_visual_tokens` (in `efficient_inference.py`) runs the first 2 VLM layers on the full prefix with `use_cache=False` (a throwaway cache, so layer-0/1 KV never pollutes the persistent cache), capturing layer-1 attention via `make_eager_attention_hook` (the wrapper records `probs` when `_current_layer_idx == 1`). It then computes rᵢ per image token and selects per camera with the three-stage procedure, after which the **full** prefix pass runs on the pruned token set. Overhead: 2 extra shallow layers on the full prefix, repaid by the remaining layers on the pruned prefix + all denoise-step cross-attn.

**Three-stage selection** (`select_diverse_tokens`): with `K_relevance = K_key + ⌊α·(K_keep − K_key)⌋`,
1. take the top `K_relevance` tokens by relevance;
2. greedily add the rest by diversity, each step picking the candidate that minimizes its maximum cosine similarity to the already-selected set (`d_j = 1 − max_{k∈S} cos(v_j, v_k)`);
3. sort the selected indices to preserve spatial layout order.

**Selection is per camera** (K tokens from each camera's 64), not global — this guarantees no camera is silently dropped entirely. Whole-camera dropping is a separate decision made from `analyze ablate` results.

**Integration invariants:**
- `embed_prefix` slices `img_emb` and the corresponding `pad_masks`/`att_masks`; `PrefixLayout`/`prefix_layout` reflects the pruned spans so attribution tooling keeps working on pruned models.
- **CFG.** Conditioned/unconditioned branches are batch-duplicated. When `_in_cfg_mode`, selection is computed on the conditioned half only and the indices are repeated for the unconditioned half, so both branches see identical tokens and the guidance direction is not corrupted.
- **`prefix_length`.** For real FLOP savings the pruned prefix must actually be shorter, so `SmolVLARECAPPolicy.__init__` recomputes `model.prefix_length` as `config.prefix_length − (SIGLIP_TOKENS_PER_CAMERA − visual_tokens_keep) * num_cameras`. Without this the pruned sequence is padded back to the baseline length and yields zero savings.
- **Fixed K, always** (no per-frame adaptive count) — keeps shapes static for batching, `torch.compile`, and TRT export.
- Pruning is applied only at inference (`not self.training`) and can be bypassed for the scoring pass (`bypass_pruning=True`).
- **Temporal amortization (rover extension, off by default):** consecutive chunks see near-identical frames; `token_prune_refresh = R` recomputes the selection every R chunks and reuses indices in between, cutting the scoring pass to 1/R. Validate with the method-agreement tooling before enabling on the rover.

**Calibrating K.** K is chosen from data, not guessed: run `analyze ablate`-style offline sweeps (action-MSE vs. K ∈ {16, 24, 32, 48} per camera) plus attribution saliency maps to sanity-check *what* is kept. Acceptance: pick the smallest K whose action MSE vs. the unpruned policy is within noise of the K=64 baseline on the eval set.

## 5. C3 — Denoise-step feature caching — deliberately dropped

**Not implemented, by design.** It is recorded here only so the decision is not "rediscovered" later as an optimization.

The paper caches self-attn/MLP outputs across diffusion steps and recomputes only every N-th step. In our stack this is **strictly dominated by SnapFlow**: SnapFlow runs the suffix at 1 NFE, so there are no steps to cache across (caching 4 of 5 steps of a 10-step loop cannot beat a single step). It only ever applies to a non-distilled `num_steps > 1` checkpoint — and production ships SnapFlow-distilled (`docs/snapflow.md`).

It is also actively harmful to the deployment target: step-dependent recompute is **TRT-hostile** (§8) — it forces per-step engine calls with cache tensors threaded as graph I/O and step-conditional control flow, breaking the clean two-static-pass shape that SnapFlow + C1 + C2 achieves.

There is intentionally **no** `action_cache_interval` config field, no cache path in `denoise_step`, and no stub. If a 10-step checkpoint ever genuinely must ship without distillation, reopen this decision in review first.

## 6. How to run a pruned config

Both switches live in `SmolVLARECAPConfig` and default to off. Because the config is a registered `PreTrainedConfig` subclass (`smolvla_recap`), it is serialized to the checkpoint's `config.json`, so a pruned variant is produced either by editing that file or by constructing the config in Python. The policy reads the fields at construction — no retraining, one checkpoint serves every ablation.

**Via the exported `config.json`** (add these keys to an existing checkpoint config):

```json
{
  "type": "smolvla_recap",
  "pruned_layers": [5, 9, 12],
  "visual_tokens_keep": 24,
  "token_prune_alpha": 0.5,
  "token_prune_k_key": 4,
  "token_prune_refresh": 1
}
```

**Via Python:**

```python
from lerobot_policy_smolvla_rl import SmolVLARECAPConfig, SmolVLARECAPPolicy

config = SmolVLARECAPConfig(
    # ... existing fields (num_vlm_layers=16, etc.) ...
    pruned_layers=[5, 9, 12],   # skip these paired (VLM, expert) layers; kept layers renumber to 0..N-1
    visual_tokens_keep=24,      # keep 24 of 64 SigLIP tokens per camera
    token_prune_alpha=0.5,
    token_prune_k_key=4,
    token_prune_refresh=1,      # >1 amortizes the scoring pass across chunks
)

policy = SmolVLARECAPPolicy(config)   # check_pruning_constraints runs here; illegal patterns raise ValueError
```

Notes:
- `pruned_layers` is a list of integer indices into the kept layers; `visual_tokens_keep` is an int (K per camera). Leave either as `None`/omitted to disable that component. They are independent — you can enable C1, C2, or both.
- `check_pruning_constraints` runs at policy construction: never prune layer 0 or the final layer, and never leave a contiguous kept-run without a cross-attention (odd) layer.
- Pick `pruned_layers` from `analyze layer-redundancy` and `visual_tokens_keep` from an `analyze ablate` K-sweep, rather than guessing.
- Pruned configs are rejected by the training entrypoints (`train_recap.py`, `train_snapflow.py`) — they are inference-only.

## 7. Training & validation guardrails

- **Training block (implemented).** `train_recap.py` and `train_snapflow.py` raise `ValueError` if `pruned_layers is not None or visual_tokens_keep is not None`, so pruning can never silently corrupt co-training/distillation.
- **Recovery / phase-2 finetune.** If layer pruning causes a measured drop, a short expert-only finetune (VLM frozen) with the target skip configuration enabled restores most of it.

## 8. Deployment / TensorRT compatibility

The TRT export/runtime lives in a **separate repo** (`ros-fhnw-autonomy/.../lerobot_ros/lerobot_ros/trt/`): `exporter.py` (ONNX export + engine build), `engine.py` (`TRTEngineRunner`), `policy.py` (`RECAPTRTPolicy` / `RECAPSnapflowTRTPolicy`), `validate.py` (per-config accuracy check vs. PyTorch). It splits SmolVLA into a **prefix engine** (images + language + state → KV cache) and a **suffix engine** (one denoise/velocity step against the cached KV); the denoise loop and CFG blend run on the host. This maps cleanly onto EfficientVLA:

- **C1 layer pruning — near-free, one hard constraint.** The exporter derives `num_layers` from the prefix's KV-cache output count (`num_layers = num_cache_tensors // 2`), and the suffix wrapper takes it explicitly, so removing layers propagates to both engines with **no TRT edits — just a re-export**. The one requirement is the contiguous dense KV re-index of §3 (survivors renumbered `0..N−1`).
- **C2 token pruning — the one real integration, and it belongs in-graph.** `torch.onnx.export` is called with **no `dynamic_axes`**, so batch (=1), prefix length and KV seq-length are frozen at export — exactly what fixed-K pruning wants. The layer-2 scoring + TopK + Gather selection belongs inside `SmolVLAPrefixWrapper.forward` (TRT supports TopK/Gather natively); the suffix engine then re-exports automatically against the shorter KV length. The `token_prune_refresh > 1` variant instead computes indices host-side and needs image preprocessing split into its own engine — more engines, only worth it if the scorer shows up in latency.
- **Because there are no `dynamic_axes`, every EfficientVLA config variant (each K, each `pruned_layers` set) requires a fresh export + engine rebuild.** Build sweep tooling around that and **add a TRT timing cache to `build_trt_engine`** to cut rebuild time.
- **C3 caching — not built (§5).** Step-dependent control flow is TRT-hostile and strictly dominated by SnapFlow at 1 step.

**Target end state: SnapFlow (1 NFE) + C1 + C2 collapses inference to two static-graph passes (prefix, suffix) — the ideal TRT shape.** Re-run `validate.py` per config (FP16 is on by default; harmless at 1 NFE, but every shipped config must pass its max-abs-error check against the PyTorch reference).

## 9. Tests

Covered in `tests/test_efficient_inference.py` (CPU, mock-policy style):

- **Three-stage selector** (`test_select_diverse_tokens`): on a synthetic 4-token pattern, `select_diverse_tokens` returns the expected key ∪ relevance ∪ diversity set (`[0, 2, 3]`), correctly sized, deduplicated, and spatially sorted.
- **Constraint validation** (`test_layer_redundancy_forward_patching`): `check_pruning_constraints` accepts `None` and rejects pruning layer 0, the final layer, or out-of-range indices.
- **Training guardrail** (`test_training_guardrails`): a pruned/token-keep config is rejected before training.
- **Dense KV re-index** (`test_kv_reindex_correctness`): with `pruned_layers=[2]` on a 4-layer model, `_layer_idx_to_cache_idx == {0:0, 1:1, 3:2}`, and `past_key_values` keys are exactly `{0,1,2}` after both the prefix pass (`fill_kv_cache=True`) and the denoise step (`fill_kv_cache=False`), with prefix KV length preserved.
- **Layer-skip equivalence** (`test_output_equivalence`): forward with `pruned_layers=[2]` matches a hand-built model with layer 2 deleted, to `atol=rtol=1e-5`.
- **KI composition** (`test_ki_composition`): under a pruned config, the prefix forward stays differentiable and gradients flow back to the VLM input (validating that the layer-skip + KI-patch + KV-reindex composition is correct).

## 10. Evaluation & calibration protocol

Offline first, robot last. Ablation axis:

| Configuration | Visual tokens (K) | Pruned layers (n) | FM/diffusion steps |
|---|---|---|---|
| Baseline (unpruned) | 64 | None | 10 |
| SnapFlow base | 64 | None | 1 |
| SnapFlow + C2 | {16, 24, 32, 48} | None | 1 |
| SnapFlow + C1 | 64 | Derived via `analyze layer-redundancy` | 1 |
| SnapFlow + C1 + C2 | Calibration sweep | Calibration sweep | 1 |

Per cell:
1. **Action-space metrics:** action MSE / per-dim MSE vs. ground truth and vs. the unmodified policy, per task (via `analyze ablate` with the efficient-inference config toggled).
2. **Latency/FLOPs:** wall-clock `select_action` on target hardware (rover Jetson if applicable, else dev GPU); FLOPs via `torch.profiler`.
3. **Closed-loop:** LIBERO suites via the existing eval scripts, then `scripts/record_eval.py` on the rover for the final config; report via `analyze eval-results` pivoted on the new config axis.

Success criteria (mirrors paper): combined config reaches ≥1.5× wall-clock on top of SnapFlow with closed-loop success within 2% of baseline; every component individually toggleable and measured.

## 11. Risks

- **KI-trained expert sensitivity to layer pruning** — mitigated by paired pruning, small n, and the finetune escape hatch (§3/§7).
- **Token pruning under distribution shift** (rover scenes ≠ calibration data): the diversity component helps, but validate per task; the per-camera K floor prevents catastrophic single-camera blindness.
- **CFG-selection mismatch** — a silent correctness bug, covered by the CFG-consistent selection path and tests.
- **Interaction with SnapFlow distillation:** distillation was trained on unpruned prefixes. Measure C2 on the distilled checkpoint (expected fine — the prefix is input conditioning, not part of the distilled dynamics), but if MSE jumps, re-distill with pruning enabled (cheap: single GPU).
