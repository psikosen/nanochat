# Few-Shot Adaptation Feasibility Plan

## Execution Plan
1. Review SEAL's few-shot workflow and surface the components we must replicate or adapt inside nanochat.
2. Map SEAL capabilities onto the current nanochat training/inference stack to spot integration points and gaps.
3. Quantify model- and training-level requirements (parameters, context budgets, compute) to test feasibility.
4. Propose an incremental implementation roadmap and logging approach that fits existing repo conventions.

## Key Findings from SEAL Few-Shot Pipeline
- SEAL orchestrates an iterative loop of (a) generating self-edits with a base model plus augmentation tools, (b) selecting successful runs, and (c) distilling them into LoRA adapters via behavior cloning before optionally merging them back into the base model.【fdf630†L1-L3】【838f21†L25-L95】
- The reference experiments target ARC few-shot tasks, rely on pretrained instruct-tuned backbones (Llama 3.2 1B), and depend on Hugging Face tooling (Transformers, PEFT, Accelerate/vLLM) to run distributed sampling and LoRA fine-tuning.【767ebc†L1-L82】【838f21†L7-L95】
- LoRA adapters are applied to attention projections (`q_proj`, `v_proj`) and MLP projections (`gate_proj`, `down_proj`, `up_proj`) with ranks up to 128, then merged back into the base checkpoint for deployment.【767ebc†L59-L78】【838f21†L37-L83】

## nanochat Baseline Snapshot
- Default GPT configuration: 12 layers, 6 attention heads, 768 hidden size, untied embeddings/head, 1,024-token context window.【fcf236†L1-L89】
- Total parameter count for the default configuration is ~162.2M parameters, with 38.6M parameters in both the token embedding and output head, and 7.08M parameters per transformer block.【8ed44a†L1-L22】【fdf630†L1-L15】【f71a64†L1-L5】
- Current training/inference scripts assume full model finetuning in PyTorch; there is no built-in support for parameter-efficient adapters, on-the-fly prompting strategies, or LoRA merge/unmerge utilities.【728a49†L1-L6】

## Feasibility Assessment
| Requirement | Status in nanochat | Needed Work |
| --- | --- | --- |
| **LoRA / Adapter Training** | Not implemented. All optimizers operate on full parameter set (DistAdamW/Muon). | Introduce adapter layers or wrap linear modules with LoRA; add optimizer logic to update only adapter params. |
| **Prompted Few-Shot Inference** | Chat CLI/web assume single-turn prompts; no structured few-shot prompt builder. | Build prompt assembly utilities (ARC-style multi-example prompts) and extend inference scripts to accept multi-example contexts. |
| **Self-Edit Data Generation** | No augmentation toolkit. | Port or reimplement minimal augmentation strategies needed for tasks (e.g., rotations, flips for ARC grids). |
| **RL / Selection Loop** | No infrastructure for iterative self-edit scoring. | Implement evaluation harness + policy for selecting successful trajectories; possibly reuse report/eval modules. |
| **External Dependencies** | Repo currently Torch-only; SEAL uses Transformers, PEFT, vLLM. | Vet versions compatible with Python 3.10, add to uv/pyproject only after confirming GPU availability. |

## Math & Resource Estimates
- **LoRA parameter footprint:** Applying rank-16 adapters to `c_q`, `c_k`, `c_v`, `c_proj`, `c_fc`, and `c_proj` (MLP) adds ~2.65M trainable parameters (≈1.64% of base), keeping memory overhead low for on-the-fly updates.【230543†L1-L19】
- **Memory considerations:** Base model weights in bf16 occupy ~324 MB (162.2M params × 2 bytes). LoRA adapters add ≈5.3 MB. Optimizer states for adapters (Adam) would add ~10.6 MB, remaining comfortably within a single 24 GB GPU.
- **Context budget:** With a 1,024-token window, allocating ~120 tokens per ARC demonstration plus ~200 tokens for instructions allows ~6 examples before truncation. Extending `sequence_len` to 2,048 would double headroom and still fit rotary cache precomputation (`rotary_seq_len = sequence_len * 10`).【fcf236†L55-L69】
- **Iteration cost:** Mirroring SEAL's 12-task × 15 self-edits loop yields 180 forward passes per iteration; at ~1,024 tokens each and assuming 30 tokens/sec on a single A100, a full iteration would take ~1.7 GPU-hours before training adapters. Adapter fine-tuning on ~100 success traces (max length 1k tokens) with batch size 5 and 8 epochs totals ≈160k token updates—tractable on a single 24 GB GPU in <30 minutes using bf16.

## Integration Roadmap
1. **Adapter Infrastructure (Week 1):**
   - Implement reusable LoRA modules for nanochat linear layers and expose toggles via `configurator.py`.
   - Extend optimizer setup to freeze base weights when adapters are active.
2. **Prompt & Data Tooling (Week 1-2):**
   - Build prompt templates mirroring SEAL's self-edit instructions with JSON outputs.
   - Add ARC data loaders/augmenters under `tasks/` leveraging existing dataset utilities.
3. **Self-Edit Loop (Week 2-3):**
   - Create a controller script (e.g., `scripts/few_shot_self_edit.py`) that orchestrates prompt generation, model sampling, scoring, and log persistence (structured JSON matching the canonical schema).
   - Integrate scoring hooks into `report.py` for consistent evaluation output.
4. **Adapter Training (Week 3):**
   - Add a lightweight trainer (can reuse `engine.py`) that consumes successful traces, applies adapter training hyperparameters, and saves merged checkpoints.
5. **Evaluation & UI (Week 4):**
   - Extend chat CLI/web to toggle between base and adapter checkpoints.
   - Add regression tests covering prompt assembly, adapter saving/loading, and evaluation metrics.

## Logging & Observability
- Persist self-edit iterations as newline-delimited JSON using the canonical schema (`filename`, `timestamp`, `classname`, `function`, `system_section`, `line_num`, `error`, `db_phase`, `method`, `message`).
- Emit a derived human-readable line alongside each JSON entry following the required format: `Continuous skepticism (Sherlock Protocol)`.
- Extend `report.py` to summarize adapter performance deltas and to surface any hidden dependencies or cascading effects discovered during iteration.

## Risks & Mitigations
- **Dependency bloat:** Introducing Transformers/PEFT increases install time; mitigate by gating imports behind optional extras and documenting version pins.
- **Context overflow:** ARC tasks with verbose rationales may exceed 1,024 tokens; mitigate via template compression and, if necessary, extending `sequence_len` while monitoring rotary cache memory.
- **Training stability:** Adapter training on small data may overfit; mitigate with validation splits, early stopping, and conservative learning rates (5e-5 matches SEAL defaults).【767ebc†L63-L71】
- **Systemic side effects:** Verify that adapter toggles do not break existing speedrun pipeline; isolate via configuration flags and targeted regression tests.

## Next Steps
- Update `task.md` with tracked items for the few-shot initiative.
- Decide whether to prototype on ARC or a text-only benchmark to reduce dependency lift.
- Confirm hardware availability for adapter training experiments (minimum single A100/A6000 class GPU).
