# Paper: Critic-Guided Rollout Post-Training of a Compact VLA for Edge Robotics

IEEE conference draft (`main.tex`, IEEEtran class). The PDF is **not** committed;
it is built by CI (`.github/workflows/paper.yml`) on every push touching `paper/`
and published as a workflow artifact and on the project page.

## Building the PDF

```bash
cd paper
latexmk -pdf main.tex
```

Only `main.tex`, `IEEEtran.cls`, and `figs/` are needed; no bibliography files
(references are inline in `thebibliography`).

## Regenerating the figures

All figure inputs are committed, so both scripts run from the repo alone:

```bash
cd paper
python make_figures.py     # figs/fig_critic.{pdf,png}, figs/fig_dose.{pdf,png}
python make_filmstrips.py  # figs/drawer_compare.png, figs/lib10_dualobj_fail.png
```

- `make_figures.py` — critic value trace (input: `data/ep345_V_real.npy`, the
  critic's per-frame value on expert episode 345 with a mid-episode progress
  stall) and the guidance-weight dose-response curves. The sweep numbers are
  inlined in the script; they come from the LIBERO-spatial n=500 evaluations
  (see `docs/paper_experiment_plan.md` for the run matrix; per-run results are
  in the wandb project).
- `make_filmstrips.py` — filmstrip figures sampled from the rollout videos in
  `../analysis_videos/` (10 evenly spaced frames per episode).

The generated figure PDFs/PNGs are committed (see `.gitignore` exceptions) so
the paper builds without a Python environment.

## Provenance of the numbers in the paper

- **Tables I–III (success rates, label composition):** n=500-per-suite LIBERO
  evaluations; experiment matrix and claim mapping in
  `docs/paper_experiment_plan.md`, narrative summary in
  `docs/recap_findings_overview.md`.
- **Jetson latency (Sec. SnapFlow, Appendix):** measured with
  `scripts/bench_jetson.py` on an NVIDIA Jetson Orin Nano (8 GB, JetPack 6,
  PyTorch 2.11, bfloat16), 30 timed chunk generations after 8 warm-up runs.
- **Rollout videos** (`../analysis_videos/`): evaluation episodes of the
  policies compared in the paper; filenames encode
  `<policy>_<suite+task>_<outcome>_<episode>`.
