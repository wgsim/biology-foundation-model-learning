# Biology Foundation Model Learning Roadmap

한국어: [README.md](README.md)

A self-study curriculum on biology foundation models (protein, DNA/genome, single-cell, multimodal) for learners who already have intermediate ML and biology backgrounds. It runs 18 weeks at 8–10 hours per week and is split into 8 modules (00–07) with lecture notes, 19 Jupyter notebooks, exercises and a capstone. The prose is in Korean. Code, model names and paper titles are in English.

- **Theme**: evaluation rigor. Each module has a deliverable and a data-leakage check. The course repeatedly compares foundation models with simple baselines, measures leakage across splits, and uses negative controls.
- **Compute**: the notebooks run on a Mac (CPU/MPS) or on Colab (T4-class GPU). DDP and SLURM templates are included but have not been run on a cluster.
- **Capstone**: a two-week plan with five tracks (A–E; E reproduces a paper), templates for the proposal, report, rubric and checklist, and dry-run records. One dry run recomputes 22 published TraitGym metric values from cached model scores. All of them match within a tolerance fixed before the results were seen. This checks the score-to-metric step, not the models themselves.
- **Status**: written with an AI coding tool (Claude Code) and cross-reviewed by other AI models (OpenAI Codex, Google Gemini). There was no human expert review and no learner pilot, so treat the content as a study log rather than a validated course. Statements that could not be verified are marked `⚠️` or `†`.
- **Start here**: [SETUP.md](SETUP.md) for the environment, then the module folders listed in the schedule of [README.md](README.md). `bash book/build.sh` builds a searchable web book.
- **License**: MIT for code (`LICENSE`), CC BY 4.0 for prose (`LICENSE-docs.md`).
