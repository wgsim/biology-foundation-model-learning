# Biology Foundation Model Learning Roadmap

한국어: [README.md](README.md)

A self-study curriculum on foundation models for biology (protein, DNA/genome, single-cell and multimodal), for learners with an intermediate background in both machine learning and biology. It takes 18 weeks at 8–10 hours per week. The course has 8 modules (00–07) with lecture notes, 19 Jupyter notebooks and exercises, and it ends with a capstone. The text is in Korean; code, model names and paper titles are in English.

- **Focus**: evaluation rigor. Each module includes a deliverable and a data-leakage check. The course repeatedly compares foundation models with simple baselines, measures leakage across splits, and uses negative controls.
- **Hardware**: the notebooks run on a Mac (CPU or MPS) or on Colab (T4-class GPU). DDP and SLURM templates are included but have not been run on a cluster.
- **Capstone**: a two-week plan with five tracks (A–E; track E reproduces a paper), templates for the proposal, report, rubric and checklist, and records of dry runs (the author's own rehearsals of the workflow, not learner pilots). One dry run recomputed 22 published TraitGym metric values from the benchmark's released model scores, and all 22 matched within a tolerance specified before the results were seen (not a formal preregistration). This checks only how the metrics are computed from the scores, not the models themselves.
- **Status**: written with an AI coding tool (Claude Code) and cross-reviewed by other AI models (OpenAI Codex and Google Gemini). No human expert has reviewed it, and no learner has piloted it, so treat it as a study log rather than a validated course. Statements that could not be verified are marked `⚠️` or `†`.
- **Start here**: [SETUP.md](SETUP.md) for the environment, then the 18-week schedule in [README.md](README.md), which links to each module. `bash book/build.sh` builds a searchable web book.
- **License**: MIT for both code and text (`LICENSE`). Models and datasets that the notebooks download keep their own licenses, and values read from the tables and figures of papers are not stored in this repository.
