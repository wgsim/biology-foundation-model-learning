# Cheatsheets

> **English translation.** AI-assisted translation of the [Korean original](README.md). The Korean original is authoritative and may be newer. Links to pages without an English version open the Korean original. In the cheatsheets' code blocks only the comments and message strings were translated; the executable code is unchanged.

One page each for the patterns and calculations actually used in the modules. The `python` blocks of each file were **run in order and confirmed** in the repository environment (`.venv`) (the `assert`s inside the blocks must pass). A `bash` block was not run unless marked. They must be re-checked with new models and with other library versions.

| File | Contents | Related modules |
|---|---|---|
| [01_embedding_recipes.md](01_embedding_recipes.en.md) | Mean embedding (excluding padding and special tokens), layer choice, variant-position embedding, masked-marginal score, common mistakes | 02 |
| [02_metrics_and_stats.md](02_metrics_and_stats.en.md) | Spearman, AUROC (per fold), AUPRC baseline, tie handling, paired comparison and group bootstrap, the function for the decision rule | 01–07 |
| [03_splits_and_leakage.md](03_splits_and_leakage.en.md) | Split units by domain, position-block·cluster·scaffold·donor splits, measuring leakage with numbers | 01, 02, 04, 05, 07 |
| [04_compute_and_memory.md](04_compute_and_memory.en.md) | Memory budget formula, measured values (Mac MPS), what is possible/impossible per environment, equipment quirks | 06 |

For terms see the [glossary.md](../glossary.en.md), for benchmarks the [benchmarks.md](../benchmarks.en.md), and for papers the [reading_list.md](../reading_list.en.md).

<!-- i18n-source: resources/cheatsheets/README.md blob:c4ceaa7831f14b4ee29cb41a9da0f00ee034fa9d -->
