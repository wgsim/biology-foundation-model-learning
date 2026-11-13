# Experiment Log Template

> **English translation.** AI-assisted translation of the [Korean original](experiment_log_template.md). The Korean original is authoritative and may be newer. Links to pages without an English version open the Korean original.

```markdown
# Experiment [number]: [one-line title]
- Date:
- Notebook/script:
- Environment: (Mac MPS / Colab T4 / cluster), Python/torch versions, seed

## Question
What do you want to find out? (one-sentence hypothesis)

## Setup
- Data (source, size):
- Split rule (decided before looking at the data):
- Model/checkpoint ID (date checked):
- Baseline:
- Changed variable (just one):

## Results
| Setting | Metric | Value | Repeats/standard deviation |
|---|---|---|---|

## Interpretation / hypotheses
- Was it as expected? If not, why?
- Check for possible leakage: homology / duplication / batch / time

## Next experiment
```

<!-- i18n-source: 00_orientation/experiment_log_template.md blob:c899c1cade21d60f1efef31195bdaa36fe5eeb6f -->
