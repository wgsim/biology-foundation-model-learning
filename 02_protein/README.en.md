# 02. Protein — Protein Language Models, Variant Effects, Structure, Design

> **English translation.** AI-assisted translation of the [Korean original](README.md). The Korean original is authoritative and may be newer. Links to pages without an English version open the Korean original.

**Duration**: Weeks 3–6 (4 weeks, about 32–40 hours) · **Compute tier**: 🟢 3 local Mac notebooks + 🟡 1 Colab notebook
**Prerequisites**: The tokenization, objectives and split design of [module 01](../01_foundations/) (especially section 5, the pitfalls of evaluation).

> Notation: † marks a statement that must be checked directly in the paper text or the model card. The cited papers are in section 02 of the [reading_list](../resources/reading_list.en.md). The numbers and model information in this module are as of 2026-11. New models appear quickly, so check the model card before you use one.

---

## Learning goals
1. Compare how protein language models (PLMs) are trained and their lineage (ESM-1b → ESM-2 → ESM C, the ProGen family, structure-aware models).
2. Build a **linear probe** on PLM embeddings and evaluate it fairly with **baselines and a split**.
3. Compute **zero-shot variant effect scores** (masked marginal) directly and evaluate them the ProteinGym way (Spearman).
4. Interpret the inputs and outputs of structure prediction models (AlphaFold 2/3, ESMFold) and **confidence metrics such as pLDDT**.
5. Explain the **possibilities and limits** of studies that look inside PLMs with attention and SAEs.
6. Distinguish the roles of design models (inverse folding, diffusion, generative LMs), and explain the difference between in silico evaluation and experimental validation.

## Four-week schedule
| Week | Topic | Notebook | Reading (mainly ★) |
|---|---|---|---|
| 3 | PLM basics: representations, probe | [01](notebooks/01_esm2_embeddings_probe.ipynb) | ESM-1b (Rives 2021), ESM-2 (Lin 2023) |
| 4 | Zero-shot variant effects, benchmarks | [02](notebooks/02_zero_shot_variant_effect.ipynb) | ESM-1v, ProteinGym, EVE, AlphaMissense |
| 5 | Structure prediction and interpretation | [03](notebooks/03_attention_contact_map.ipynb), [04](notebooks/04_esmfold_plddt_colab.ipynb) | AlphaFold 2·3, Rao 2021, InterPLM |
| 6 | Design + module deliverable | (reading and write-up) | ProteinMPNN, RFdiffusion, ESM-IF1, ESM3 |

**Module deliverable**: A table comparing, on one ProteinGym assay, simple baselines (one-hot, amino-acid composition) and an ESM-2 embedding probe under two splits, **a random split and a position-level split**, and its interpretation ([exercises.md](exercises.en.md) exercise 1).

---

## 1. Data and pretraining

- Protein sequence DBs have **large phylogenetic and species sampling bias and many duplicates**. The pretraining corpus is usually a set of representative sequences clustered by sequence identity (e.g., the UniRef family). The data composition that `UR50D` in the ESM-2 checkpoint name refers to should be checked in the model card and the paper†.
- The ESM-1b paper reported that in a model trained on about 250 million sequences (86 billion amino acids), 2nd- and 3rd-level (secondary and tertiary) structure information is held in the representations and can be read out with a linear projection (per the abstract).
- **Evaluation sequences and pretraining sequences can overlap.** This is the problem of the "Common" row of the table in module 01 5.2. Interpret the ProteinGym results with the point in mind that the DMS target proteins may be in the pretraining corpus.

## 2. Model lineage and roles

| Family | Representative models | Input/objective | Use | Notes |
|---|---|---|---|---|
| MLM PLM | ESM-1b, ESM-2 (8M–15B), ProtTrans | Single sequence, MLM | Embeddings, variant scores | ESM-2: reports that growing the model up to 15B makes atomic-level structure information appear in the representations (Lin 2023 abstract) |
| MLM PLM (new) | ESM C | Single sequence, MLM | Embeddings | Released 2024-12, as a technical blog and model card rather than a paper. Check the weight license in the model card |
| Structure-aware PLM | SaProt | Residues + Foldseek structure tokens | Embeddings, variant scores | Needs a structure |
| Autoregressive PLM | ProGen, ProGen2 | Single sequence, next token | Generation, likelihood | Control-tag-based generation (ProGen) |
| Multi-track generation | ESM3 | Sequence, structure, function, etc.† | Conditional generation | Check the paper for details† |
| Structure prediction | AlphaFold 2, ESMFold, AlphaFold 3, ESMFold2, Boltz-1/2, Chai-1 | Sequence (+MSA) (+ligands, nucleic acids) | 3D coordinates + confidence | AF2 is MSA-based, ESMFold is single sequence + PLM, AF3 is diffusion-based complex prediction |
| Design (sequence) | ProteinMPNN, ESM-IF1 | Backbone structure → sequence (inverse folding) | Sequence design | ESM-IF1 expanded its training data with AF2-predicted structures |
| Design (backbone) | RFdiffusion | Diffusion model | Backbone/function design | A pipeline of design, ProteinMPNN and structure re-prediction is common |
| Variant effect | ESM-1v, EVE, AlphaMissense | Sequence distribution/evolutionary information | Pathogenicity/fitness prediction | EVE is an MSA-based generative model, trained without labels |
| Benchmarks | TAPE, FLIP, PEER, ProteinGym | — | Evaluation | ProteinGym: per the paper abstract, 250 or more DMS assays + clinical data. The v1.3 substitution file downloaded in this module contains 217 assays |

**ESMFold2 (2026)** is a structure prediction model released by Biohub (the ESM team). According to the model card, it handles all biomolecules including small molecules, DNA and RNA, the MSA input is optional, and it outputs pLDDT, pAE, pTM and iPTM. The weights are tens of GB, which is beyond the scope of local hands-on work. The supporting paper is a bioRxiv preprint (reading_list 02), and the performance claims are what the authors reported.

---

## 3. Evaluating representations: embeddings and linear probe

### 3.1 Extracting embeddings
- When a sequence is passed through a residue-level tokenizer, special tokens such as `<cls>`/`<eos>` are added, and the model outputs a hidden state at each layer.
- **Sequence-level embedding**: the average over residue positions (mean pooling). In the average, you must **exclude padding and the special tokens `<cls>`/`<eos>`**. Note that the attention mask filters out **only padding**, and `<cls>`/`<eos>` have mask=1. So the special-token positions must be set to 0 separately (see `embed()` in notebook 01). It matters especially when sequences of different lengths are mixed.
- **Which layer to use**: the last layer is not always the best. For some tasks an intermediate layer is better, so compare by layer†. `embed(..., layer=)` in notebook 01 takes a `hidden_states` index. Index 0 is the output of the embedding layer, and the last index is the same as `last_hidden_state`; verify this in the notebook. The values of intermediate layers are values before the final LayerNorm, so their scale may differ†.

### 3.2 Linear probe design principles
1. Keep the embeddings **frozen** and train only a linear model (Ridge/logistic). In this way you can see the quality of the representation itself.
2. **Build the baselines first.** Use one-hot (position × residue) and amino-acid composition, and if possible add a BLOSUM-based score.
3. **Choose the split to fit the purpose.** If you do a random split on DMS variant data, other substitutions at the same position go into train. So it is an optimistic evaluation for claiming generalization to **new positions**. Conversely, if the purpose is "to fill in the **unmeasured substitutions** of positions that have already been measured", a random split can be the evaluation that fits that purpose. Decide the purpose first and use the split that fits it (for generalization to new positions, a **position-level/contiguous-block split**).
4. Choose the regularization strength **inside train**, and look at the test set only once.

> **Hands-on [01_esm2_embeddings_probe](notebooks/01_esm2_embeddings_probe.ipynb)**

---

## 4. Zero-shot variant effect prediction

Apply the score definitions of module 01 3.1 to PLMs.

| Method | Computation | Number of forward passes | Notes |
|---|---|---|---|
| masked marginal | Mask position $i$ → $\log p(\text{mut})-\log p(\text{wt})$ | Sequence length L (1 per position) | The standard method for single substitutions |
| wild-type marginal | 1 pass of the wt sequence without masking → the same formula | 1 | The cheapest, but the wt residue is visible in the input |
| pseudo-log-likelihood ratio | PLL of the variant and wt sequences each | Number of variants × L | Expensive, extendable to multiple variants |

Meier et al. 2021 compare several scoring methods. Check the details in the paper†.

### 4.1 Evaluation: the ProteinGym way
- For each assay, compute the **Spearman correlation** (score vs DMS score) and average. The average in notebook 02 is only a **simple average** over the 7 selected assays. It is not the same as the official ProteinGym aggregation procedure, which goes through per-type aggregation, so check the official documentation when comparing with official scores†. Even the same model performs very differently depending on the assay type (stability, activity, binding, expression, fitness, etc.). So **do not judge by one average; look by type**.
- DMS measurements have **noise**, so the upper bound of the correlation is not 1.
- Distinguish single substitutions from multiple substitutions. The masked marginal in the table above is for single substitutions. If you extend it to multiple substitutions by summing scores, you ignore epistasis.
- Clinical variant pathogenicity prediction (AlphaMissense, EVE) has a **different goal** from DMS fitness prediction. Do not compare them with the same metric.

> **Hands-on [02_zero_shot_variant_effect](notebooks/02_zero_shot_variant_effect.ipynb)**: Model size (8M–650M) × several assays.

---

## 5. Structure prediction and confidence

### 5.1 Model comparison (concepts)
| Model | Input | Key idea | Confidence outputs |
|---|---|---|---|
| AlphaFold 2 | Sequence + MSA | End-to-end prediction combining the MSA and a structure module | pLDDT, PAE, pTM |
| ESMFold | Single sequence | A folding module on top of ESM-2 representations. Reported to be much faster than AF2 and to have similar accuracy on sequences the PLM understands well (low perplexity) (per the abstract in the HF documentation) | pLDDT, PAE, pTM (checked from the output keys of the HF implementation) |
| AlphaFold 3 | Sequences, ligands, nucleic acids, etc. | Complex prediction by diffusion-based structure generation | pLDDT, PAE, ipTM† |
| ESMFold2 | Sequence (+MSA optional) | See the model card | pLDDT, pAE, pTM, iPTM |

### 5.2 How to read pLDDT
- pLDDT is a per-residue confidence of the prediction. Depending on the model it uses a 0–1 or a 0–100 scale, so **check the scale**. Regions with low values are intrinsically disordered or uncertain in the prediction.
- A high pLDDT is **not a guarantee that the structure is right**, and it is not directly the same as stability or activity either. The approach of using the change in pLDDT for variant-effect prediction has limits†. You verify this directly in exercise 5.
- PAE and pTM indicate the relative arrangement between domains and the confidence of the overall fold. The domain arrangement cannot be known from pLDDT alone.

> **Hands-on [04_esmfold_plddt_colab](notebooks/04_esmfold_plddt_colab.ipynb)** 🟡: The weights are large, so a Colab GPU is recommended. It was confirmed by running from start to finish on a Colab T4 (2026-11-06, a single environment). Watch the memory. If the weights are first loaded into CPU RAM, the kernel can die on a 12 GB runtime, so the notebook puts the weights straight onto the GPU. In that case it uses about 14 GB of GPU memory at the instantaneous peak. Details are at the top of the notebook.

---

## 6. Looking inside: attention and SAE

- **Attention → contact** (Rao et al. 2021): The attention maps of language models learn residue contacts from the unsupervised MLM objective alone. According to the abstract, large-capacity models outperformed the unsupervised contact-prediction pipeline. Rather than using the raw attention of a single head as it is, **corrections such as symmetrization and APC** are commonly applied, and a supervised-trained probe is often used for the final contact prediction†.
- **Sparse autoencoder (InterPLM)**: Training an SAE on ESM-2 embeddings gives thousands of interpretable features per layer, and some correspond to binding sites, structural motifs and so on. According to the abstract, concept alignment is better than with individual neurons, and a larger PLM (650M) holds more concepts than 8M. **The causal role of a feature must be verified separately.**
- Interpretation studies share common cautions. Correlation is not causation, post-hoc labeling is biased, and results that work well in synthetic settings have limits in generalization (see module 01 notebook 02).

> **Hands-on [03_attention_contact_map](notebooks/03_attention_contact_map.ipynb)**: Comparison with small protein structures (PDB).

---

## 7. Distinguishing the roles of design models

| Question | Model family | Input → output |
|---|---|---|
| What structure does this sequence have? | Structure prediction | Sequence → coordinates |
| What sequence fits this backbone? | Inverse folding (ProteinMPNN, ESM-IF1) | Backbone → sequence |
| How do I make a new backbone/structure? | Diffusion model (RFdiffusion) | Condition → backbone |
| How do I generate the sequence itself? | Generative LM (ProGen, ESM3) | Condition/prompt → sequence |

As **in silico evaluation**, self-consistency — refolding the designed sequence with a structure predictor and comparing it with the target structure — and the prediction confidence (pLDDT) are commonly used. However, these are only **proxy metrics for experimental success** such as expression, folding and function, and the predictor and the designer may share the same biases. Biosafety and data-license issues are covered in module 06.

---

## 8. Comprehension check

**Q1.** Why must `<cls>`, `<eos>` and padding tokens be excluded when averaging ESM-2 sequence embeddings?
<details><summary>Answer</summary>
Special tokens and padding are not sequence information. When sequences of different lengths are batched together, the amount of padding differs from sequence to sequence, so the average is distorted depending on the length and the batch composition. The attention mask sets only the padding to 0 and keeps `<cls>`/`<eos>` at 1, so these two positions must be excluded from the average separately for only the real residue positions to be averaged. (Check the tokenizer's special-token setup in the model card†)
</details>

**Q2.** Why is a variant-level random split optimistic on DMS data, and what split evaluates generalization to new positions?
<details><summary>Answer</summary>
If other substitution variants at the same position are in train, the score comes out high just from the model memorizing the per-position mean effect (neighboring-variant leakage). Split by position or by contiguous segment so that the test positions are not in train at all. Choose the split according to whether the goal is "predicting new variants" or "predicting new positions/new proteins".
</details>

**Q3.** What is the computational cost and the difference between masked marginal and wild-type marginal?
<details><summary>Answer</summary>
The masked marginal does one forward pass per position (L passes), with that position masked, to obtain the distribution at that position. The wild-type marginal obtains the distributions of all positions with 1 forward pass without masking, so it is much cheaper. However, the wt residue to be predicted is exposed in the input, so the distribution differs. Check the performance difference with a benchmark.
</details>

**Q4.** Why might two models need to be used differently even when their mean Spearman on ProteinGym is the same?
<details><summary>Answer</summary>
Performance can differ by assay type (stability, activity, binding, expression, etc.), and even with the same mean the strengths differ. Look separately at the performance on the type that matches your purpose, the variance, and the ability to handle multiple substitutions.
</details>

**Q5.** In a region with a high pLDDT, can you tell from pLDDT whether one variant greatly reduces stability?
<details><summary>Answer</summary>
You should assume you cannot. pLDDT is the local confidence of the predicted structure and does not directly measure the effect of a variant on the folding free energy. To know the variant effect, you need a separate method such as a zero-shot score or a stability predictor, together with experimental validation. In exercise 5 you directly check the correlation between the change in pLDDT and the DMS.
</details>

**Q6.** Why is ESMFold faster than AlphaFold 2, and what should you watch for regarding accuracy?
<details><summary>Answer</summary>
It is much faster because it uses the PLM representation of a single sequence as input, without MSA search or processing. Its accuracy has been reported to be similar to AF2 on sequences the PLM understands well (low perplexity). However, there can be a gap where the PLM does not understand well, such as sequences that benefit from a deep MSA. Check the perplexity/confidence of the target sequence before use.
</details>

**Q7.** Distinguish inverse folding, structure prediction and backbone generation (diffusion) by input and output.
<details><summary>Answer</summary>
Structure prediction: sequence → coordinates. Inverse folding: backbone → sequence. Backbone generation: condition (motif, binding target, etc.) → new backbone coordinates. A design pipeline is often chained in the order backbone generation → inverse folding → structure re-prediction.
</details>

**Q8.** Why does the success of self-consistency (refolding) of a designed sequence not guarantee experimental success?
<details><summary>Answer</summary>
Structure predictors also have biases and errors that come from their training data, and the designer and the evaluator may share similar assumptions. Expression level, solubility, the actual folding pathway and functional activity are outside the range of prediction. Therefore wet-lab validation is needed.
</details>

**Q9.** Why do you symmetrize and apply the APC correction when reading contacts from attention maps?
<details><summary>Answer</summary>
A contact is a symmetric relation between a pair of residues, whereas attention has a direction, from the query to the key. There is also a background bias in which particular positions are attended to a lot in general. Corrections such as symmetrization and APC reduce this positional bias and bring out the pair signal† (Rao 2021). In the "effect of the corrections" table of notebook 03, the mean precision over 3 proteins rose in the order raw → symmetrization → +APC (e.g., 150M 0.03 → 0.23 → 0.36). On the other hand, the variant that sets the diagonal (self-attention) to 0 made no consistent difference.
</details>

**Q10.** What should you be careful about when interpreting the result of an InterPLM-style SAE study that "a feature corresponds to a binding site"?
<details><summary>Answer</summary>
The correspondence is correlational evidence. Whether manipulating the feature changes the model output, that is, whether there is a causal relationship, must be confirmed with a separate experiment. The labeling method of automatic interpretation, the incompleteness of the annotation DB and the choice of threshold determine the result. Check in the paper body how far the steering and annotation-augmentation experiments that the paper presents support it.
</details>

---

## 9. Reading (reading_list section 02)
| Importance | Item | Week |
|---|---|---|
| ★ | Rives 2021 (ESM-1b), Lin 2023 (ESM-2/ESMFold) | 3 |
| ★ | Meier 2021 (ESM-1v), Notin 2023 (ProteinGym) | 4 |
| ★ (variant track) | Cheng 2023 (AlphaMissense) / ◆ Frazer 2021 (EVE) | 4 |
| ★ | Jumper 2021 (AF2), Abramson 2024 (AF3) + Addendum | 5 |
| ◆ | Rao 2021 (attention→contact), Simon & Zou 2025 (InterPLM) | 5 |
| ★ (design track) | Dauparas 2022 (ProteinMPNN) / ◆ Watson 2023 (RFdiffusion), Hsu 2022 (ESM-IF1) | 6 |
| ◆ | Hayes 2025 (ESM3), Madani 2023 (ProGen), SaProt | 6 |
| ◆ [벤치마크] (benchmark) | TAPE, FLIP, PEER | 3–4 |
| ○ [최신] (latest) | ESM C model card, ESMFold2 preprint | 5 |

## 10. Notebooks
| Notebook | Tier | Content | Download |
|---|---|---|---|
| [01_esm2_embeddings_probe](notebooks/01_esm2_embeddings_probe.ipynb) | 🟢 | ESM-2 embeddings + Ridge probe, baselines, random vs position split | ESM-2 8M·35M, ProteinGym |
| [02_zero_shot_variant_effect](notebooks/02_zero_shot_variant_effect.ipynb) | 🟢 | masked marginal score, Spearman by model size, BLOSUM baseline | ESM-2 8M–650M, ProteinGym |
| [03_attention_contact_map](notebooks/03_attention_contact_map.ipynb) | 🟢 | attention → contact map, APC, precision by model size | ESM-2, PDB files |
| [04_esmfold_plddt_colab](notebooks/04_esmfold_plddt_colab.ipynb) | 🟡 | Fold with ESMFold and interpret pLDDT | ESMFold v1 (about 8.4 GB) |

The notebooks download the data and models. The save location is shown at the top of each notebook. Exercises: [exercises.md](exercises.en.md)

**Reference results.** These are values obtained on a Mac M1 Pro 32GB, MPS, Python 3.12, torch 2.15.0 and transformers 5.22.1, and they may differ with versions and devices.
| Notebook | What to check | Expected value (approximate) | Time |
|---|---|---|---|
| 01 | UBC9 test Spearman — random: B1 / B2 (+position) / ESM-2 35M variant-position embedding | 0.42 / 0.70 / 0.63 | about 1 minute |
| 01 | UBC9 test Spearman — position-block: B1 / B2 / ESM-2 35M variant-position embedding | 0.28 / 0.30 / 0.15 (the ESM features fall short of the baselines) | |
| 02 | Mean Spearman over 7 assays: BLOSUM62 / 8M / 35M / 150M / 650M | 0.24 / 0.21 / 0.28 / 0.42 / 0.45 | about 2 minutes |
| 02 | wt marginal vs masked marginal at 150M: mean Spearman / forward time per assay | 0.414 vs 0.420 / about 0.05 s vs about 3.6 s (about 70 times) | |
| 03 | contact head precision@L/5 (mean of 3 proteins): 8M / 35M / 150M / 650M | 0.30 / 0.55 / 0.83 / 0.89 | about 25 seconds |
| 03 | Unsupervised attention corrections (mean of 3 proteins, 150M): raw / symmetrization / +APC | 0.03 / 0.23 / 0.36 | |
| 03 | contact head, at sequence distance ≥ 24 (8M→650M) | 0.25 / 0.55 / 0.66 / 0.82 | |
| 04 | (Colab T4, Python 3.13, torch 2.14.1, transformers 5.22.1; the numbers were the same in 3 runs, including 2 after the loading method was changed) Mean pLDDT — ubiquitin / GB1 / shuffled sequence / `(GS)n` | 90.5 / 89.0 / 42.8 / 27.3 | about 1.5 minutes (weights cached) |
| 04 | CA RMSD to the experimental structure — ubiquitin (1UBQ) / GB1 (1PGA) | 0.84 Å / 0.42 Å | |

## Next
Move on to [03_dna_genome](../03_dna_genome/).

<!-- i18n-source: 02_protein/README.md blob:0b406191f32f9b73fcb9df584008bb54af30655a -->
