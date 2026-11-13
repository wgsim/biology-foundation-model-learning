# Self-Assessment (10 ML Questions · 10 Biology Questions)

> **English translation.** AI-assisted translation of the [Korean original](self_assessment.md). The Korean original is authoritative and may be newer. Links to pages without an English version open the Korean original.

**Method**: Cover the answers and write your own first. Then open the `<details>` and compare. For the scoring criteria and the recommended path, see [README.md](README.en.md#h-59e2821a).

---

## A. ML

**A1.** For a sequence of length L, what are the time complexity of standard self-attention and the memory complexity of its attention map?
<details><summary>Answer</summary>
The time of the attention computation is O(L²·d) and the attention-map memory is O(L²). For a whole layer, the cost of the Q·K·V linear projections, O(L·d²), is added, giving O(L²d + Ld²); for long sequences with L ≫ d, the L² term dominates. FlashAttention-style methods keep the attention exact while reducing memory access, which brings memory use close to linear, but the amount of computation itself is still proportional to L².
</details>

**A2.** What is the difference between BERT-style masked language modeling (MLM) and an autoregressive (causal) language model, and which biological tasks does each fit naturally?
<details><summary>Answer</summary>
MLM predicts masked tokens from bidirectional context, and it suits embedding extraction and masked-marginal scoring of variant effects. An autoregressive model predicts the next token from the preceding tokens only, and it suits sequence generation and scoring the log-likelihood of a whole sequence. (For example, ESM-2 is MLM, while ProGen and the Evo family are autoregressive. See reading_list.)
</details>

**A3.** What is the definition of perplexity, and why must the perplexities of two models with different tokenizers not be compared directly?
<details><summary>Answer</summary>
Perplexity is exp(the mean negative log-likelihood per token). If the token unit differs (single nucleotide vs k-mer vs BPE), "one token" carries a different amount of information, so the values cannot be compared directly. To compare them, you must convert to a per-base or per-residue unit (bits per base, etc.). Even so, the conversion alone does not always make the comparison fair. The sequences, the evaluated region and the probability definition must be the same, and the masked/pseudo-likelihood of an MLM and the normalized sequence likelihood of an autoregressive model are different quantities.
</details>

**A4.** What does LoRA freeze, and what does it train?
<details><summary>Answer</summary>
The weights W learned in pretraining are frozen, and only the product ΔW = BA of the low-rank matrices B (d×r) and A (r×k) is trained (r ≪ d). The number of trainable parameters and the optimizer memory drop sharply.
</details>

**A5.** What do a linear probe and fine-tuning each evaluate or optimize, and why are probes often used when comparing the quality of representations learned in pretraining?
<details><summary>Answer</summary>
A probe freezes the representation and trains only a linear layer on top of it, so it measures whether the information is linearly encoded in the representation. It is advantageous for a fair comparison of representations and is less affected by hyperparameters. Fine-tuning updates all (or some) of the weights to raise the final performance, but it brings a larger risk of overfitting, a higher retraining cost and greater variability when models are compared.
</details>

**A6.** Why does a random train/test split overestimate performance on biological sequence data?
<details><summary>Answer</summary>
Sequences are evolutionarily related (homologous), so they are not independent samples. With a random split, close relatives of a test sequence are in the training set, so the model scores by recognizing "a new member of a known family" rather than generalizing to a new family (a high score is possible from memorization alone). This is an overestimate when the goal is generalization to new families; if the goal is to predict new members of known families, it can be an appropriate evaluation for that goal. You test this directly in module 01 of this course.
</details>

**A7.** What is the difference between Pearson and Spearman correlation, and why is Spearman often used in variant-effect prediction benchmarks?
<details><summary>Answer</summary>
Pearson is based on a linear relationship, and Spearman on ranks (a monotonic relationship). The scale and distribution of experimental measurements (fitness, etc.) differ between assays, so the score scales do not match, and Spearman, which looks only at ranks, is therefore commonly used.
</details>

**A8.** How do you recognize overfitting from training/validation curves, and how do you mitigate it?
<details><summary>Answer</summary>
If the training loss keeps falling but the validation loss starts to rise, it is overfitting. Mitigate it with early stopping, regularization (dropout/weight decay), more data or augmentation, and a smaller model.
</details>

**A9.** What is the intuition behind the InfoNCE loss used in contrastive learning?
<details><summary>Answer</summary>
It is a softmax cross-entropy that makes the similarity of a positive pair (e.g., image–caption) in a batch larger than that of the negative pairs. The temperature τ controls how sharp the distribution is. (It underlies models such as CLIP and CONCH.)
</details>

**A10.** What is the difference between bf16 and fp16?
<details><summary>Answer</summary>
bf16 has the same exponent range as fp32, so its representable range is wide and loss scaling is generally unnecessary, but its mantissa precision is lower. fp16 has higher mantissa precision but a narrow range, so it needs loss scaling to prevent overflow and underflow.
</details>

---

## B. Biology

**B1.** How many codons are there, and how many amino acids? What is this relationship called?
<details><summary>Answer</summary>
64 codons (including 3 stop codons) specify 20 amino acids, and this is called the degeneracy of the genetic code. Because synonymous codons that specify the same amino acid exist, synonymous variants are possible.
</details>

**B2.** What is the reverse complement of 5'-ATGC-3'? Why does it matter in DNA model design?
<details><summary>Answer</summary>
It is 5'-GCAT-3'. The two sequences are alternative representations of the same double-stranded molecule, read from its two strands. For that reason, some models build reverse-complement data augmentation or RC equivariance into their architecture (e.g., Caduceus in reading_list 03). However, for **tasks where strand direction matters**, such as transcription direction or strand-specific tasks, the two sequences must not be treated as identical.
</details>

**B3.** Why is the coevolution signal in a multiple sequence alignment (MSA) used to predict residue contacts in proteins?
<details><summary>Answer</summary>
When one of two residues in spatial contact changes, the other tends to undergo a compensating mutation, which creates a statistical dependence between the two positions in an MSA. Contacts are estimated by reading this signal. (Rao 2021: the attention maps of language models also learn contacts. It is in reading_list 02.)
</details>

**B4.** What does deep mutational scanning (DMS) measure?
<details><summary>Answer</summary>
It is an experiment that measures, at large scale, a functional score (fitness) for each of many variants of a protein (usually every possible single amino-acid substitution), using a functional selection or screening assay. It provides the ground-truth data for variant-effect benchmarks such as ProteinGym.
</details>

**B5.** Distinguish missense, synonymous and nonsense variants.
<details><summary>Answer</summary>
A missense variant changes the amino acid. A synonymous variant changes the codon, but the amino acid stays the same. A nonsense variant creates a premature stop codon, which shortens the protein.
</details>

**B6.** What are the main properties of single-cell RNA-seq count data, and what makes the analysis difficult?
<details><summary>Answer</summary>
The main properties are sparsity (many 0s, including dropout), differences in library size between cells, overdispersed counts, and batch effects. For this reason, normalization and log transformation, or probabilistic models based on the negative binomial or ZINB distribution (e.g., scVI in reading_list 04), are used.
</details>

**B7.** What is the difference between a promoter and an enhancer? Explain it in connection with why sequence-to-expression models need long contexts.
<details><summary>Answer</summary>
A promoter is a regulatory region near the transcription start site that helps initiate basal transcription. An enhancer is a regulatory element that depends relatively little on distance and orientation and can regulate gene expression from far away. Capturing the influence of distal enhancers requires a context of tens to hundreds of kb (this is the motivation of the Enformer and Borzoi families; see reading_list 03).
</details>

**B8.** What does AlphaFold's pLDDT mean?
<details><summary>Answer</summary>
It is a per-residue confidence score for the prediction (0–100). A low value may mean that the prediction is uncertain or that the region is intrinsically disordered. It is a different metric from the accuracy of the whole structure (e.g., TM-score).
</details>

**B9.** Distinguish homolog, ortholog and paralog.
<details><summary>Answer</summary>
Homologs are sequences that share a common ancestor (the umbrella term). Orthologs are homologous genes that diverged by speciation. Paralogs are homologous genes that diverged by gene duplication.
</details>

**B10.** Why is a significant GWAS variant not necessarily the causal variant? What role are variant-effect prediction models expected to play here?
<details><summary>Answer</summary>
Because of linkage disequilibrium (LD), many variants around the causal variant appear significant together. Models are expected to narrow down the candidates by predicting the functional impact of variants (as an aid to fine-mapping), but a model score does not mean causality.
</details>

---

## Scoring notes
| Area | Number correct | Notes (weak topics) |
|---|---|---|
| ML (A) | / 10 | |
| Biology (B) | / 10 | |

<!-- i18n-source: 00_orientation/self_assessment.md blob:23ac504b253cdf6d941605b8bcf5e421581fadff -->
