# Finding Evidence for Scientific Claims (SciFact)

**Inter-IIT Tech Meet 15.0 — NLP Selection Bootcamp** · Mihir Gune (24b2443)

Given a scientific claim, find the abstracts in a 5,183-abstract corpus that support or contradict it (**retrieval**),
then decide which (**verification**); if no abstract does, the claim is NOT ENOUGH INFO. Built with open-source
encoders only (≤ 400M parameters, no generative LLMs); all tuning and model selection on the training and validation
claims.

This repository is the extracted submission (`24b2443_nlp_bootcamp.zip`): `code/`, `predictions/`, `results.json`,
`report.md`. This README summarises the system, the results, and **every experiment that was tried**.

---

## Repository contents

```
├── README.md          ← this file: system, results, experiment log, analysis
├── report.md          ← answers to Q1–Q5 (≤ 100 words each)
├── results.json       ← settings, models, validation scores, timings, memory
├── predictions/       ← 7 eval runs (.trec), 2 validation runs for Q1, 3 Task 5 CSVs (all pass check_format.py)
└── code/              ← the scripts that produce every submitted file + code/README.md (15 ordered commands)
```

## Results (validation split)

| Task | System | nDCG@10 | Recall@100 | MRR |
|---|---|---|---|---|
| 1 | BM25, tuned (k1 = 1.2, b = 0.3, stemming, stop words, title) | 0.865 | 0.981 | 0.850 |
| 2 | Dense, general: `BAAI/bge-base-en-v1.5` | 0.898 | 1.000 | 0.888 |
| 2 | Dense, biomedical: `ncbi/MedCPT` query + article encoders | 0.826 | 1.000 | 0.791 |
| 3 | Hybrid: BM25 over abstracts + citing training claims, bge-base, MedCPT (min-max fusion 1.0 / 0.5 / 0.25) | 0.949 | 1.000 | 0.942 |
| 3 | Rerank: hybrid top 30 by MedCPT cross-encoder fine-tuned on training claims | **0.968** | 1.000 | 0.964 |
| 4 | bge-base fine-tuned, random negatives | 0.906 | 0.991 | 0.900 |
| 4 | bge-base fine-tuned, hard negatives (from our BM25 + dense rankings) | 0.920 | 1.000 | 0.914 |

| Task 5 – verification | Precision | Recall | F1 | 5-fold CV F1 |
|---|---|---|---|---|
| Zero-shot NLI (starting point) | 0.603 | 0.613 | 0.608 | 0.593 |
| **Submitted: cross-encoders select evidence, NLI ensemble labels it** | 0.760 | 0.766 | **0.763** | **0.763** |

Validation has 162 claims (106 with evidence), so differences below ≈ 0.01 nDCG@10 or ≈ 0.02 F1 are within noise.
Every adopted change was checked with a paired bootstrap test (retrieval) or 5-fold cross-validation inside validation
(verification); "CV" below always means that. All timings are CPU numbers (4 vCPU, no GPU).

## The final system

```
claim
  │
  ├─ BM25 over (abstract + training claims that cite it) ─┐
  ├─ bge-base dense similarity ───────────────────────────┼─ min-max fusion → hybrid top 100        (t3_hybrid)
  └─ MedCPT dense similarity ─────────────────────────────┘
                                                            │
                     fine-tuned MedCPT cross-encoder re-orders the top 30                           (t3_rerank)
                                                            │
                                         top 3 abstracts per claim
                                                            │
   is it evidence?  → average of zero-shot + fine-tuned cross-encoder score ≥ threshold
   which direction? → average of zero-shot + fine-tuned DeBERTa NLI (max over sentences)            (t5_verification)
   nothing passes   → NOT ENOUGH INFO
```

## Reproducing

```bash
pip install -r code/requirements.txt
# put the provided data/ folder (corpus.jsonl, train/val/eval_claims.jsonl) at the repository root
cd code    # then the 15 commands in code/README.md, in order
```

Pretrained models download from the HuggingFace Hub. The full pipeline takes ≈ 15 hours on a 4-vCPU machine
(fine-tuning dominates), well under an hour on a GPU. The final steps (`make_t3_rerank.py`, `t5_verify_v2.py`)
reproduce the submitted files byte for byte from cached scores.

---

# Experiment log

Everything tried, in the order it was done, with the outcome. ✅ = in the submission, ❌ = tried and rejected,
⏸ = works but deliberately left out.

## 0. Looking at the data first

Two findings shaped the whole design:

- **Gold abstracts recur across splits.** 62 of the 112 validation gold abstracts are also gold for some training claim;
  only 365 of 5,183 abstracts (7%) are ever evidence, 143 of them for two or more claims.
- **Claims come in negated pairs.** SciFact writes a claim from a paper and then a negated version ("X *lacks* Y" /
  "X *shows* Y"). About a third of every split (train 31%, validation 33%, eval 30%) has a near-duplicate training claim;
  on validation the pair shares the abstract with the **label flipped** in 30 of 35 cases.

Other data facts used later:

| Finding | Number |
|---|---|
| Evidence sentences in the last third of the abstract (results / conclusions) | 53% (first third: 11%) |
| Claim content words found in the gold abstract | 70% on average; 13% of pairs share less than half |
| Claims containing a negation word: CONTRADICT vs SUPPORT | 16% vs 6% |
| Claims mixing SUPPORT and CONTRADICT evidence | 0 (a claim's evidence always points one way) |
| Labels (validation): SUPPORT / CONTRADICT / NOT ENOUGH INFO | 41% / 25% / 35% |

## 1. Task 1 – BM25

| Experiment | Result | |
|---|---|---|
| Own sparse-matrix BM25; grid of 896 configurations (k1 × b × stemming × stop words × tokenizer × title) | **0.865** (k1 = 1.2, b = 0.3) | ✅ |
| Default k1 = 0.9, b = 0.4 | 0.863: tuning k1/b is within noise; stemming (+0.024) and the title (+0.008) matter | |
| Metric cross-checked against `pytrec_eval` (identical) and `rank_bm25` (0.861) | — | |

## 2. Retrieval ideas without neural models

| Experiment | Result | |
|---|---|---|
| **Document expansion: append to each abstract the training claims that cite it** | 0.865 → **0.928** (p < 0.001); no effect on claims whose gold abstract is never cited (0.887 → 0.887) | ✅ (inside the hybrid) |
| RM3 pseudo-relevance feedback | 0.869 (+0.003, p = 0.30) | ❌ |
| LSA (256-dim SVD) retrieval / fused with BM25 | 0.673 / ≤ 0.861 | ❌ |
| Best-sentence BM25 fused with document BM25 | −0.011 | ❌ |
| Title boosting | −0.003 to −0.023 | ❌ |
| Learning-to-rank (logistic regression on lexical features, trained on training claims) | 0.867 (+0.002, p = 0.40) | ❌ |
| Lexical verification baseline (no neural model) | F1 0.44 | (floor for Task 5) |

Lesson: lexical signals were saturated; gains had to come from meaning (neural models) or from the training claims.

## 3. Task 2 – Dense retrieval

| Model | nDCG@10 | Index time | Latency / query | Memory | |
|---|---|---|---|---|---|
| `BAAI/bge-base-en-v1.5` (general) | **0.898** | 18.5 min | 152 ms | 433 MB | ✅ |
| `ncbi/MedCPT` query + article encoders (biomedical) | 0.826 | 24 min | 138 ms | 850 MB | ✅ |

The general model beat the biomedical one (MedCPT was trained on short search queries; claims are full sentences).
BM25 wins on rare exact terms (claims 673, 1261: "LDL-cholesterol", "reactive oxygen species"); bge wins on paraphrases
(766, 1311: "medications" ↔ "pharmacotherapy", "lncRNAs" ↔ "LincRNA"). Per claim, BM25 wins 14, bge 16, 76 tie (Q1).

## 4. Task 3 – Hybrid fusion

| Experiment | Result | |
|---|---|---|
| Min-max fusion, expanded BM25 + bge-base + MedCPT (1.0 / 0.5 / 0.25), from a grid over z-score / min-max / RRF and weights | **0.949** (+0.021 over the best single system) | ✅ |
| Same without the expansion (BM25 + bge-base) | 0.915 | |
| RRF (rank fusion) | below min-max: it discards how confident each scorer is | ❌ |
| Adding the Task 4 fine-tuned bi-encoder to the hybrid | +0.003 (p = 0.30) | ❌ |
| Dense claim-to-claim matching (new claim vs training claims citing each abstract) | +0.0025 (p = 0.12); 0.915 → 0.942 without expansion: same signal as the expansion | ❌ |

## 5. Task 3 – Reranking (from harmful to best system)

| Experiment (hybrid top 30) | nDCG@10 | |
|---|---|---|
| `BAAI/bge-reranker-base` (general) | **0.870, hurt**: it prefers closely related studies over the annotated evidence | ❌ |
| `cross-encoder/ms-marco-MiniLM-L-6-v2`, plain / with citing training claims | 0.890 / 0.940 | ❌ |
| `ncbi/MedCPT-Cross-Encoder`, plain / with citing training claims | 0.913 / 0.956 | |
| **MedCPT cross-encoder fine-tuned on training claims** (BCE, BM25 hard negatives, leave-one-out expansion, 2 epochs) | **0.968** (+0.019 over hybrid, p = 0.10; 13 claims better, 7 worse) | ✅ |
| Fusing reranker and first-stage scores | 0.962, but not allowed: the format requires pure reranker order | ❌ |

Cost: 9.2 s/query on CPU for k = 30, ≈ 30× the hybrid. Rerank depth chosen on validation (k = 10 / 20 / 30 → 30).

## 6. Task 4 – Fine-tuning a bi-encoder

Contrastive (InfoNCE) training on the 440 (claim, gold abstract) training pairs, in-batch negatives plus one explicit
negative per pair; validation and training-claim nDCG@10 tracked after every epoch, best validation epoch kept.

| Model | Before | Random negatives | Hard negatives (BM25 ∪ base model, ranks 3–30, gold removed) | |
|---|---|---|---|---|
| bge-small | 0.870 | 0.882 | 0.899 | ❌ (superseded) |
| **bge-base** | 0.877 | **0.906** (epoch 1) | **0.920** (epoch 2) | ✅ |

**Overfitting (Q3):** with random negatives, validation peaked after epoch 1 and fell (0.906 → 0.896 → 0.891 → 0.892)
while training-claim nDCG kept rising (0.960 → 0.972). Hard negatives overfit less (0.920 → 0.917).
bge-base needed batch 8 on a 15 GB machine (batch 16 ran out of memory).

## 7. Task 5 – Verification

| Experiment | F1 | CV F1 | |
|---|---|---|---|
| Lexical features, no neural model | 0.44 | — | ❌ |
| Zero-shot NLI (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`), max over sentences, one threshold | 0.608 | 0.593 | ❌ |
| NLI fine-tuned on training rationales (+ mined neutral sentences) | 0.627 | 0.608 | |
| Gate by retrieval score | 0.612 | 0.571 (overfits) | ❌ |
| **Cross-encoder decides *whether* an abstract is evidence, NLI decides *which label*** (zero-shot / fine-tuned / averaged NLI labels) | — | 0.659 / 0.726 / **0.741** | |
| Gate with the fine-tuned cross-encoder only | 0.759 | 0.737 | ❌ |
| Gate with the average of both cross-encoders (hybrid top 3) | 0.771 | 0.751 | |
| **Same, verifying the t3_rerank top 3** (118 vs 112 of 124 gold abstracts reachable) | **0.763** | **0.763** | ✅ |
| Separate thresholds for claims with / without a training twin; clinical vs molecular topics; label-bias term | — | 0.745 / 0.748 / 0.743 vs 0.751 (overfit) | ❌ |
| Learned "how many abstracts" model trained on validation (nested CV) | — | 0.745–0.747 | ❌ |
| Learned "how many abstracts" model trained on the 647 training claims (unbiased zero-shot features) | held-out 0.745 | 0.757 | ❌ |
| Paired-claim prior (copy a near-duplicate training claim's abstracts with flipped labels) | 0.690 on top of fine-tuned NLI | 0.682 | ⏸ bypasses the retriever and NLI classifier the task asks for |

**The key insight:** NLI is good at *direction* (given the correct abstract it picks the right label 91.5% of the time)
but poor at deciding *whether* an abstract is evidence at all (AUC 0.68). The biomedical cross-encoder does that far
better (AUC 0.96): 89% of gold candidates pass its threshold vs 8% of non-gold ones. Splitting the two decisions cut
false alarms on NOT-ENOUGH-INFO claims from 30 to 10.

## 8. Error analysis (validation, submitted verifier)

| True \ predicted | SUPPORT | CONTRADICT | NOT ENOUGH INFO |
|---|---|---|---|
| SUPPORT (65) | 55 | 3 | 7 |
| CONTRADICT (40) | 4 | 32 | 4 |
| NOT ENOUGH INFO (56) | 7 | 3 | 46 |

40 of 162 claims are not exactly right: 11 miss all evidence, 11 miss a second abstract or add an extra one, 10 are
false alarms on NOT-ENOUGH-INFO claims, 6 have the wrong label, 2 pick the wrong abstract. **32 of 40 errors are about
how many abstracts to return, not the label.** Only 2 are retrieval misses (5%). Significantly harder claims
(Fisher exact test): several gold abstracts (67% wrong, p = 0.002), no near-duplicate training claim (32% vs 9%,
p = 0.002), molecular / cell-biology topics (29% vs 12% for clinical, p = 0.023).

## 9. Robustness checks

| Check | Result |
|---|---|
| Gain on claims the expansion cannot help (gold abstract never cited in training, 45 claims) | BM25 0.887 → rerank 0.945: most of the gain is genuine retrieval improvement |
| Optimism of choosing settings on validation (choose on 4/5, score the held-out 1/5) | hybrid weights +0.007, rerank depth +0.005, verification thresholds 0 |
| Leakage | none: expansion, fine-tuning and hard negatives use training claims only; validation only chooses settings |
| Calibration | predicted NOT ENOUGH INFO on validation 35.2% vs true 34.6% |
| Eval vs validation | eval predicts more NOT ENOUGH INFO (42%): its top abstracts are less often cited by training claims (38% vs 44%), so fewer pass the evidence threshold; near-duplicate rates are similar (30% vs 33%) |
| Reproducibility | final steps regenerate the submitted files byte for byte; all files pass `check_format.py` |

## 10. Engineering issues fixed along the way

- The DeBERTa NLI checkpoint is stored in fp16 and produced NaN when fine-tuned that way → all models force-loaded in
  float32 (zero-shot results unchanged).
- Unsorted batches padded every sentence to full-abstract length → length-sorted batching (identical results, several
  times faster).
- bge-base fine-tuning at batch 16 ran out of memory → batch 8.

## Limitations and next step

- The training-claim expansion and the reranker's use of citing training claims work because of how SciFact was built
  (sibling and negated claims share abstracts). They use only training data, as the rules allow, but would not transfer
  to a corpus without labelled neighbouring claims.
- The remaining errors are about *how many* abstracts to return. A learned selection model lost (0.745 vs 0.763) because
  unbiased features on the training claims had to come from weaker zero-shot models. Next step: cross-fitted
  fine-tuning of the cross-encoder (k models, each holding out part of the training claims) to give unbiased but strong
  features (report Q5).
