
## Exercise 2 — Baseline: Linear SVM
For the baseline in exercise 1.3, a **Linear Support Vector Machine (SVM)** was used.

**Classification Report:**

---

## Exercise 2 — Fine-Tuning
The fine-tuning of the model was performed with the following parameters:

- **Learning rate:** `2e-5`  
- **Weight decay:** `0.01`  
- **Early stopping:** patience = 2  
- **Metric for best model:** `f1_score`  
- **Parameters origin:** taken from a Hugging Face tutorial (see [main README](../README.md))  

**Results after fine-tuning:**

| Epoch | Training Loss | Validation Loss | Accuracy | Precision | Recall | F1 Score |
|-------|---------------|-----------------|----------|-----------|--------|----------|
| 1     | 0.445000      | 0.366510        | 0.843340 | 0.827957  | 0.866792 | 0.846929 |
| 2     | 0.284000      | 0.354573        | 0.841463 | 0.868421  | 0.804878 | 0.835443 |
| 3     | 0.202700      | 0.394936        | 0.845216 | 0.821678  | 0.881801 | 0.850679 |
| 4     | 0.135800      | 0.443272        | 0.852720 | 0.871542  | 0.827392 | 0.848893 |


**Evaluation metrics (final model):**

| Metric                  | Value     |
|-------------------------|-----------|
| Eval loss               | 0.354573  |
| Eval accuracy           | 0.841463  |
| Eval precision          | 0.868421  |
| Eval recall             | 0.804878  |
| Eval F1 score           | 0.835443  |
| Eval runtime (s)        | 1.1857    |
| Samples per second      | 899.05    |
| Steps per second        | 14.338    |
| Epoch                   | 4.0       |



---

## Final Exercise — Named Entity Recognition (NER) on GENETAG
For the last exercise, I chose a custom task: **Named Entity Recognition (NER)**.

- **Dataset:** [GENETAG](https://huggingface.co/datasets/bigbio/genetag)  
- **Model:** [PubMedBERT](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract)  
  - Pretrained exclusively on PubMed's biomedical literature  
  - A classification head was added for the NER task  
- **Goal:** Automatically recognize names of gene products within scientific articles.  
  - This is potentially useful for databases of bioproduct interactions, where one of the most important metrics is the **co-occurrence** in published articles.

---

### Fine-Tuning Details
The fine-tuning parameters were the same as in the previous task:

- Learning rate: `2e-5`  
- Weight decay: `0.01`  
- Early stopping: patience = 2  
- Metric for best model: `f1_score`

---

### Named Entity Highlighting
A small function was implemented to highlight the names of the recognized entities.

**Example (from GENETAG sentences):**

**Example with entities highlighted**

Large <span style="color:red; font-weight:bold;">T antigen</span> was coimmunoprecipitated by antibodies to epitope-tagged <span style="color:red; font-weight:bold;">TBP</span> , endogenous <span style="color:red; font-weight:bold;">TBP</span> , <span style="color:red; font-weight:bold;">hTAF ( II ) 100</span> , <span style="color:red; font-weight:bold;">hTAF ( II ) 130</span> , and <span style="color:red; font-weight:bold;">hTAF ( II ) 250</span> , under conditions where <span style="color:red; font-weight:bold;">holo-TFIID</span> would be precipitated .

We propose a model in which <span style="color:red; font-weight:bold;">Sro7</span> function is involved in the targeting of the <span style="color:red; font-weight:bold;">myosin proteins</span> to their intrinsic pathways .

Glycogen synthesis and catabolism , gluconeogenesis , glycolysis , motility , cell surface properties and adherence are modulated by <span style="color:red; font-weight:bold;">csrA</span> in *Escherichia coli* , while the production of several secreted virulence factors , the <span style="color:red; font-weight:bold;">plant hypersensitive response elicitor HrpN ( Ecc )</span> and , potentially , other secondary metabolites are regulated by <span style="color:red; font-weight:bold;">rsmA</span> in *Erwinia carotovora* .

In one acromegalic patient visual improvement was obtained while the abnormal <span style="color:red; font-weight:bold;">GH</span> secretion remained unaltered .


---

### Final Report

**Results after fine-tuning (NER task):**

| Epoch | Training Loss | Validation Loss | Accuracy | Precision | Recall   | F1 Score |
|-------|---------------|-----------------|----------|-----------|----------|----------|
| 1     | 0.109300      | 0.070608        | 0.974505 | 0.869890  | 0.934348 | 0.900968 |
| 2     | 0.050000      | 0.067243        | 0.975055 | 0.868076  | 0.942222 | 0.903631 |
| 3     | 0.031000      | 0.081520        | 0.976880 | 0.907505  | 0.906085 | 0.906795 |
| 4     | 0.017900      | 0.099117        | 0.976880 | 0.908529  | 0.904834 | 0.906678 |
| 5     | 0.011300      | 0.121359        | 0.976537 | 0.892331  | 0.922251 | 0.907044 |


**Evaluation metrics (final model, NER task):**

| Metric                  | Value     |
|-------------------------|-----------|
| Eval loss               | 0.070608  |
| Eval accuracy           | 0.974505  |
| Eval precision          | 0.869890  |
| Eval recall             | 0.934348  |
| Eval F1 score           | 0.900968  |
| Eval runtime (s)        | 5.5247    |
| Samples per second      | 905.028   |
| Steps per second        | 56.655    |
| Epoch                   | 5.0       |




