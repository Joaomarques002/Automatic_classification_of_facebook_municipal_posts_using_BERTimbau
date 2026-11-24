# Automatic Classification of Facebook Municipal Posts using BERTimbau

## Overview

This repository contains the complete artefact developed for the Master's Dissertation:

**Automatic Classification of Municipal Facebook Posts: A BERTimbau-Based Approach**

The project focuses on the automatic thematic classification of Portuguese municipal Facebook posts using Natural Language Processing (NLP) and Transformer-based language models.

The artefact was developed under the Design Science Research (DSR) methodology and aims to support:

- Research in Political Science, Sociology and Public Administration.
- Large-scale studies of municipal communication.
- Public communication analysis.
- Future expansion of thematic monitoring systems for Portuguese municipalities.

The final system classifies posts into 11 strategic municipal themes using a fine-tuned BERTimbau model.

## Main Contributions

The repository includes:

- Facebook data extraction pipeline.
- Dataset curation and preprocessing tools.
- Label normalization workflow.
- TF-IDF baseline models.
- BERTimbau fine-tuning pipeline.
- Frozen Encoder vs Full Fine-Tuning ablation study.
- Bootstrap-based statistical evaluation.
- Utilities for inference and prediction on unseen posts.

## Repository Structure

```
Automatic_classification_of_facebook_municipal_posts_using_BERTimbau/

├── data/
│   ├── raw/
│   └── processed/
├── docs/
├── results/
├── src/
│   ├── processing/
│   ├── modeling/
│   └── evaluation/
├── README.md
├── requirements.txt
├── .gitignore
└── .env.example
```

## Datasets

### Raw Data

Located in `data/raw/`:

| File | Description |
|---|---|
| `DATA_socialmedia_localgov_pt.csv` | Original municipal posts dataset used in the first phase of the study. |
| `dataset_new.csv` / `dataset_new.json` | Expanded corpus used in the second phase of the research. |
| `label_mapping.csv` | Mapping file used during label normalization. |

### Processed Data

Located in `data/processed/`:

**Study 1 Dataset**

`dataset_labels_normalizado.csv` - Dataset used in the first phase of the dissertation and in the initial BERTimbau experiments.

Label column:

new_sector_norm

**Study 2 Dataset**

`dataset_new_merged_train_ready.csv` - Expanded dataset used in the second phase of the dissertation, ablation studies and final experiments.

Label column:

CategoryName

## Taxonomy

The final version of the classifier uses 11 categories:

1. Cultura
2. Saúde
3. Ambiente, sustentabilidade e serviços públicos de água, saneamento e resíduos urbanos
4. Juventude, educação, ciência e tecnologia
5. Presidente, Câmara e Assembleia
6. Ação social, inclusão, igualdade, terceira idade e voluntariado
7. Desporto
8. Turismo, paisagem, património e gastronomia
9. Obras municipais
10. Economia, desenvolvimento económico e emprego
11. Segurança, proteção civil e prevenção da violência

## Installation

### Create Virtual Environment

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Linux/macOS:
```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Workflow

### 1. Exploratory Data Analysis

Generate descriptive statistics of the extracted corpus:

```bash
python src/evaluation/eda_report.py
```

Output: `results/eda_report.txt`

### 2. Label Normalization

Normalize the original annotation taxonomy and generate the final thematic categories:

```bash
python src/processing/normalize_labels.py
```

Output: `data/processed/dataset_labels_normalizado.csv`

### 3. Expanded Dataset Construction

Merge newly collected posts with human annotations:

```bash
python src/processing/merge_new_with_labels.py
```

Output:
- `dataset_new_merged_train_ready.csv`

### 4. TF-IDF Baselines

Run classical machine learning baselines:

```bash
python src/modeling/run_tfidf_baseline.py
```

Models: LinearSVC, Logistic Regression, Complement Naive Bayes.

Outputs stored in `results/baseline_out/`.

### 5. Train BERTimbau

Train the main classification model:

```bash
python src/modeling/train_bertimbau.py
```

Generated outputs include:
- `metrics.json`
- `classification_report.txt`
- `confusion_matrix.png`
- trained tokenizer
- trained model weights

### 6. Inference

**Single post:**
```bash
python src/modeling/predict_bertimbau.py text="Example post"
```

**CSV batch prediction:**
```bash
python src/modeling/predict_bertimbau.py csv=input.csv out=predictions.csv
```

### 7. Ablation Study

Compare Frozen Encoder vs. Full Fine-Tuning:

```bash
python src/modeling/run_finetuning_ablation.py
```

Outputs: CSV summary, JSON summary, Markdown report.

### 8. Statistical Validation

**Bootstrap confidence intervals:**
```bash
python src/modeling/bootstrap_ci.py
```

**Bootstrap significance analysis:**
```bash
python src/modeling/bootstrap_ablation.py
```

Outputs in `bootstrap_results/`, including confidence intervals and performance comparisons.

## Default Training Configuration

By default, train_bertimbau.py uses:

Dataset:
`data/processed/dataset_labels_normalizado.csv`

Text column:
post_text

Label column:
new_sector_norm

## Final Experiment Reproduction

The final dissertation results were obtained on:

`data/processed/dataset_new_merged_train_ready.csv`

using:

label=CategoryName

Example:

```bash
python src/modeling/train_bertimbau.py file=data/processed/dataset_new_merged_train_ready.csv label=CategoryName
```

## Results Summary

The best performing configuration was obtained using:

- BERTimbau
- Full Fine-Tuning
- Class Weights
- Early Stopping

Final performance:

| Metric | Value |
|---|---|
| Accuracy | 0.8145 |
| Macro F1 | 0.7956 |
| Weighted F1 | 0.8156 |

## Computational Requirements

The complete pipeline can be executed on standard consumer hardware without requiring a dedicated GPU.

Experiments reported in the dissertation were executed on a laptop-class machine with 12 GB RAM.

## Replication Roadmap

For researchers only interested in reproducing the dissertation results:

1. Install dependencies.
2. Use dataset_labels_normalizado.csv or dataset_new_merged_train_ready.csv.
3. Run train_bertimbau.py.
4. Run report_bertimbau.py.
5. Optionally execute:
   - run_finetuning_ablation.py
   - bootstrap_ablation.py
   - bootstrap_ci.py