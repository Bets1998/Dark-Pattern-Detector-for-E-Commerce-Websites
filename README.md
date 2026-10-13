# Dark Pattern Detector for E-Commerce Websites

An NLP project that screens short e-commerce messages for **potential dark-pattern language**, comparing classical machine learning, a bidirectional LSTM and fine-tuned DistilBERT.

## Project Overview

E-commerce interfaces may use urgency, scarcity, social proof or guilt-based wording to influence purchasing decisions.

This project trains binary classifiers to distinguish between:

- **0 — Not Dark Pattern**
- **1 — Potential Dark Pattern**

Text alone cannot establish whether a claim is false or intentionally deceptive. Predictions are screening signals that require contextual review.

## Dataset

The notebook combines two CSV files:

| Source file | Original rows |
|---|---:|
| `patterns.csv` | 416 |
| `pattern_classifications.csv` | 2,700 |
| **Combined** | **3,116** |

After removing missing or invalid entries, duplicate text, formatting variants and conflicting normalized labels, **1,954 examples** remain.

The stratified partitions contain:

| Partition | Examples |
|---|---:|
| Training | 1,367 |
| Validation | 293 |
| Test | 294 |

The test set contains **225 non-dark** and **69 dark-pattern** examples.

## Preprocessing

- Validate binary labels and remove empty text.
- Normalize HTML, URLs and whitespace for classical and sequence models.
- Preserve negation, digits and punctuation that may carry useful signals.
- Create normalized keys to detect formatting duplicates before splitting.
- Remove normalized groups with conflicting labels.
- Verify that duplicate keys do not cross dataset partitions.
- Fit TF-IDF, the sequence tokenizer and Word2Vec using training data only.

DistilBERT uses the original text with its pretrained tokenizer and a maximum input length of **64 tokens**.

## Models Compared

### Classical Machine Learning

TF-IDF features are evaluated with:

- Logistic Regression
- Calibrated Linear SVM
- Multinomial Naive Bayes

Five-fold stratified cross-validation selects the classical model family. GridSearchCV tunes its hyperparameters, and validation predictions determine its decision threshold.

**Logistic Regression** is the selected classical model.

### Word2Vec and BiLSTM

Word2Vec embeddings trained on the training partition initialize a bidirectional LSTM. Training uses class weights, early stopping and learning-rate reduction.

### Fine-Tuned DistilBERT

The project fine-tunes `distilbert-base-uncased` using weighted cross-entropy loss. Validation F1 determines the retained checkpoint, and validation predictions determine the operating threshold.

## Model Selection

The deployment decision considers validation F1 and a preference for a lighter classical model when performance is close.

DistilBERT is selected because its validation F1 improvement exceeds the notebook’s **0.03** cutoff.

Its validation-selected decision threshold is approximately **0.525**. Model selection and thresholds are locked before final test reporting.

## Results

The following results are recorded in the notebook’s held-out test outputs:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Average Precision |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.8095 | 0.5699 | 0.7681 | 0.6543 | 0.8808 | 0.7206 |
| Word2Vec + BiLSTM | 0.7789 | 0.5189 | 0.7971 | 0.6286 | 0.8634 | 0.6724 |
| Fine-tuned DistilBERT | **0.8810** | **0.7833** | 0.6812 | **0.7287** | **0.9181** | **0.7838** |

Precision, recall and F1 refer to the **positive dark-pattern class**, not macro averages. The notebook labels Average Precision as “PR-AUC.”

DistilBERT has the highest test F1 and precision, while BiLSTM has the highest recall. DistilBERT produces **13 false positives and 22 false negatives** on the 294 test examples.

## Explainability and Diagnostics

The notebook includes:

- LIME explanations for the classical classifier.
- Classification reports and confusion matrices.
- ROC and precision–recall curves.
- Training and validation curves for BiLSTM.
- Qualitative inspection of false positives and false negatives.
- 1,000 bootstrap resamples for F1 and Average Precision confidence intervals.

The confidence intervals overlap, so the point estimates should not be interpreted as proof of statistically significant superiority.

## Exported Model Artifacts

Running the export cells creates:

- `model_artifacts/classical_pipeline.pkl`
- `model_artifacts/distilbert_dark_pattern/`
- `model_artifacts/metadata.json`

The metadata records the recommended model, label names, thresholds and maximum transformer input length.

The notebook provides a `predict_dark_pattern()` function for single-text inference.

## Deployment Scope

This notebook implements model training, evaluation, artifact export and inference.

It describes a separate Streamlit/API deployment extension. Those application files must be included separately to run the interface.

The trained models perform **binary classification only**. Any category suggestions provided by a separate application are rule-based explanations, not predictions from an evaluated multiclass model.

## Technologies

- Python, Jupyter Notebook
- pandas, NumPy
- matplotlib, seaborn, wordcloud
- scikit-learn
- Gensim
- TensorFlow/Keras
- PyTorch
- Hugging Face Transformers
- LIME

## Running the Notebook

1. Download or clone the repository.
2. Install the libraries imported by the notebook.
3. Place `patterns.csv` and `pattern_classifications.csv` in the notebook’s working directory, or update their paths.
4. Open `Capstone Project 2_Dark Pattern Detector(2).ipynb`.
5. Run the cells in order.

Initial DistilBERT loading requires internet access to download pretrained model files. A GPU can accelerate training, although the saved notebook run used CPU-based PyTorch.

## Limitations

- The dataset is small and imbalanced.
- Short snippets can lack the context needed to interpret intent.
- Formatting-based deduplication does not remove every semantically similar example.
- Classifications may reflect ambiguous or inconsistent source labels.
- Model scores are not guaranteed real-world probabilities.
- Long inputs may be truncated.
- Results measure performance on this dataset and do not establish reliability across all e-commerce websites.

## Future Improvements

- Expand and independently review the labelled dataset.
- Evaluate generalisation on unseen websites.
- Add surrounding interface context.
- Explore probability calibration.
- Train and evaluate a separate multiclass category model when sufficient reliable category labels are available.
