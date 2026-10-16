# Component 1: Triage Machine Learning Pipeline

## Purpose

Build and export a lightweight PyTorch classifier that labels each incoming
customer message as either `0` (`Normal`) or `1` (`Urgent`). The classifier
must work with both English and Arabic e-commerce support text and must be
small enough for low-latency inference in the backend.

## Scope

- Download and combine the English and Arabic review datasets.
- Normalize the source labels into the two application labels.
- Clean and tokenize multilingual text.
- Train, validate, and evaluate a bidirectional recurrent classifier.
- Export the model and every preprocessing artifact required for inference.

## Dependencies

Install the following packages in the ML environment:

```text
torch
scikit-learn
pandas
numpy
nltk
datasets
```

## Implementation

### 1. Prepare the data

Create `ml_pipeline/preprocess.py` with a repeatable data preparation
function. It should:

1. Download `IberaSoft/ecommerce-reviews-sentiment`.
2. Download `Ruqiya/Arabic_Reviews_of_SHEIN`.
3. Identify the review text and sentiment columns from each dataset.
4. Normalize whitespace, remove empty records, and preserve Arabic characters.
5. Map negative, angry, or strongly dissatisfied records to `1`.
6. Map neutral and positive records to `0`.
7. Add a `language` field (`en` or `ar`) for analysis.
8. Balance the two classes without leaking records between splits.
9. Save deterministic train, validation, and test files in `data/processed/`.

The split must be stratified by urgency label and preferably by language so
that both languages are represented in every evaluation set.

### 2. Build the text representation

Use one shared vocabulary for both languages. Reserve IDs for padding and
unknown tokens, and persist the vocabulary with the model checkpoint. The
preprocessing code used during training must be importable by the backend so
inference cannot silently use a different tokenization scheme.

Recommended sequence behavior:

- Normalize casing only where it is safe for the source language.
- Truncate long messages to a documented maximum length.
- Pad shorter sequences with the reserved padding token.
- Store the maximum length and vocabulary configuration in the checkpoint
  metadata.

### 3. Implement and train the classifier

Create `ml_pipeline/train_classifier.py` containing:

- `UrgencyClassifier(nn.Module)`.
- An embedding layer.
- A bidirectional LSTM or GRU.
- A pooled or final hidden representation.
- A linear layer producing two class logits.

Train with `CrossEntropyLoss` and AdamW. Track validation loss, accuracy, and
macro or weighted F1 after every epoch. Select the checkpoint using validation
F1 rather than training loss alone. Set random seeds and record the training
configuration for reproducibility.

### 4. Evaluate quality

Report results for:

- Overall accuracy and F1.
- English-only accuracy and F1.
- Arabic-only accuracy and F1.
- The confusion matrix.
- A small manually inspected set of urgent and normal examples.

If one language or class performs materially worse, rebalance or improve
preprocessing before exporting the model.

### 5. Export the inference bundle

Save the final artifacts under `backend/saved_models/`:

```text
triage_model.pth
triage_vocab.json
triage_config.json
```

The checkpoint must include the architecture dimensions, class mapping,
maximum sequence length, and preprocessing version. The backend service should
be able to load the bundle without importing the training script's global
state.

## Integration contract

The backend triage service will expose a method equivalent to:

```text
classify(message: str) -> int
```

It must return only the documented class values (`0` or `1`) and should also
make the human-readable label available to the API layer. Inference should run
in evaluation mode and should not calculate gradients.

## Completion criteria

- Both source datasets are reproducibly transformed into the two target labels.
- The model and preprocessing artifacts are exported.
- Evaluation metrics are recorded for both languages.
- A clean environment can load the exported bundle and classify a sample
  message.
- The backend can consume the exported artifacts without retraining.

## Execution status

- [x] Created the root `requirements.txt` and installed its dependencies in
  the project `.venv`, created from the `master` Conda Python installation.
- [x] Implemented reusable multilingual preprocessing, vocabulary, model,
  training, and evaluation modules under `ml_pipeline/`.
- [x] Validated Python syntax, model construction, and end-to-end training and
  artifact export with deterministic synthetic data.
- [x] Selected `hugginglearners/amazon-reviews-sentiment-analysis` as the
  replacement English dataset. It contains a supported CSV with
  `reviewText` and `overall` fields.
- [x] Replace the placeholder in `.env` with a Hugging Face token if desired.
  The selected datasets are public, so a token is optional unless rate limits
  require authentication.
- [x] Prepared deterministic train, validation, and test splits from the
  approved English and Arabic datasets.
- [x] Trained and exported the triage model, vocabulary, and configuration to
  `backend/saved_models/`.
- [x] Evaluated the exported model: overall test accuracy `0.79`, weighted F1
  `0.79`; English accuracy `0.83`; Arabic accuracy `0.76`.
