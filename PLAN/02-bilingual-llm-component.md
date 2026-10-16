# Component 2: Bilingual Local Response Generator

## Purpose

Fine-tune and locally deploy a quantized language model that drafts polite,
useful e-commerce support replies in English and Arabic. The model generates a
draft for an agent to review; it does not automatically send messages to
customers.

## Scope

- Prepare instruction/input/response training examples.
- Add an Arabic portion to the primarily English support dataset.
- Fine-tune a 7B-class model with Unsloth and LoRA in 4-bit precision.
- Export a local model or merged adapter for backend inference.
- Define safe generation and language-preservation behavior.

## Dependencies and hardware assumptions

Install these packages in a separate LLM environment:

```text
unsloth
transformers
peft
trl
bitsandbytes
datasets
torch
```

Training targets a GPU with approximately 6 GB of available VRAM. If the
selected base model cannot meet that limit, reduce sequence length or batch
size before changing the product requirements.

## Implementation

### 1. Prepare the dataset

Create a preprocessing script in `llm_pipeline/` that loads
`bitext/Bitext-customer-support-llm-chatbot-training-dataset` and writes
JSONL records using a stable schema:

```json
{
  "instruction": "Draft a helpful support response.",
  "input": "Customer message",
  "response": "Agent-ready response"
}
```

The script should:

- Remove malformed or empty records.
- Normalize field names and whitespace.
- Keep the original customer intent.
- Translate approximately 20 percent of examples to Arabic.
- Mark the target language explicitly when possible.
- Split data before augmentation to avoid duplicate examples across splits.
- Write train and validation JSONL files under `data/processed/`.

Arabic examples must be reviewed for meaning, politeness, and preservation of
names, order numbers, and other placeholders. Do not translate sensitive
identifiers into fabricated values.

### 2. Configure LoRA fine-tuning

Create `llm_pipeline/finetune.py` or an equivalent notebook. Load the selected
7B base model in 4-bit mode and apply LoRA to the standard attention
projection modules supported by that model. Keep the base weights frozen.

Document:

- Base model identifier and license.
- Quantization type.
- LoRA rank, alpha, dropout, and target modules.
- Maximum sequence length.
- Batch size, gradient accumulation, learning rate, and seed.
- Number of epochs, expected to be one or two.

Use `SFTTrainer` with a prompt template that clearly separates instruction,
customer input, and assistant response. The template must be identical during
training and inference.

### 3. Validate the generated drafts

Evaluate held-out examples for:

- Correct language selection.
- Relevance to the stated customer issue.
- Professional and non-confrontational tone.
- No invented refunds, delivery dates, or policy guarantees.
- Preservation of placeholders and order references.

Record sample outputs and basic loss metrics. Human review is required for
Arabic quality because loss alone does not establish translation or tone
quality.

### 4. Export the local model

Save the result to:

```text
backend/saved_models/llm-finetuned/
```

The export must include the tokenizer and the model configuration. If LoRA
weights are kept separate, document the exact base model and adapter loading
sequence. If they are merged, verify that the merged model can be loaded by
the same Transformers version used by the backend.

## Inference contract

The backend service will expose a method equivalent to:

```text
generate_reply(message: str, language: str, urgency: int) -> str
```

The service should provide a short system prompt that identifies the model as
a drafting assistant, requests the input language, and prohibits claiming that
an action was completed. Generation should use bounded token limits and
deterministic or low-temperature settings suitable for support workflows.

## Completion criteria

- Train and validation JSONL files are reproducible.
- The model trains within the documented VRAM target.
- The exported model and tokenizer load from a local path without network
  access.
- English and Arabic samples are reviewed and meet the tone requirements.
- The backend can generate a draft without exposing training-only code.

<!-- BEGIN MASTER PLAN SYNC -->
## Master plan alignment

> This section is generated from [smart-ticketing-system.md](./smart-ticketing-system.md). Do not edit it directly.

## Phase 2: Local LLM (Bilingual Response Generator)

**Goal:** Fine-tune to draft polite, e-commerce support responses in both Arabic and English.

* **Requirements & Libraries:** `unsloth`, `transformers`, `peft`, `trl`, `bitsandbytes`, `datasets`.
* **Dataset:** `bitext/Bitext-customer-support-llm-chatbot-training-dataset` (translate ~20% to Arabic for a bilingual mix).

* **Tasks for AI Assistant:**
  * [ ] **Dataset Formatting:** Write a script to convert the dataset into an Instruction/Input/Response JSONL format.
  * [ ] **Fine-Tuning Script:** Create a Jupyter Notebook or Python script using **Unsloth** to load a 7B model in 4-bit precision.
  * [ ] **LoRA Configuration:** Apply LoRA to standard attention modules to keep VRAM usage under 6GB during training.
  * [ ] **Training:** Use `SFTTrainer` to train the adapter for 1-2 epochs.
  * [ ] **Export:** Merge the LoRA weights and save the model locally to `backend/saved_models/llm-finetuned/`.

<!-- END MASTER PLAN SYNC -->
