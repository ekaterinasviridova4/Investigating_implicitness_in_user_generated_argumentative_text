This repository contains the dataset and implementation details of the paper "Is there anything more deceptive than an obvious fact? Investigating implicitness in user-generated argumentative text" accepted at LREC 2026.

The project focuses on three main tasks:

1.  **Premise/Claim Identification**: Distinguishing between premises and claims.
2.  **Implicitness/Explicitness Detection**: Determining if an argument component is implicit or explicit.
3.  **Fine-grained Classification**: Classifying argument components into fine-grained categories.

## Project Structure

### Data (`data/`)
The `data` directory contains the datasets used for training and evaluation of encoder-decoder models and for zero-shot and fine-tuning of LLMs.
-   **`data/conll/`**: Annotated datasets in CoNLL format.
    -   **Microtext**: A corpus of short argumentative texts annotated for three tasks.
    -   **CMV (ChangeMyView)**: A dataset from the Reddit r/ChangeMyView community annotated for three tasks.
    -   **Combined**: Merged datasets for encoder-decoder training annotated for three tasks.
-   **`data/jsonl/`**: Processed data in JSONL format, suitable for LLM fine-tuning.

### Source Code (`src/`)
The `src` directory is organized by model type and experimental phase:

#### 1. Preprocessing (`src/preprocessing/`)
-   `preprocess_conll.py`: Converts `.conll` files into `.jsonl` format for LLM training.

#### 2. Encoder-Decoder Baselines (`src/encoder_decoder/`)
Jupyter notebooks for baseline experiments using Transformer models:
-   **BERT**
-   **RoBERTa**
-   **Longformer**

#### 3. LLM Fine-tuning (`src/llm_finetuning/`)
Python scripts for fine-tuning Large Language Models (Mistral 7B/24B, Llama 3 8B) on the argument mining tasks.
-   Scripts are named by model and task (e.g., `mistral_24B_finetune_finegrained.py`).
-   Uses Hugging Face `transformers`, `peft` (LoRA), and `bitsandbytes` (quantization).

#### 4. LLM Zero-Shot (`src/llm_zero_shot/`)
Notebooks and scripts for zero-shot inference.
-   Includes experiments with **GPT**, **Llama 3 8B**, and **Mistral 7B/24B**.
-   Evaluates model performance without pre-training.

#### 5. Evaluation (`src/evaluation/`)
Scripts to evaluate model predictions.
-   Calculates metrics like F1-score, Precision, and Recall for LLM fine-tuning.
-   Includes `evaluate_bert_jaccard.py` for span-based evaluation.

### Configuration (`configs/`)
Contains configuration scripts for running experiments on high-performance clusters (using OAR scheduler).
-   `oar_config_llama.sh`, `oar_config_mistral.sh`: Job submission scripts with environment variables and resource requests (GPU, time, memory).

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Install dependencies**:
    It is recommended to use Python 3.10+ and a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The project requires `transformers>=4.57.0` and PyTorch 2.0+.*


## Cite us

