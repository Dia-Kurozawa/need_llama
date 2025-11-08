# Novel Classification with BERT and LLaMA Ensemble Models

This repository contains implementations of ensemble models for novel classification tasks, specifically designed to classify text samples as either "common" or "unique" novels. Two different approaches are implemented using state-of-the-art transformer models: BERT and LLaMA.

## Overview

The project includes two main Python scripts that implement different approaches to novel classification:

1. **BERT-based Ensemble Model**: Uses BERT with soft voting ensemble technique, data augmentation, and R-Drop regularization
2. **LLaMA-based Classification Model**: Uses Meta-Llama-3.1-8B-Instruct with LoRA fine-tuning and quantization techniques

Both models aim to identify unique characteristics in novel texts and classify them accordingly.

## Features

### BERT Ensemble Model (`novel_classification_bert_ensemble.py`)
- BERT-based classification with fine-tuning
- Data augmentation using curriculum learning
- R-Drop regularization to improve model stability
- Soft voting ensemble of multiple models
- Comprehensive evaluation metrics (classification report, confusion matrix)
- Dummy class technique for improved performance

### LLaMA Classification Model (`need_llama.py`)
- LLaMA 3.1 8B Instruct model fine-tuning
- LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- 8-bit quantization for reduced memory usage
- Class-balanced loss function
- ROC-AUC evaluation metrics
- Training and inference time measurements

## Requirements

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- Pandas
- NumPy
- Scikit-learn
- PEFT (for LoRA)
- BitsAndBytes (for quantization)
- Matplotlib (for visualization)
- balanced-loss (for class balanced loss function)

Install required packages:
```bash
pip install torch transformers pandas numpy scikit-learn peft bitsandbytes matplotlib openpyxl balanced-loss
```

## Data Format

The models expect data in Excel format with the following columns:
- `text`: The novel text content
- `label`: Binary label (0 for "Common", 1 for "Unique")

Data files referenced in the code:
- `novel_generate_v3.xlsx`: Main dataset
- `novel_generate_v3_onlyU_curri2_X.xlsx`: Augmentation datasets (where X = 1, 2, 3)

## Usage

### BERT Ensemble Model

```bash
python novel_classification_bert_ensemble.py
```

Key parameters that can be adjusted:
- `model_name`: Pretrained BERT model ("bert-base-uncased")
- `BATCH_SIZE`: Training batch size (default: 16)
- `EPOCH`: Number of training epochs (default: 3)
- `LR`: Learning rate (default: 5e-5)
- `alpha`: R-Drop regularization weight (default: 2)

### LLaMA Classification Model

```bash
python need_llama.py
```

Before running, you need to obtain an LLaMA access token. Follow these easy steps:
1. Head to the official LLaMA repository on Hugging Face: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct.
2. Follow the on-screen instructions to get your LLaMA access token.
3. Once you have your access token, simply insert it into the Python code at line 11, where you'll see the placeholder text "llama_token = '[insert-llama-access-token-here]'. This will enable you to run the code successfully.

Key parameters that can be adjusted:
- `model_name`: Pretrained LLaMA model ("meta-llama/Meta-Llama-3.1-8B-Instruct")
- `BATCH_SIZE`: Training batch size (default: 1)
- `EPOCH`: Number of training epochs (default: 1)
- `quantization_config`: 8-bit or 4-bit quantization settings

## Model Architecture

### BERT Ensemble Approach
1. Uses three instances of BERT models trained on different augmented datasets
2. Applies curriculum learning for data augmentation
3. Implements R-Drop regularization for consistency
4. Combines predictions using soft voting ensemble technique
5. Includes a dummy class (3 classes instead of 2) to improve performance

### LLaMA Approach
1. Utilizes Meta-LLaMA-3.1-8B-Instruct model
2. Applies LoRA for parameter-efficient fine-tuning
3. Uses 8-bit quantization to reduce memory footprint
4. Implements class-balanced loss function to handle imbalanced data
5. Evaluates with ROC-AUC metrics in addition to standard classification metrics

## Results

Both models output comprehensive evaluation metrics:
- Classification reports with precision, recall, and F1-score
- Confusion matrices
- ROC curves and AUC scores (LLaMA model)
- Performance comparison across different random states

Output files:
- Prediction results saved in Excel format
- ROC curve plots (PNG format for LLaMA model)

## Acknowledgments

- Hugging Face for providing the Transformers library
- Meta AI for the LLaMA models
- Google for BERT models
- Authors of R-Drop and curriculum learning techniques