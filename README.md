# LoRA Fine-Tuning for Text Classification with RoBERTa

This repository demonstrates fine-tuning a pre-trained RoBERTa model on the AG News dataset using **Low-Rank Adaptation (LoRA)** for text classification. The project achieves high accuracy while significantly reducing the number of trainable parameters, making it suitable for environments with limited computing resources.

## Overview

The AG News dataset is a collection of news articles categorized into 4 classes: World, Sports, Business, and Sci/Tech. This implementation uses LoRA - a parameter-efficient fine-tuning technique that reduces trainable parameters by 99.2% while maintaining competitive performance.

## Features

- **Parameter-efficient fine-tuning** with LoRA adapter layers
- **Pre-trained RoBERTa model** from Hugging Face
- **Custom training loop** with PyTorch
- **Comprehensive evaluation metrics** including accuracy, precision, recall, and F1 score
- **Modular code structure** for easy adaptation to other datasets

## Installation

To set up the environment, install the required dependencies:

```bash
pip install torch transformers datasets peft accelerate scikit-learn matplotlib
```

## Model Architecture

The model uses RoBERTa as a base with LoRA adapters attached to:
- Query and value matrices in attention layers
- Linear layers
- Word and position embeddings

```python
lora_config = LoraConfig(
    r=4,                      # Rank of the update matrices
    lora_alpha=16,            # Scaling factor
    target_modules=["query", "value", "linear", "word_embeddings", "position_embeddings"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS
)
```

## Training Parameters

- **Batch Size**: 64
- **Learning Rate**: 1e-4
- **Epochs**: 3
- **Weight Decay**: 0.01
- **Warmup Steps**: 500
- **LoRA Rank**: 4
- **LoRA Alpha**: 16

## Performance Metrics

| Metric | Value |
|--------|-------|
| Test Accuracy | 93.54% |
| Test Precision | 93.55% |
| Test Recall | 93.54% |
| Test F1 Score | 93.53% |
| Trainable Parameters | 950,384 (0.76% of total) |
| Total Parameters | 125,599,092 |

## Training Progress

The model shows consistent improvement during training:

| Epoch | Training Loss | Validation Loss | Validation Accuracy |
|-------|---------------|-----------------|---------------------|
| 1     | 0.237         | 0.216           | 92.66%              |
| 2     | 0.168         | 0.186           | 93.78%              |
| 3     | 0.143         | 0.194           | 93.64%              |

## Results Visualization

### Training Loss
![Training Loss](https://github.com/user-attachments/assets/be663e40-21e2-43ad-a3dd-29a7802bc97b)


### Validation Loss
![Validation Loss](https://github.com/user-attachments/assets/d86440f8-757e-4cf9-95fc-e7e71e79813a)


### Validation Accuracy
![Validation Accuracy](https://github.com/user-attachments/assets/786dbb25-fe32-49c4-bdfe-cc5891aacb19)


## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/kiritgpillai/LoRA-AGNews-Classifier.git
   ```

2. Run the notebook to:
   - Load and preprocess the AG News dataset
   - Initialize the RoBERTa model with LoRA
   - Train the model with parameter-efficient fine-tuning
   - Evaluate on test data

3. For inference on new data:
   ```python
   # Load the fine-tuned model
   model = AutoModelForSequenceClassification.from_pretrained("./final_model")
   tokenizer = AutoTokenizer.from_pretrained("./final_model")
   
   # Inference
   inputs = tokenizer("This is a news article about technology", return_tensors="pt")
   outputs = model(**inputs)
   predictions = torch.argmax(outputs.logits, dim=-1)
   ```

---

### Authors
Kirit Govindaraja Pillai - kx2222@nyu.edu  
Ruochong Wang - rw3760@nyu.edu  
Saketh Raman Ramesh - sr7714@nyu.edu  

---
