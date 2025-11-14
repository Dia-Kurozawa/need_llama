"""
BERT-based Hard Voting Ensemble Model for Novel Classification Task

This script implements a BERT-based hard voting ensemble model for classifying text samples 
as either common or unique novels. The model uses hard voting ensemble technique 
with data augmentation and R-Drop regularization to improve classification performance.

Key features:
1. BERT-based classification with fine-tuning
2. Data augmentation using curriculum learning
3. R-Drop regularization to improve model stability
4. Hard voting ensemble of multiple models
5. Comprehensive evaluation metrics
"""

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from balanced_loss import Loss
from transformers import BertForSequenceClassification, BertTokenizer, TrainingArguments, Trainer, set_seed

# Set random seed for reproducibility
set_seed(42)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import os

# Change working directory - Note: This path is specific to the original author's environment
os.chdir('/HOME/shuang/HSURA')

# ==================== Configuration ====================
# Model and training parameters
model_name = "bert-base-uncased"  # Pretrained BERT model
BATCH_SIZE = 16                   # Training batch size
max_len = 512                     # Maximum sequence length
LR = 5e-5                         # Learning rate
EPOCH = 2                         # Number of training epochs
alpha = 2                         # R-Drop regularization weight
loss_type = "cross_entropy"       # Loss function type
class_balanced = False            # Whether to use class balanced loss

# Random state for data splitting
random_state = 111

# Data files
main_data_file = 'novel_generate_v3.xlsx'  # Main dataset
augmentation_files = [
    'novel_generate_v3_onlyU_curri2_1.xlsx',   # Augmentation dataset 1
    'novel_generate_v3_onlyU_curri2_2.xlsx',   # Augmentation dataset 2
    'novel_generate_v3_onlyU_curri2_3.xlsx'    # Augmentation dataset 3
]

# Output files - include random_state in filenames for experiment tracking
df_vote_name = f'df_vote_{loss_type}_{random_state}_2-123_dummy_epoch{EPOCH}.xlsx'
df_rank_name = f'df_rank_{loss_type}_{random_state}_2-123_dummy_epoch{EPOCH}.xlsx'

# Column names for storing prediction results
columns = ['2^1', '2^2', '2^3', 'label', 'hard_vote', 'X_test']

# ==================== Data Loading ====================
# Load main dataset
df = pd.read_excel(main_data_file, keep_default_na=False)
idx_values = df.index.values.tolist()
text_values = df.text.values.tolist()
label_values = df.label.values.tolist()

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)

# Initialize dataframes for storing results
df_vote = pd.DataFrame(columns=columns)
df_rank_concat = pd.DataFrame()

# ==================== Custom Dataset Class ====================
class Dataset(torch.utils.data.Dataset):
    """
    Custom dataset class for handling tokenized text data
    """
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Convert encodings to tensors
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

# ==================== Metrics Computation ====================
def compute_metrics(p):
    """
    Compute accuracy metric for model evaluation
    
    Args:
        p: Tuple of predictions and labels
        
    Returns:
        Dictionary with accuracy metric
    """
    pred, labels = p
    pred = np.argmax(pred, axis=-1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    return {"accuracy": accuracy}

# ==================== Custom Trainer with R-Drop ====================
class CustomTrainer(Trainer):
    """
    Custom trainer class implementing R-Drop regularization
    R-Drop encourages consistent predictions for the same input by minimizing
    the KL divergence between outputs of two forward passes
    """
    
    def compute_kl_loss(self, p, q):
        """
        Compute KL divergence loss between two probability distributions
        
        Args:
            p, q: Logits from two forward passes
            
        Returns:
            KL divergence loss
        """
        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
        # Sum losses across dimensions
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()
        loss = (p_loss + q_loss) / 2
        return loss

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute total loss including cross-entropy and R-Drop regularization
        
        Args:
            model: BERT model
            inputs: Input batch
            return_outputs: Whether to return model outputs
            
        Returns:
            Total loss and optionally model outputs
        """
        labels = inputs.get("labels")
        # Two forward passes for R-Drop
        outputs_1 = model(**inputs)
        outputs_2 = model(**inputs)      
        logits_1 = outputs_1.get('logits')
        logits_2 = outputs_2.get('logits')
        
        # Compute cross-entropy loss
        # Using dummy class (3 classes instead of 2) to improve performance
        loss_fct = Loss(
            loss_type=loss_type,
            samples_per_class=[6590, 92, 0],  # Sample counts for each class (including dummy)
            class_balanced=class_balanced
        )
        ce_loss = 0.5 * (loss_fct(logits_1, labels) + loss_fct(logits_2, labels))
        
        # Compute KL divergence loss for R-Drop
        kl_loss = self.compute_kl_loss(logits_1, logits_2)
        
        # Total loss = Cross-entropy loss + α * KL divergence loss
        loss = ce_loss + kl_loss * alpha
        return (loss, outputs_1) if return_outputs else loss

# ==================== Main Training and Evaluation Loop ====================
# Process each augmentation file
for times, excel_name in enumerate(augmentation_files):
    print(f"Processing augmentation file {times+1}: {excel_name}")
    
    # Initialize model with dummy class (3 classes) for better performance
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
    
    # Split data into train/validation and test sets
    # First split: 60% train/val, 40% test
    idx_train_ori, idx_test, X_train_ori, X_test, y_train_ori, y_test = train_test_split(
        idx_values, text_values, label_values, shuffle=True, test_size=0.4, random_state=random_state)
    
    # Second split: 2/3 train, 1/3 validation (overall 40% train, 20% validation)
    idx_train, idx_val, X_train, X_val, y_train, y_val = train_test_split(
        idx_train_ori, X_train_ori, y_train_ori, shuffle=True, test_size=1/3, random_state=random_state)
    
    # Apply curriculum learning data augmentation
    # Select only unique (label=1) samples from training set for augmentation
    df_train = pd.DataFrame({'chosen': idx_train, 'label': y_train})
    idx_train_onlyU = df_train[df_train.label == 1].chosen.to_list()

    # Load augmentation data and apply to training set
    df_curri = pd.read_excel(excel_name, keep_default_na=False)
    X_train = X_train + df_curri.loc[idx_train_onlyU].text.to_list() 
    y_train = y_train + df_curri.loc[idx_train_onlyU].label.to_list()

    # Tokenize text data
    X_train_tokenized = tokenizer(X_train, padding='max_length', truncation=True, max_length=max_len)
    X_test_tokenized = tokenizer(X_test, padding='max_length', truncation=True, max_length=max_len)
    
    # Create dataset objects
    train_dataset = Dataset(X_train_tokenized, y_train)
    val_dataset = Dataset(X_test_tokenized, y_test)
    
    # Training arguments
    args = TrainingArguments(
        output_dir="output",
        evaluation_strategy="epoch",           # Evaluate after each epoch
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        num_train_epochs=EPOCH,
        save_strategy='epoch',                 # Save model after each epoch
        fp16=True,                             # Use mixed precision training
        weight_decay=0.01,                     # Weight decay for regularization
    )
    
    # Initialize custom trainer
    trainer = CustomTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    trainer.train()
    
    # Create test dataset and make predictions
    test_dataset = Dataset(X_test_tokenized)
    raw_pred, _, _ = trainer.predict(test_dataset)

    # Post-process predictions - hard voting uses class predictions (0 or 1)
    y_pred = np.argmax(raw_pred, axis=1)
    
    # Print evaluation results for this model
    print(f'Augmentation file: {excel_name}')
    print(classification_report(y_test, y_pred, target_names=["Common", "Unique"], digits=4)) 
    print(confusion_matrix(y_test, y_pred))
    
    # Store predictions for ensemble
    if times > 1:
        df_rank = pd.DataFrame({f'y_pred_{times}': raw_pred[:, 1], 'y_test': y_test})
    else:
        df_rank = pd.DataFrame({f'y_pred_{times}': raw_pred[:, 1]})
    
    # Save raw predictions for hard voting
    df_vote[columns[times]] = y_pred                       # Hard predictions for each model
    df_vote['label'] = y_test                              # True labels
    df_vote['X_test'] = X_test                             # Test texts
    
    # Combine ranking results
    df_rank_concat = pd.concat([df_rank_concat, df_rank], axis=1)

# ==================== Hard Voting Ensemble ====================
# Perform hard voting by taking majority vote from all models
for i in range(len(df_vote)):
    df_vote.iloc[i, -2] = np.argmax(np.bincount(df_vote.iloc[i, :len(augmentation_files)]))

df_vote.hard_vote = df_vote.hard_vote.astype(int)

# Print ensemble results
print('Hard Voting Ensemble Results')
print(classification_report(df_vote.label, df_vote.hard_vote, target_names=['Common', 'Unique'], digits=4)) 
print(confusion_matrix(df_vote.label, df_vote.hard_vote))

# Save results to Excel files
df_vote.to_excel(df_vote_name, index=False)
df_rank_concat.to_excel(df_rank_name, index=False)

print("Training and evaluation completed. Results saved to Excel files.")