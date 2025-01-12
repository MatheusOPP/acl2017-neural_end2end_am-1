import torch
from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser for Training and Testing')

    parser.add_argument('--train_file', type=str, required=True, help='Path to the training file')
    parser.add_argument('--test_file', type=str, required=True, help='Path to the testing file')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs for training (default: 10)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training (default: 16)')
    parser.add_argument('--target', type=str, choices=['component', 'relation', 'all'], default='all',
                        help="Target to train for, can be 'component', 'relation', or 'all' (default: all)")
    parser.add_argument('--output_file', type=str, help='Path to the output file')

    return parser.parse_args()

def accuracy_score(y_true, y_pred):
    total_samples = sum(len(sublist) - np.count_nonzero(sublist == -100) for sublist in y_true)
    correct_predictions = 0
    
    for trues, preds in zip(y_true, y_pred):
        for true, pred in zip(trues, preds):
            if true == pred and true != -100:
                correct_predictions += 1
    
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    return accuracy

import numpy as np

def f1_score(y_true, y_pred):
    # Calculate precision and recall
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for trues, preds in zip(y_true, y_pred):
        for true, pred in zip(trues, preds):
            if true != -100:
                if true == pred:
                    true_positives += 1
                elif pred != -100:
                    false_positives += 1
            elif pred != -100:
                false_negatives += 1
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    # Calculate F-1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1

def read_dataset(file_path, target='component'):
    with open(file_path, 'r') as file:
        splits = file.read().strip().split('\n\n')  # Split by empty lines to get paragraphs
        data = []
        tags = []
        for split in splits:
            rows = split.strip().split('\n')  # Split by lines to get sentences
            tokens = []
            labels = []
            for row in rows:
                parts = row.split('\t')
                tokens.append(parts[1])  # Add token
                if target == 'component':
                    labels.append(parts[-1].split(":")[0])
                elif target == 'relation':
                    labels.append(':'.join(parts[-1].split(":")[-2:]) if "MajorClaim" not in parts[-1] else 'O')
                elif target == 'all':
                    labels.append(parts[-1])
            data.append(tokens)  # Concatenate tokens into a single string for each paragraph
            tags.append(labels)
        return data, tags    
    


def tokenize_and_align_labels(tags, tokenized_inputs, label_all_tokens=True):
    tags_ids = [[tag2id[tag] for tag in doc] for doc in tags]

    labels = []
    for i, label in enumerate(tags_ids):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)
        
    return labels



class PersuasiveEssaysDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

args = parse_arguments()

model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

# Example usage:
train_file_path = args.train_file
test_file_path = args.test_file 

target = args.target

X_train, y_train = read_dataset(train_file_path, target=target)
X_test, y_test = read_dataset(test_file_path, target=target)

max_seq_length = min(max(len(tokens) for tokens in X_train+X_test), 512)

unique_tags = set(tag for doc in y_train+y_test for tag in doc)
tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

train_encodings = tokenizer.batch_encode_plus(X_train, is_split_into_words=True, add_special_tokens=True, truncation=True, return_offsets_mapping=True, max_length=max_seq_length, padding='max_length')        # Tokenize and pad labels"
val_encodings = tokenizer.batch_encode_plus(X_val, is_split_into_words=True, add_special_tokens=True, truncation=True, return_offsets_mapping=True, max_length=max_seq_length, padding='max_length')
test_encodings = tokenizer.batch_encode_plus(X_test, is_split_into_words=True, add_special_tokens=True, truncation=True, return_offsets_mapping=True, max_length=max_seq_length, padding='max_length')

train_labels = tokenize_and_align_labels(y_train, train_encodings)
val_labels = tokenize_and_align_labels(y_val, val_encodings)
test_labels = tokenize_and_align_labels(y_test, test_encodings)

print(train_labels)

train_encodings.pop("offset_mapping") # we don't want to pass this to the model
val_encodings.pop("offset_mapping")
test_encodings.pop("offset_mapping")

train_dataset = PersuasiveEssaysDataset(train_encodings, train_labels)
val_dataset = PersuasiveEssaysDataset(val_encodings, val_labels)
test_dataset = PersuasiveEssaysDataset(test_encodings, test_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = DistilBertForTokenClassification.from_pretrained('distilbert-base-uncased', num_labels=len(unique_tags))
model.to(device)

optim = torch.optim.AdamW(model.parameters(), lr=5e-5)

epochs = args.n_epochs

for epoch in tqdm(range(epochs)):
    model.train()
    
    train_predictions = []
    train_losses = []
    train_accuracy = []
    train_true_labels = []
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        train_losses.append(loss.item())
        
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2)
        train_predictions.extend(predictions.cpu().numpy())
        train_true_labels.extend(labels.cpu().numpy())
        loss.backward()
        optim.step() 
    
    # Validation loop
    model.eval()
    val_losses = []
    val_predictions = []
    true_labels = []
    with torch.no_grad():
        for val_batch in val_loader:
            input_ids = val_batch['input_ids'].to(device)
            attention_mask = val_batch['attention_mask'].to(device)
            labels = val_batch['labels'].to(device)
            val_outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            val_loss = val_outputs.loss
            val_losses.append(val_loss.item())
            
            logits = val_outputs.logits
            predictions = torch.argmax(logits, dim=2)
            val_predictions.extend(predictions.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    avg_val_loss = sum(val_losses) / len(val_losses)
    
    train_accuracy = accuracy_score(train_true_labels, train_predictions)    
    val_accuracy = accuracy_score(true_labels, val_predictions)
    train_f1 = f1_score(train_true_labels, train_predictions)
    val_f1 = f1_score(true_labels, val_predictions)
    
    print(f"Epoch {epoch+1}, Training Loss:' {loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Val F1-Score: {val_f1:.4f}")
    
    model.save_pretrained(args.output_file)