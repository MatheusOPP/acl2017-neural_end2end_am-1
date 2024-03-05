import torch
from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split


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

def read_dataset(file_path):
    with open(file_path, 'r') as file:
        data = file.read().strip().split('\n\n')  # Split by empty lines to get paragraphs
        paragraphs = []
        paragraphs_labels = []
        for paragraph in data:
            sentences = paragraph.strip().split('\n')  # Split by lines to get sentences
            tokens = []
            labels = []
            for sentence in sentences:
                parts = sentence.split('\t')
                tokens.append(parts[1])  # Add token
                labels.append(parts[-1].split(":")[0])
            paragraphs.append(tokens)  # Concatenate tokens into a single string for each paragraph
            paragraphs_labels.append(labels)
        return paragraphs, paragraphs_labels

model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

# Example usage:
train_file_path = 'data/conll/Paragraph_Level/train.dat' 
test_file_path = 'data/conll/Paragraph_Level/test.dat' 

X_train, y_train = read_dataset(train_file_path)
X_test, y_test = read_dataset(test_file_path)

max_seq_length = max(len(tokens) for tokens in X_train+X_test)+11

unique_tags = set(tag for doc in y_train for tag in doc)
tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

train_encodings = tokenizer.batch_encode_plus(X_train, is_split_into_words=True, add_special_tokens=True, return_offsets_mapping=True, max_length=max_seq_length, padding='max_length')        # Tokenize and pad labels"
val_encodings = tokenizer.batch_encode_plus(X_val, is_split_into_words=True, add_special_tokens=True, return_offsets_mapping=True, max_length=max_seq_length, padding='max_length')
test_encodings = tokenizer.batch_encode_plus(X_test, is_split_into_words=True, add_special_tokens=True, return_offsets_mapping=True, max_length=max_seq_length, padding='max_length')

def encode_tags(tags, encodings):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels

train_labels = encode_tags(y_train, train_encodings)
val_labels = encode_tags(y_val, val_encodings)
test_labels = encode_tags(y_test, test_encodings)

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

epochs = 100

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