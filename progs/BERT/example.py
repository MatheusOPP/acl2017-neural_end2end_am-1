import torch
from transformers import BertTokenizer, BertForTokenClassification
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

dataset = pd.read_csv('data/conll/Paragraph_Level/train.dat',sep='\t')

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForTokenClassification.from_pretrained(model_name, num_labels=67)

dataset = pd.read_csv('data/conll/Paragraph_Level/train.dat',sep='\t')

max_seq_length = 128

# Generate label_map
def generate_label_map(labels):
    unique_labels = set(label for sentence_labels in labels for label in sentence_labels)
    label_map = {label: i for i, label in enumerate(unique_labels)}
    label_map_rev = {i: label for label, i in label_map.items()}
    return label_map, label_map_rev

# Prepare and tokenize your data
# Assuming X_train, y_train, X_test, y_test are your tokenized sentences and corresponding labels
# Convert sentences and labels into BERT tokens and labels
def tokenize_data(paragraphs, labels):
    tokenized_texts = []
    tokenized_labels = []
    for paragraphs, label in zip(paragraphs, labels):
        # Tokenize the sentence
        tokenized_sentence = tokenizer.encode(' '.join(paragraphs), add_special_tokens=True)
        # Tokenize and pad labels
        tokenized_label = [[l] for l in label]
        tokenized_label = tokenized_label[:max_seq_length-2]  # trim excess tokens
        tokenized_label = [-100] + tokenized_label + [-100] * (max_seq_length - len(tokenized_label) - 1)  # pad
        tokenized_texts.append(tokenized_sentence)
        tokenized_labels.append(tokenized_label)
    return tokenized_texts, tokenized_labels

# Convert data into PyTorch tensors
train_inputs, train_labels = tokenize_data(X_train, y_train)
test_inputs, test_labels = tokenize_data(X_test, y_test)

# Convert inputs and labels into PyTorch tensors
train_inputs = torch.tensor(train_inputs)
train_labels = torch.tensor(train_labels)
test_inputs = torch.tensor(test_inputs)
test_labels = torch.tensor(test_labels)

# Create DataLoader for training and testing
train_data = TensorDataset(train_inputs, train_labels)
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)

test_data = TensorDataset(test_inputs, test_labels)
test_dataloader = DataLoader(test_data, batch_size=32)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, num_labels), labels.view(-1))
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs, labels = batch
        outputs = model(inputs)
        _, preds = torch.max(outputs, 2)
        preds = preds.cpu().numpy().tolist()
        labels = labels.cpu().numpy().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels)

# Convert predictions and labels back to original format
predicted_labels = [label_map_rev[p] for preds, labels in zip(all_preds, all_labels) for p in preds]
true_labels = [label_map_rev[l] for labels in all_labels for l in labels]

# Evaluate
print(classification_report(true_labels, predicted_labels))