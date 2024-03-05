import torch
from transformers import BertTokenizer, BertForTokenClassification
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import pandas as pd
from tqdm import tqdm

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
    
# Generate label_map
def generate_label_map(labels):
    unique_labels = set(label for sentence_labels in labels for label in sentence_labels)
    label_map = {label: i for i, label in enumerate(unique_labels)}
    label_map_rev = {i: label for label, i in label_map.items()}
    return label_map, label_map_rev


def tokenize_data(paragraphs, labels):
    tokenized_texts = []
    tokenized_labels = []
    mask = []
    for paragraphs, label in zip(paragraphs, labels):
        # Tokenize the sentence
        tokenized_paragraph = tokenizer.encode(' '.join(paragraphs), add_special_tokens=True, max_length=max_seq_length, padding='max_length')        # Tokenize and pad labels
        tokenized_label = [label_map[l] for l in label]
        tokenized_label = tokenized_label[:max_seq_length-2]  # trim excess tokens
        tokenized_label = [-100] + tokenized_label + [-100] * (max_seq_length - len(tokenized_label) - 1)  # pad
        
        attn_mask = [1 if tok != -100 else 0 for tok in tokenized_paragraph]
        
        tokenized_texts.append(tokenized_paragraph)
        tokenized_labels.append(tokenized_label)
        mask.append(attn_mask)
    return torch.tensor(tokenized_texts), torch.tensor(tokenized_labels), torch.tensor(mask)

def generate_label_map(labels):
    unique_labels = set(label for sentence_labels in labels for label in sentence_labels)
    label_map = {label: i for i, label in enumerate(unique_labels)}
    label_map_rev = {i: label for label, i in label_map.items()}
    
    label_map['[PAD]'] = -100
    label_map_rev[-100] = '[PAD]'
    
    return label_map, label_map_rev

def get_unique_elements(list_of_lists):
    # Flatten the list of lists
    flattened_list = [item for sublist in list_of_lists for item in sublist]
    # Get unique elements from the flattened list
    unique_elements = list(set(flattened_list))
    return unique_elements


model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)

# Example usage:
train_file_path = 'data/conll/Paragraph_Level/train.dat' 
test_file_path = 'data/conll/Paragraph_Level/test.dat' 

X_train, y_train = read_dataset(train_file_path)
X_test, y_test = read_dataset(test_file_path)

max_seq_length = max(len(tokens) for tokens in X_train+X_test)+2

label_map, label_map_rev = generate_label_map(y_train)

train_inputs, train_labels, train_mask = tokenize_data(X_train, y_train)
test_inputs, test_labels, test_mask = tokenize_data(X_test, y_test)

# Create DataLoader for training and testing
train_data = TensorDataset(train_inputs, train_labels, train_mask)
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)

test_data = TensorDataset(test_inputs, test_labels, test_mask)
test_dataloader = DataLoader(test_data, batch_size=32)

num_labels = len(get_unique_elements(y_train))
model = BertForTokenClassification.from_pretrained(pretrained_model_name_or_path='model_seq_tag_14')
criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
num_epochs = 50
learning_rate = 0.1

# # Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# for epoch in tqdm(range(num_epochs)):
#     model.train()
#     for i, batch in enumerate(train_dataloader):
#         batch = tuple(t.to(device) for t in batch)
#         inputs, labels, masks = batch
#         optimizer.zero_grad()
#         outputs = model(inputs, attention_mask = masks)
#         logits = outputs.logits  # Access logits from the TokenClassifierOutput
#         logits = logits.view(-1, num_labels)  # Reshape logits
#         labels = labels.view(-1)  # Reshape labels
#         loss = criterion(logits, labels)
#         loss.backward()
#         optimizer.step()
        
#     model.save_pretrained('model_seq_tag_{}'.format(epoch))

# Evaluation
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for batch in tqdm(test_dataloader):
        batch = tuple(t.to(device) for t in batch)
        inputs, labels, masks = batch
        outputs = model(inputs, attention_mask=masks)
        logits = outputs.logits  # Extract logits from TokenClassifierOutput
        _, preds = torch.max(logits, 2)  # Compute max along dimension 2 (assuming 2 is the dimension for classes)
        preds = preds.cpu().numpy().tolist()
        labels = labels.cpu().numpy().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels)

# Convert predictions and labels back to original format
predicted_labels = [label_map_rev[p] for preds, labels in zip(all_preds, all_labels) for p in preds]
true_labels = [label_map_rev[l] for labels in all_labels for l in labels]

# Evaluate
print(predicted_labels)
