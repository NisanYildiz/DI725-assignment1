"""
Prepare the customer review dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.

Adapted from https://github.com/caglarmert/DI725/tree/main/assignment_1
"""
import os
import pickle
import requests
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Parameters
RANDOM_SEED = 42
TRAIN_SPLIT = 0.8  # 80% training, 20% validation

### Since we are now interested in sentiment analysis, we need seperate
### conversations and labeles saved


input_file_path = os.path.join(os.path.dirname(__file__), 'train.csv')
df = pd.read_csv(input_file_path)

# Print basic dataset info
print(f"Dataset shape: {df.shape}")
print(f"Number of conversations: {len(df)}")

# get all the unique characters that occur in this text

all_text = ' '.join(df['conversation'].astype(str).tolist())
print(f"\nTotal text length: {len(all_text):,} characters")

# Create character vocabulary for text
chars = sorted(list(set(all_text)))
text_vocab_size = len(chars)
print(f"Text vocabulary size: {text_vocab_size} unique characters")
print(f"Character set: {chars}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode_text(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode_text(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


# Create mappings for sentiment labels
# Convert labels to strings just in case they're numeric
sentiment_labels = sorted(list(set(df['customer_sentiment'].astype(str))))
label_vocab_size = len(sentiment_labels)
print(f"\nNumber of unique sentiment labels: {label_vocab_size}")
print(f"Labels: {sentiment_labels}")

label_stoi = {label:i for i,label in enumerate(sentiment_labels)}
label_itos = {i:label for i,label in enumerate(sentiment_labels)}

# Print label to index mapping
print("\nLabel to index mapping:")
for label, idx in label_stoi.items():
    print(f"  {label} -> {idx}")

def encode_label(s):
    return label_stoi[s]
def decode_label(s):
    return label_itos[s]

# Split into train and validation sets
train_df, val_df = train_test_split(df, train_size=TRAIN_SPLIT, random_state=RANDOM_SEED, stratify=df['customer_sentiment'])
print(f"\nTraining set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")

# Process and encode training and test data
train_texts = train_df['conversation'].astype(str).tolist()
train_labels = train_df['customer_sentiment'].astype(str).tolist()

val_texts = val_df['conversation'].astype(str).tolist()
val_labels = val_df['customer_sentiment'].astype(str).tolist()

# Encode all texts and labels
train_text_ids = [encode_text(text) for text in train_texts]
val_text_ids = [encode_text(text) for text in val_texts]
train_label_ids = [encode_label(label) for label in train_labels]
val_label_ids = [encode_label(label) for label in val_labels]

# export 

train_data = {
    'raw_texts': train_texts,
    'text_ids': train_text_ids,
    'label_ids': np.array(train_label_ids, dtype=np.int32)
}

val_data = {
    'raw_texts': val_texts,
    'text_ids': val_text_ids,
    'label_ids': np.array(val_label_ids, dtype=np.int32)
}

# Save the arrays and metadata
with open(os.path.join(os.path.dirname(__file__), 'train_sentiment.pkl'), 'wb') as f:
    pickle.dump(train_data, f)

with open(os.path.join(os.path.dirname(__file__), 'val_sentiment.pkl'), 'wb') as f:
    pickle.dump(val_data, f)

# Save metadata for processing
meta = {
    'text_vocab_size': text_vocab_size,
    'text_itos': itos,
    'text_stoi': stoi,
    'label_vocab_size': label_vocab_size,
    'label_itos': label_itos,
    'label_stoi': label_stoi,
    'sentiment_labels': sentiment_labels
}

with open(os.path.join(os.path.dirname(__file__), 'sentiment_meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("\nData preparation complete!")
