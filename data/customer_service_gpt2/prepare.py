import os
import requests
import tiktoken
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

RANDOM_SEED = 42
TRAIN_SPLIT = 0.8  # 80% training, 20% validation


input_file_path = os.path.join(os.path.dirname(__file__), 'train.csv')
df = pd.read_csv(input_file_path)

# Print basic dataset info
print(f"Dataset shape: {df.shape}")
print(f"Number of conversations: {len(df)}")

# only get customer comments from the whole conv.

customer_texts = []

for conversation in df["conversation"]:
  customer_text = ""
  for line in conversation.split("\n\n"):
    if line.startswith("Customer:"):
      customer_text += line

  customer_texts.append(customer_text)

df["customer_conversation"] = customer_texts

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

#split into train and val
train_df, val_df = train_test_split(df, train_size=TRAIN_SPLIT, random_state=RANDOM_SEED, stratify=df['customer_sentiment'])
print(f"\nTraining set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")

# Process and encode training and test data
train_texts = train_df["customer_conversation"].astype(str).tolist()
train_labels = train_df['customer_sentiment'].astype(str).tolist()

val_texts = val_df["customer_conversation"].astype(str).tolist()
val_labels = val_df['customer_sentiment'].astype(str).tolist()



# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_text_ids = [enc.encode_ordinary(text) for text in train_texts]
val_text_ids = [enc.encode_ordinary(text) for text in val_texts]
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

# Save the arrays
with open(os.path.join(os.path.dirname(__file__), 'train_sentiment.pkl'), 'wb') as f:
    pickle.dump(train_data, f)

with open(os.path.join(os.path.dirname(__file__), 'val_sentiment.pkl'), 'wb') as f:
    pickle.dump(val_data, f)

print("\nData preparation complete!")
