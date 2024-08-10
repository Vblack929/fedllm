import os
import pandas as pd
import random

# Define the base path for the dataset
data_basepath = "./datasets/imdb"

# Function to load text and labels from files
def load_split(data_dir, split):
    # Load text and labels
    with open(os.path.join(data_dir, f"{split}.txt"), 'r') as text_file:
        texts = text_file.readlines()

    with open(os.path.join(data_dir, f"{split}_labels.txt"), 'r') as label_file:
        labels = [int(label.strip()) for label in label_file.readlines()]

    # Return a list of tuples (text, label, 0) - third element can be used for additional processing
    return [(text.strip(), label, 0) for text, label in zip(texts, labels)]

# Load train and test datasets
train_data = load_split(data_basepath, "train")
test_data = load_split(data_basepath, "test")

# Split train data to create a dev set if it doesn't exist
dev_rate = 0.1
dev_split_index = int(len(train_data) * dev_rate)
random.shuffle(train_data)  # Shuffle the train data before splitting
dev_data = train_data[:dev_split_index]
train_data = train_data[dev_split_index:]

# Optional: Save the dev split to disk (for consistency)
def save_split(data, split, basepath):
    texts, labels, _ = zip(*data)  # Unpack the list of tuples
    with open(os.path.join(basepath, f"{split}.txt"), 'w') as text_file:
        for text in texts:
            text_file.write(f"{text}\n")

    with open(os.path.join(basepath, f"{split}_labels.txt"), 'w') as label_file:
        for label in labels:
            label_file.write(f"{label}\n")

# Save the dev data
save_split(dev_data, "dev", data_basepath)