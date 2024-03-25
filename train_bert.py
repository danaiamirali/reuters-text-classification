import pandas as pd
from utils import loader
from models import bert
from models.helpers import preprocess
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import torch
import os
import numpy as np

if __name__ == "__main__":

    print("Loading data...")
    # data = loader.load_data("./data/")
    f = 'data/{fn}.sgm'
    files = [f.format(fn='reut2-00'+str(i)) for i in range(15)]
    data = loader.load_files(files)
    print(f"Loaded {len(data)} documents.")

    df = pd.DataFrame(data)
    # drop rows
    df = df[["title", "body", "topics"]]
    df = df.dropna()

    df["body"] = (df["title"] + df["body"]).apply(preprocess)
    df = df.drop("title", axis=1)

    num_appearances = df["topics"].value_counts()
    topics_to_keep = num_appearances[num_appearances > 3].index
    old_len = len(df.index)
    df = df[df["topics"].isin(topics_to_keep)]
    new_len = len(df.index)
    print(f"Filtered {old_len - new_len} records.")
    train_dataset, test_dataset = train_test_split(df, train_size=0.7, random_state=42, shuffle=True, stratify=df["topics"])

    print(df.head())

    # load all the topics
    mlb = MultiLabelBinarizer()
    mlb.fit(df["topics"].to_list())
    topics = mlb.classes_
    train_topics = mlb.transform(train_dataset["topics"].to_list())
    test_topics = mlb.transform(test_dataset["topics"].to_list())

    train_dataset["topics"] = list(train_topics)
    test_dataset["topics"] = list(test_topics)

    MAX_LEN = 200
    TRAIN_BATCH_SIZE = 256
    VALID_BATCH_SIZE = 256
    EPOCHS = 3
    LEARNING_RATE = 1e-05
    NUM_LABELS = len(mlb.classes_)

    print("Training model...")
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    model = bert.train_model(train_dataset, test_dataset, MAX_LEN, TRAIN_BATCH_SIZE, VALID_BATCH_SIZE, EPOCHS, LEARNING_RATE, TRAIN_SIZE=0.7, NUM_LABELS=NUM_LABELS, FREEZE_BERT=True)
    print("Model trained.")

    PATH = f"trained_models/BERT-cased-{EPOCHS}"
    if not os.path.exists("trained_models"):
        os.makedirs("trained_models")
    torch.save(model.state_dict(), PATH)
    print(f"Model saved to {PATH}")