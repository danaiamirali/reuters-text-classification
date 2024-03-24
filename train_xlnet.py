import pandas as pd
from utils import loader
from models import xlnet
from models.helpers import preprocess
from sklearn.preprocessing import MultiLabelBinarizer
import torch
import os

if __name__ == "__main__":

    print("Loading data...")
    # data = loader.load_data("./data/")
    f = 'data/{fn}.sgm'
    files = [f.format(fn='reut2-00'+str(i)) for i in range(10)]
    data = loader.load_files(files)
    print(f"Loaded {len(data)} documents.")

    df = pd.DataFrame(data)
    # drop rows
    df = df[["title", "body", "topics"]]
    df = df.dropna()

    df["body"] = (df["title"] + df["body"]).apply(preprocess)
    df = df.drop("title", axis=1)

    # load all the topics
    mlb = MultiLabelBinarizer()
    mlb.fit(df["topics"].to_list())
    topics = mlb.classes_
    topics = mlb.transform(df["topics"].to_list())

    df["topics"] = list(topics)

    MAX_LEN = 200
    TRAIN_BATCH_SIZE = 256
    VALID_BATCH_SIZE = 256
    EPOCHS = 3
    LEARNING_RATE = 1e-05
    NUM_LABELS = len(mlb.classes_)

    print("Training model...")
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    model = xlnet.train_model(df, MAX_LEN, TRAIN_BATCH_SIZE, VALID_BATCH_SIZE, EPOCHS, LEARNING_RATE, TRAIN_SIZE=0.7, NUM_LABELS=NUM_LABELS, FREEZE_XLNET=True)
    print("Model trained.")

    PATH = f"trained_models/XLNet-cased-{EPOCHS}"
    if not os.path.exists("trained_models"):
        os.makedirs("trained_models")
    torch.save(model.state_dict(), PATH)
    print(f"Model saved to {PATH}")