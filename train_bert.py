import pandas as pd
from utils import loader
from models import bert
from sklearn.preprocessing import MultiLabelBinarizer

if __name__ == "__main__":

    print("Loading data...")
    # data = loader.load_data("./data/")
    f = 'data/{fn}.sgm'
    files = [f.format(fn='reut2-00'+str(i)) for i in range(2)]

    data = loader.load_files(files)
    print(f"Loaded {len(data)} documents.")

    df = pd.DataFrame(data)
    # drop rows
    df = df[["title", "body", "topics"]]
    df = df.dropna()

    df["body"] = (df["title"] + df["body"]).apply(bert.preprocess)
    df = df.drop("title", axis=1)

    # load all the topics
    mlb = MultiLabelBinarizer()
    mlb.fit(df["topics"].to_list())
    topics = mlb.classes_
    topics = mlb.transform(df["topics"].to_list())

    df["topics"] = list(topics)

    MAX_LEN = 200
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 32
    EPOCHS = 3
    LEARNING_RATE = 1e-05
    NUM_LABELS = len(mlb.classes_)

    print("Training model...")
    model = bert.train_model(df, MAX_LEN, TRAIN_BATCH_SIZE, VALID_BATCH_SIZE, EPOCHS, LEARNING_RATE, NUM_LABELS=NUM_LABELS)
    print("Model trained.")

    model.save_pretrained("trained_models/bert")
    print("Model saved to trained_models/bert")