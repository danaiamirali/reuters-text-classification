from utils import loader, config
import torch
from torch.utils.data import Dataset, DataLoader
import regex as re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.body = dataframe.body
        self.targets = np.array([topic for topic in dataframe.topics])
        self.max_len = max_len

    def __len__(self):
        return len(self.body)

    def __getitem__(self, index):
        body = str(self.body.iloc[index])
        body = " ".join(body.split())

        inputs = self.tokenizer.encode_plus(
            body,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

def preprocess(data: str) -> torch.Tensor:
    if not isinstance(data, str):
        print(type(data))
    # we will keep casing
    # replace numbers
    data = re.sub(r"\d+", "<NUM>", data)
    return data

def get_train_test_loaders(
    model: str,
    split: float = 0.75,
    filter: bool = True,
    filter_threshold: int = 10,
    num_files: int = None
) -> DataLoader:
    print("Loading data...")
    data_path = config("dataset_path")
    if num_files == None:
        data = loader.load_data(data_path)
    else:
        f = data_path + "/{fn}.sgm"
        files = [f.format(fn='reut2-00'+str(i)) for i in range(10)]
        data = loader.load_files(files)

    print(f"Loaded {len(data)} documents.")

    df = pd.DataFrame(data)
    # drop rows
    df = df[["title", "body", "topics"]]
    df = df.dropna()
    df["body"] = (df["title"] + df["body"]).apply(preprocess)
    df = df.drop("title", axis=1)

    num_appearances = df["topics"].value_counts()
    if filter:
        topics_to_keep = num_appearances[num_appearances > filter_threshold].index
        old_len = len(df.index)
        df = df[df["topics"].isin(topics_to_keep)]
        new_len = len(df.index)
        print(f"Filtered {old_len - new_len} records.")
    train_dataset, test_dataset = train_test_split(df, train_size=split, random_state=42, shuffle=True, stratify=df["topics"])

    mlb = MultiLabelBinarizer()
    mlb.fit(df["topics"].to_list())
    topics = mlb.classes_
    train_topics = mlb.transform(train_dataset["topics"].to_list())
    test_topics = mlb.transform(test_dataset["topics"].to_list())

    train_dataset["topics"] = list(train_topics)
    test_dataset["topics"] = list(test_topics)

    if model == "bert":
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    elif model == "xlnet":
        from transformers import XLNetTokenizer
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    else:
        raise f"Model {model} not recognized."

    training_set = CustomDataset(train_dataset, tokenizer, 200)
    testing_set  = CustomDataset(test_dataset, tokenizer, 200)

    train_params = config("train_params")
    test_params = config("test_params")

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    return training_loader, testing_loader, len(mlb.classes_)
