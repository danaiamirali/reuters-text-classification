from utils import config
import torch
from torch.utils.data import Dataset, DataLoader
import regex as re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import string
import nltk
from nltk.corpus import stopwords
import os
from bs4 import BeautifulSoup

nltk.download("stopwords", quiet=True)

class Loader:
    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Applies basic preprocessing to the given text, including:
            - Cleaning whitespace.
            - Removing punctuation.
            - Removing non-alphanumeric characters.
            - Removing stopwords.

        We leave casing and more advanced preprocessing techniques 
        to be applied as needed outside of this function.

        Args:
            text (str): The text to be preprocessed.
        
        Returns:
            str: The preprocessed text.
        """
        # Remove non-alphanumeric characters
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text.strip())
        # Remove whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove punctuation and stopwords
        text = " ".join(
            [
                word
                for word in text.split()
                if word not in string.punctuation and word not in stopwords.words("english")
            ]
        )
        return text

    @staticmethod
    def load_file(file_path: str) -> list[dict]:
        """
        Load the REUTERS documents from an individual SGM file.

        Args:
            file_path (str): The path to the SGM file.
        
        Returns:
            list[dict]: A list of dictionaries, where each dictionary contains the
            following keys:
                - title: The title of the document.
                - body: The body of the document.
                - topics: The topics of the document.
                - places: The places mentioned in the document.
                - people: The people mentioned in the document.
                - orgs: The organizations mentioned in the document.
                - exchanges: The stock exchanges mentioned in the document.
                - companies: The companies mentioned in the document.
        """
        if file_path.endswith(".sgm"):
            try:
                with open(file_path, "r", encoding="utf8") as f:
                    doc = f.read()
            except:
                print(f"Error reading file: {file_path}")
                return []

        documents = []
        soup = BeautifulSoup(doc, "html.parser")
        reuters_elements = soup.find_all("reuters")

        for element in reuters_elements:
            title = element.find("title")
            body = element.find("body")
            topics = [topic.text for topic in element.find("topics").find_all("d")]
            places = [place.text for place in element.find("places").find_all("d")]
            people = [person.text for person in element.find("people").find_all("d")]
            orgs = [org.text for org in element.find("orgs").find_all("d")]
            exchanges = [exchange.text for exchange in element.find("exchanges").find_all("d")]
            companies = [company.text for company in element.find("companies").find_all("d")]

            # Create dictionary for each document, and append to the list
            document = {
                "title": Loader.preprocess_text(title.text) if title else None,
                "body": Loader.preprocess_text(body.text) if body else None,
                "topics": topics if len(topics) > 0 else None,
                "places": places if len(places) > 0 else None,
                "people": people if len(people) > 0 else None,
                "orgs": orgs if len(orgs) > 0 else None,
                "exchanges": exchanges if len(exchanges) > 0 else None,
                "companies": companies if len(companies) > 0 else None
            }

                # print(document)

            documents.append(document)
        
        return documents
        
    @staticmethod
    def load_files(files: list[str]) -> list[dict]:
        """
        Load the REUTERS documents from a list of SGM files.
        """
        documents = []
        for file in files:
            documents.extend(Loader.load_file(file))
        return documents

    @staticmethod
    def load_directory(directory: str, count: int = None) -> list[dict]:
        """
        Load all the REUTERS documents from the SGM files in the given directory.

        Args:
            directory (str): The directory containing the SGM files.
        
        Returns:
            list[dict]: A list of dictionaries, where each dictionary contains the
            following keys:
                - title: The title of the document.
                - body: The body of the document.
                - topics: The topics of the document.
                - places: The places mentioned in the document.
                - people: The people mentioned in the document.
                - orgs: The organizations mentioned in the document.
                - exchanges: The stock exchanges mentioned in the document.
                - companies: The companies mentioned in the document.
        """
        docs = []
        i = 0
        for files in os.listdir(directory):
            if files.endswith(".sgm") and (count is None or i < count):
                file_path = os.path.join(directory, files)
                print(f"Loading {file_path}...")
                documents = Loader.load_file(file_path)
                docs.extend(documents)
                print(f"Loaded {file_path}.")
                i += 1
            
        return docs

    # @staticmethod
    # def get_labels(category: str, path: str = None) -> list[str]:
    #     """
    #     Gets all possible labels for a given category.

    #     Args:
    #         label (str): The category to get labels for.
        
    #     Returns:
    #         list[str]: A list of all possible labels for the given category.
    #     """
        
    #     file = f"../data/all-{category}-strings.lc.txt" if path is None else f"{path}/all-{category}-strings.lc.txt"
    #     try:
    #         with open(file, "r") as f:
    #             return [line.strip() for line in f.readlines()]
    #     except:
    #         print(f"Error reading file: {file}")
    #         return None

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

def preprocess(data: str) -> str:
    if not isinstance(data, str):
        print(type(data))
    # we will keep casing
    # replace numbers
    data = re.sub(r"\d+", "<NUM>", data)
    return data

def get_tokenizer(model: str):
    if model == "bert":
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    elif model == "xlnet":
        from transformers import XLNetTokenizer
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    else:
        raise f"Model {model} not recognized."
    
    return tokenizer

def get_train_test_loaders(
    model: str,
    split: float = 0.75,
    filter: bool = True,
    filter_threshold: int = 10,
    num_files: int = None
) -> tuple[DataLoader, DataLoader, int]:
    print("Loading data...")
    data_path = config("dataset_path")
    if num_files == None:
        data = Loader.load_directory(data_path)
    else:
        f = data_path + "/{fn}.sgm"
        files = [f.format(fn='reut2-00'+str(i)) for i in range(10)]
        data = Loader.load_files(files)

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

    tokenizer = get_tokenizer(model)

    training_set = CustomDataset(train_dataset, tokenizer, 200)
    testing_set  = CustomDataset(test_dataset, tokenizer, 200)

    train_params = config("train_params")
    test_params = config("test_params")

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    return training_loader, testing_loader, len(topics)
