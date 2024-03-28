"""
Util code for loading the data from the SGM files.
Also includes basic data preprocessing.
"""
import os
from bs4 import BeautifulSoup

import re
import string
import nltk
from nltk.corpus import stopwords
import numpy as np

nltk.download("stopwords", quiet=True)

import threading

class loader:
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
                "title": loader.preprocess_text(title.text) if title else None,
                "body": loader.preprocess_text(body.text) if body else None,
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
        
    def load_files(files: list[str]) -> list[dict]:
        """
        Load the REUTERS documents from a list of SGM files.
        """
        documents = []
        for file in files:
            documents.extend(loader.load_file(file))
        return documents

    def load_data(directory: str, count: int = None) -> list[dict]:
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
                documents = loader.load_file(file_path)
                docs.extend(documents)
                print(f"Loaded {file_path}.")
                i += 1
            
        return docs

    def get_labels(category: str, path:str = None) -> list[str]:
        """
        Gets all possible labels for a given category.

        Args:
            label (str): The category to get labels for.
        
        Returns:
            list[str]: A list of all possible labels for the given category.
        """
        file = f"../data/all-{category}-strings.lc.txt" if path is None else f"{path}/all-{category}-strings.lc.txt"
        try:
            with open(file, "r") as f:
                return [line.strip() for line in f.readlines()]
        except:
            print(f"Error reading file: {file}")
            return None
    
def find_optimal_thresholds(targets: np.ndarray, outputs: np.ndarray, metric_func, candidate_thresholds, num_labels):
    optimal_thresholds = []

    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)
    if not isinstance(outputs, np.ndarray):
        outputs = np.array(outputs)
    
    for label_idx in range(num_labels):
        best_threshold = 0
        best_metric = float('-inf')
        
        for threshold in candidate_thresholds:
            # Apply threshold to specific label
            adjusted_outputs = outputs[:, label_idx] >= threshold
            metric = metric_func(targets[:, label_idx], adjusted_outputs)
            
            if metric > best_metric:
                best_metric = metric
                best_threshold = threshold
                
        optimal_thresholds.append(best_threshold)
    
    return optimal_thresholds

import numpy as np
from sklearn import metrics 

def freeze_model(model):
    for param in model.l1.parameters():
        param.requires_grad = False

    return model

def freeze_layers(model, num_layers):
    for i in range(num_layers):
        for param in model.l1.encoder.layer[i].parameters():
            param.requires_grad = False

    return model

def eval_metrics(targets, outputs, thresholds: list):
    # Ensure targets and outputs are numpy arrays for element-wise operations
    targets = np.array(targets)
    outputs = np.array(outputs)

    # Apply per-label thresholding
    # Assuming outputs and thresholds are appropriately aligned
    for i, threshold in enumerate(thresholds):
        outputs[:, i] = outputs[:, i] >= threshold
    
    # Calculate metrics after thresholding
    accuracy = metrics.accuracy_score(targets, outputs)
    hamming_loss = metrics.hamming_loss(targets, outputs)
    f1_score_micro = metrics.f1_score(targets, outputs, average='micro', zero_division=np.nan)
    f1_score_macro = metrics.f1_score(targets, outputs, average='macro', zero_division=np.nan)
    clf_report = metrics.classification_report(targets, outputs, zero_division=np.nan)

    # Print overall metrics
    print(f"Validation Accuracy Score = {accuracy}")
    print(f"Validation Hamming Loss = {hamming_loss}")
    print(f"Validation F1 Score (Micro) = {f1_score_micro}")
    print(f"Validation F1 Score (Macro) = {f1_score_macro}")
    print(clf_report)

    # Debugging: Write comparison of targets and outputs to file
    with open("debug-log.txt", "w") as log_file:
        log_file.write("Index, Target, Output, Subset Accuracy, Real Accuracy (%)\n")
        for index, (target_row, output_row) in enumerate(zip(targets, outputs)):
            # Here, the comparison is row-wise (per example, not per label)
            correct = np.array_equal(target_row, output_row)
            
            # Calculate subset accuracy
            subset_accuracy = (target_row == output_row).sum() / len(target_row)
            is_correct = 1 if correct else 0
            # Convert arrays to strings for logging
            target_str = np.array2string(target_row, separator=',')
            output_str = np.array2string(output_row, separator=',')
            log_file.write(f"{index}, {target_str}, {output_str}, {is_correct}, {subset_accuracy:.2f}%\n")


def config(attr):
    """
    Retrieves the queried attribute value from the config file. Loads the
    config file on first call.
    """
    if not hasattr(config, "config"):
        with open("config.json") as f:
            config.config = eval(f.read())
    node = config.config
    for part in attr.split("."):
        node = node[part]
    return node