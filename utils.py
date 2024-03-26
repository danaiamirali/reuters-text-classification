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

nltk.download("stopwords")

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