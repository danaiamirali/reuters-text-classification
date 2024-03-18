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


def load_data(file_path: str) -> list[dict]:
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
    data = []
    if file_path.endswith(".sgm"):
        try:
            with open(file_path, "r", encoding="utf8") as f:
                data.append(f.read())
        except:
            print(f"Error reading file: {file_path}")
            return []

    documents = []
    for doc in data:
        soup = BeautifulSoup(doc, "html.parser")
        # print(soup.prettify())
        # print(soup.find_all("reuters"))
        reuters_elements = soup.find_all("reuters")

        for element in reuters_elements:
            title = element.find("title")
            body = element.find("body")
            topics = element.find("topics")
            places = element.find("places")
            people = element.find("people")
            orgs = element.find("orgs")
            exchanges = element.find("exchanges")
            companies = element.find("companies")

            # Create dictionary for each document, and append to the list
            document = {
                "title": title.text if title else None,
                "body": preprocess_text(body.text) if body else None,
                "topics": topics.text if topics else None,
                "places": places.text if places else None,
                "people": people.text if people else None,
                "orgs": orgs.text if orgs else None,
                "exchanges": exchanges.text if exchanges else None,
                "companies": companies.text if companies else None
            }

            # print(document)

            documents.append(document)
        
        return documents

def load_all_data(directory: str) -> list[dict]:
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

    for files in os.listdir(directory):
        if files.endswith(".sgm"):
            file_path = os.path.join(directory, files)
            documents = load_data(file_path)
            docs.extend(documents)
        
    return docs

if __name__ == "__main__":
    # Run this file to test the loader
    file = "data/reut2-000.sgm"
    output_file = f"{file}-load.txt"

    # Test preprocessing
    # soup = BeautifulSoup(file, "html.parser")
    # sample_text = soup.find_all("reuters")[0].find("body").text
    # print(
    #     """
    #     Before preprocessing: 
    #     {text}
    #     After preprocessing:
    #     {preprocessed_text}
    #     """.format(
    #         text=sample_text, preprocessed_text=preprocess_text(sample_text)
    #     )
    # )

    # Test loading data
    print(f"Loading {file}...")
    with open(output_file, "w") as f:
        for doc in load_data(file):
            f.write(str(doc) + "\n---\n")
    