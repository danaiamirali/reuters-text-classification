from transformers import AutoTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import tensorflow as tf
import regex as re

"""
This file is used to train a BERT model on the Reuters dataset.

The model will perform multi-label classification, 
predicting the topics of each document.

"""

def preprocess(data: str) -> tf.Tensor:
    if not isinstance(data, str):
        print(type(data))
    # we will keep casing
    # replace numbers
    
    data = re.sub(r"\d+", "<NUM>", data)
    # tokenize
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    data = tokenizer.encode(data, return_tensors="tf", padding=True, truncation=True, max_length=512)
    return data

def thresholded_accuracy(y_true, y_pred, threshold=0.5):
    # convert predictions to binary
    y_pred = tf.cast(y_pred > threshold, tf.int32)
    # calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.reduce_all(tf.equal(y_true, y_pred), axis=1), tf.float32))
    return accuracy

def macro_f1(y_true, y_pred):
    y_true = y_true.numpy()
    y_pred = tf.round(y_pred).numpy()

    return f1_score(y_true, y_pred, average="macro")

def train_model(X: list, y: list, verbose: bool = True) -> TFBertForSequenceClassification:
    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)   

    model = TFBertForSequenceClassification.from_pretrained("bert-base-cased")

    model.compile(optimizer= tf.keras.optimizers.Adam(3e-5) , loss="binary_crossentropy", metrics=[thresholded_accuracy, macro_f1])

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=32)

    y_pred = model.predict(X_test)

    if verbose:
        print("Evaluating model...")   
        print(f"Thresholded accuracy: {thresholded_accuracy(y_test, y_pred)}")
        print(f"Macro F1: {macro_f1(y_test, y_pred)}")
        print("--------------------------")

    return model
