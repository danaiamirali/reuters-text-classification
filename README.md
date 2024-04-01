# Reuters Text Classification

We use the [Reuters-21578 Text Categorization Collection](https://kdd.ics.uci.edu/databases/reuters21578/reuters21578.html) to perform multi-label text classification with Transformer-based models.

## Results
### Finetuning BERT
After finetuning the last two layers of a BERT-cased model + an added dense layer for 22 epochs (refer to `train_bert.py`), we have the following Validation Data result:
| Metric                      | Value                  |
|-----------------------------|------------------------|
| Exact-Match Accuracy   | 0.9065     |
| Hamming Loss     | 0.0026  |
| Micro-F1 | 0.9426     |
| Macro-F1 | 0.833     |
