# Reuters Text Classification

We use the [Reuters-21578 Text Categorization Collection](https://kdd.ics.uci.edu/databases/reuters21578/reuters21578.html) to perform multi-label text classification with Transformer-based models. Given a Reuters news articles, we need to classify it into one or more "topic."

The dataset is imbalanced, and contains data of variable length.

Distribution of Tokens (using `bert-base-cased`):
| Metric                 | Value                 |
|------------------------|-----------------------|
| Average | 121.95                |
| Standard deviation      | 109.25                |
| Maximum                | 2976                  |
| Minimum                | 4                     |

Distribution of Topics:
| Topic                         | Count |
|-------------------------------|-------|
| [earn]                        | 3687  |
| [acq]                         | 1994  |
| [crude]                       | 326   |
| [trade]                       | 307   |
| [money-fx]                    | 243   |
| ... | ... |
| [veg-oil, sun-oil, cotton-oil]| 1     |
| [veg-oil, coconut-oil]        | 1     |
| [l-cattle]                    | 1     |
| [livestock, carcass, hog]     | 1     |
| [ipi, gnp, grain]             | 1     |

## Results
### BERT
After finetuning the last two layers of a BERT-cased model + an added dense layer for 22 epochs (refer to `train_bert.py`), we have the following Validation Data result:
| Metric                      | Value                  |
|-----------------------------|------------------------|
| Exact-Match Accuracy   | 0.9065     |
| Hamming Loss     | 0.0026  |
| Micro-F1 | 0.9426     |
| Macro-F1 | 0.833     |

### XLNet
After finetuning the last two layers of a XLNet-cased model + an added dense layer for 12 epochs (refer to `train_xlnet.py`), we have the following Validation Data result:
| Metric                      | Value                  |
|-----------------------------|------------------------|
| Exact-Match Accuracy   | 0.923     |
| Hamming Loss     | 0.0019  |
| Micro-F1 | 0.956     |
| Macro-F1 | 0.905     |
