# SVM Classifier for Credit Card Approvals

This script uses a Support Vector Machine (SVM) to classify credit card approval decisions based on various applicant attributes.

## Run with Google Colab
You can run this notebook in Google Colab by clicking the button below:

- Preprocessing
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14psva_4VoKTRoI5dSur4BYwlbiToNfOx?usp=sharing)

- Credit Card Approvals
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14psva_4VoKTRoI5dSur4BYwlbiToNfOx?usp=sharing)

## Arguments

- `--data`: Path to the dataset. Default is './data/credit_card_approvals.csv'.
- `--output`: Path to the output directory. Default is './output'.
- `--target`: Target value to classify. Default is 'Approved'.
- `--test_size`: Size of the test set. Default is 0.25.
- `--scaler`: Scaler for the features. Options are "standard", "maxmin", "robust". Default is 'standard'.


## Usage

```bash
python3 svm_classifier.py --data <data> --output <output> --target <target> --test_size <test_size> --scaler <scaler>
```

## Example
```bash
python3 svm_classifier.py
```
or
```bash
python3 svm_classifier.py --data './data/credit_card_approvals.csv' --output './output' --target 'Approved' --test_size 0.25 --scaler 'standard'

```