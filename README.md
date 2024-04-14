# SVM Classifier for Credit Card Approvals

This script uses a Support Vector Machine (SVM) to classify credit card approval decisions based on various applicant attributes.

The output folder contains:
- [Classification Report](output/classification_report.csv): Summary records of each run
- [Model Performance](output/model_performance.csv): Summary best params, accuracy and confusion matrix of each run
- [Predicted Results](output/y_red.csv): Show approval status of the dataset

## Environment
- Python 3.9.6

Install require library:
```bash
pip3 install -r requirements.txt
```

Quick run:
```bash
./classify.sh

# Execute line below to get permissions to the script before running:
chmod +x classify.sh
```

## Run with Google Colab
You can run this notebook in Google Colab by clicking the button below:

- Preprocessing
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1vvBdxLtQdWCcyZddEPGAvbH1OgIX3aDp/view?usp=sharing)

- Credit Card Approvals
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/19tSCIFZSYzyve8tXRX6XcPqsPzDtIpoU/view?usp=sharing)

## SVM Classifier

### Arguments

- `--preprocess`: Specify if the dataset whether it is preprocessed. Default is `False`.
- `--data`: Path to the dataset. Default is './data/credit_card_approvals.csv'.
- `--output`: Path to the output directory. Default is './output'.
- `--target`: Target value to classify. Default is 'Approved'.
- `--test_size`: Size of the test set. Default is 0.25.
- `--scaler`: Scaler for the features. Options are "standard", "maxmin", "robust". Default is 'standard'.

### Usage

```bash
python3 svm_classifier.py --preprocess --data <data> --output <output> --target <target> --test_size <test_size> --scaler <scaler>
```

### Example
```bash
python3 svm_classifier.py
```
or
```bash
python3 svm_classifier.py --preprocess --data './data/raw_credit_card_approvals.csv' --output './output' --target 'Approved' --test_size 0.25 --scaler 'standard'
```

## Preprocessing
This script preprocesses the raw credit card approvals dataset to prepare it for the SVM classifier.
This can be done individually as below and the output of preprocessed data is saved as `preprocessed_credit_card_approvals.csv` inside folder `data` as default.

### Arguments

- `--data`: Path to the raw dataset. Default is './data/raw_credit_card_approvals.csv'.
- `--output`: Path to the output directory where the preprocessed data will be saved. Default is './data'.
- `--target`: Target value to classify. Default is 'Approved'.
- `--test_size`: Size of the test set. Default is 0.25.
- `--scaler`: Scaler for the features. Options are "standard", "maxmin", "robust". Default is 'standard'.

### Usage

```bash
python3 preprocess.py --data <data> --output <output> --target <target> --test_size <test_size> --scaler <scaler>
```

### Example

```bash
python3 preprocess.py
```
or
```bash
python3 preprocess.py --data './data/raw_credit_card_approvals.csv' --output './data' --target 'Approved' --test_size 0.25 --scaler 'standard'
```