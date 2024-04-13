"""_summary_: This script is used to classify the dataset by SVM.
Author: Quang Huy Phung, Dinh Minh Nguyen, Luong Phuong Truc Huynh
Date: 2024-04-13
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

class SVMClassifier:
    def __init__(self, args):
        super(SVMClassifier, self).__init__()
        
        self.data_path = args.data
        self.output_dir = args.output
        self.target_value = args.target
        self.test_size = args.test_size
        self.scaler_type = args.scaler
        self.data = pd.read_csv(self.data_path)
        self.scaler = self.init_scaler(self.scaler_type)
        self.X_train, self.X_test, self.y_train, self.y_test = self.preprocessing_data()

    def label_encode_categorical(self, data):
        le = LabelEncoder()
        for col in data.columns:
            if data[col].dtypes=='object':
                data[col]=le.fit_transform(data[col])
        return data

    def preprocessing_data(self):
        data = self.data.drop_duplicates()
        data = self.label_encode_categorical(data)
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        data = pd.DataFrame(imp.fit_transform(data), columns = data.columns)
        X = data.drop(self.target_value, axis=1)
        y = data[self.target_value]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        return X_train, X_test, y_train, y_test

    def init_scaler(self, scaler):
        scaler_map = {
            'standard': StandardScaler(),
            'maxmin': MinMaxScaler(),
            'robust': RobustScaler(),
        }
        return scaler_map.get(scaler, StandardScaler())

    def tune_hyperparameter(self):
        param_grid = {
            'C': np.linspace(2 ** -5, 2 ** 15, 4),
            'kernel': ['rbf'],
            'gamma': np.linspace(2 ** -15, 2 ** 3, 4)
        }
        model_clf = svm.SVC()
        grid = GridSearchCV(model_clf, param_grid, refit = True, verbose = 3)
        return grid

    def train(self, model):
        model.fit(self.X_train, self.y_train)
        return model

    def evaluate(self, model):
        accuracy = model.score(self.X_test, self.y_test)
        return accuracy

    def predict(self, model):
        return model.predict(self.X_test)

    def save_results(self, results):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        results.to_csv(os.path.join(self.output_dir, 'results.csv'), index=False)

    def run(self):
        grid = self.tune_hyperparameter()
        model = self.train(grid)
        print("Best parameters found: ", model.best_params_)
        
        best_model = model.best_estimator_
        
        best_model = self.train(best_model)
        
        accuracy = self.evaluate(best_model)
        print(f"Accuracy: {accuracy}")
        
        y_pred = self.predict(best_model)
        
        y_pred = pd.DataFrame(y_pred, columns=[self.target_value])
        results = pd.concat([pd.DataFrame(self.data.drop(self.target_value, axis=1), columns=self.data.columns[:-1]), y_pred], axis=1)
        self.save_results(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify by SVM')
    parser.add_argument('--data', type=str, default='./data/credit_card_approvals.csv',
                        help='Path to the dataset.')
    parser.add_argument('--output', type=str, default='./output',
                        help='Path to the output directory.')
    parser.add_argument('--target', type=str, default='Approved',
                        help='Target value to classify.')
    parser.add_argument('--test_size', type=float, default=0.25,
                        help='Size of the test set.')
    parser.add_argument('--scaler', type=str, default='standard',
                        help='Scaler for the features. Options: "standard", "maxmin", "robust".')
    args = parser.parse_args()
    
    classifier = SVMClassifier(args)
    
    classifier.run()