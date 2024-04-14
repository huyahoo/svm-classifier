"""_summary_: This script is used to classify the dataset by SVM.
Author: 
- Quang Huy Phung
- Dinh Minh Nguyen 
- Luong Phuong Truc Huynh
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
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

from log import Logger
from preprocessing import Preprocessor

class SVMClassifier:
    def __init__(self, args):
        super(SVMClassifier, self).__init__()
        
        self.data_path = args.data
        self.output_dir = args.output
        self.target_feature = args.target
        self.test_size = args.test_size
        self.scaler_type = args.scaler
        self.data = pd.read_csv(self.data_path)
        self.scaler = self.init_scaler(self.scaler_type)
        self.X_train, self.X_test, self.y_train, self.y_test = self.preprocessing_data()
        
        self.logger = Logger(self.output_dir)

    @staticmethod
    def label_encode_categorical(data):
        """
        Converts categorical data into numerical form using LabelEncoder. 
        This is necessary because machine learning algorithms typically work with numerical data.

        Args:
            data (pd.DataFrame): The data to be encoded.

        Returns:
            pd.DataFrame: The data with categorical variables encoded as numerical values.
        """
        le = LabelEncoder()
        
        for col in data.columns:
            if data[col].dtypes=='object':
                data[col]=le.fit_transform(data[col])
        return data

    @staticmethod
    def init_scaler(scaler):
        """
        Initializes the scaler based on the provided type. 

        Args:
            scaler (str): The type of scaler to use. Options are 'standard', 'maxmin', and 'robust'.

        Returns:
            Scaler: The initialized scaler. Default is StandardScaler if the provided type is not recognized.
        """
        scaler_map = {
            'standard': StandardScaler(),
            'maxmin': MinMaxScaler(),
            'robust': RobustScaler(),
        }
        return scaler_map.get(scaler, StandardScaler())
    
    @staticmethod
    def tune_hyperparameter():
        """
        Tunes the hyperparameters of the SVM using GridSearchCV. 

        Returns:
            GridSearchCV: The GridSearchCV object after fitting. This object can be used to access the best parameters found.
        """
        param_grid = {
            'C': np.linspace(2 ** -5, 2 ** 15, 4),
            'kernel': ['rbf'],
            'gamma': np.linspace(2 ** -15, 2 ** 3, 4)
        }
        model_clf = svm.SVC()
        grid = GridSearchCV(model_clf, param_grid, refit = True, verbose = 3)
        return grid
    
    def preprocessing_data(self):
        """
        Pre-processes the data by removing duplicates, encoding categorical variables, handling missing values, 
        splitting the data into training and test sets, and scaling the features.

        Returns:
            tuple: A tuple containing the training and test sets for the features (X) and the target variable (y).
        """
        # Remove duplicates
        data = self.data.drop_duplicates()
        
        # Convert categorical data into numerical form
        data = self.label_encode_categorical(data)
        
        # Handle missing values
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        data = pd.DataFrame(imp.fit_transform(data), columns = data.columns)
        
        X = data.drop(self.target_feature, axis=1)
        y = data[self.target_feature]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42)
        
        # Normalize the data
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test

    def train(self, model):
        """
        Trains the provided model using the training data.

        Args:
            model (sklearn estimator): The machine learning model to be trained.

        Returns:
            sklearn estimator: The trained model.
        """
        model.fit(self.X_train, self.y_train)
        return model

    def evaluate(self, model, y_pred):
        """
        Evaluates the provided model by calculating its accuracy on the test set.

        Args:
            model (sklearn estimator): The machine learning model to be evaluated.

        Returns:
            float: The accuracy of the model on the test set.
        """
        print("Best parameters found: ", model.best_params_, "\n")
        
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Accuracy: {accuracy}", "\n")
        
        clf_report = classification_report(self.y_test, y_pred, output_dict=True)
        clf_report_df = pd.DataFrame(clf_report).transpose()
        print(f"Classification report: \n{clf_report_df}", "\n")
        
        conf_mat = confusion_matrix(self.y_test, y_pred,)
        tn, fp, fn, tp = conf_mat.ravel()
        
        print("Confusion matrix:")
        print(f"| TN | FP |   | {tn:2d} | {fp:2d} |")
        print(f"|----|----| = |----|----|")
        print(f"| FN | TP |   | {fn:2d} | {tp:2d} |")
        
        self.logger.log(model.best_params_, accuracy, conf_mat, clf_report_df)
        
        return accuracy, clf_report, conf_mat

    def predict(self, model):
        """
        Makes predictions on the test set using the provided model.

        Args:
            model (sklearn estimator): The machine learning model to make predictions with.

        Returns:
            np.array: The predictions made by the model on the test set.
        """
        return model.predict(self.X_test)

    def save_results(self, pred_results):
        """
        Saves the provided results to a CSV file in the specified output directory.

        Args:
            results (pd.DataFrame): The results to be saved. This should be a DataFrame where each row represents a result.
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        pred_results = pd.DataFrame(pred_results, columns=[self.target_feature])
        results = pd.concat([pd.DataFrame(self.data.drop(self.target_feature, axis=1), columns=self.data.columns[:-1]), pred_results], axis=1)
        
        results.to_csv(os.path.join(self.output_dir, 'results.csv'), index=False)

    def run(self):
        grid = self.tune_hyperparameter()
        model = self.train(grid)
        
        best_model = model.best_estimator_
        best_model = self.train(best_model)
        
        y_pred = self.predict(best_model)
        
        self.evaluate(model, y_pred)
        
        self.save_results(y_pred)
        

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