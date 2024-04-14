import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

class Preprocess():
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.continuous_variables = ['Age', 'Debt', 'YearsEmployed', 'CreditScore', 'Income']
        self.scaler = StandardScaler()

    def rename_columns(self, col_names):
        self.df.columns = col_names

    def calculate_missing_values(self):
        missing_values = []
        for col in self.df.columns:
            missing_count = (self.df[col] == '?').sum()
            missing_percentage = np.round(100 * missing_count / len(self.df), 2)
            missing_values.append({'Feature': col, 'NumberMissing': missing_count, 'PercentageMissing': missing_percentage})
        mv_df = pd.DataFrame(missing_values)
        return mv_df

    def replace_missing_values(self):
        self.df.replace('?', np.nan, inplace=True)

    def fill_missing_values(self):
        median_age = self.df['Age'].median()
        self.df['Age'].fillna(median_age, inplace=True)
        self.df['Age'] = self.df['Age'].astype(float)

    def normalize(self):
        norm_variables = [f'{var}Norm' for var in self.continuous_variables]
        self.df[norm_variables] = self.scaler.fit_transform(self.df[self.continuous_variables])

    def log_transform(self):
        for var in self.continuous_variables:
            self.df[f'{var}Log'] = np.log(self.df[var] + 1)  # Adding 1 to avoid logarithm of zero or negative values
        log_transformed_variables = [f'{var}Log' for var in self.continuous_variables]
        self.df[log_transformed_variables] = self.scaler.fit_transform(self.df[log_transformed_variables])

    def handle_categorical_variables(self):
        # Add your code here to handle categorical variables

    def save_to_csv(self, file_path):
        self.df.to_csv(file_path, index=False)

    def split_data(self, test_size, random_state):
        train_df, test_df = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return train_df, test_df
    
def run():
    preprocessor = Preprocess('./data/raw_credit_card_approvals.csv')
    preprocessor.rename_columns(['Gender', 'Age', 'Debt', 'Married', 'BankCustomer', 'Industry', 'Ethnicity', 'YearsEmployed', 'PriorDefault', 'Employed', 'CreditScore', 'DriversLicense', 'Citizen', 'ZipCode', 'Income', 'Approved'])
    preprocessor.calculate_missing_values()
    preprocessor.replace_missing_values()
    preprocessor.fill_missing_values()
    preprocessor.normalize()
    preprocessor.log_transform()
    preprocessor.handle_categorical_variables()
    preprocessor.save_to_csv('clean_data.csv')
    train_df, test_df = preprocessor.split_data(test_size=0.2, random_state=42)
    train_df.to_csv('train_set.csv', index=False)
    test_df.to_csv('test_set.csv', index=False)
        
if __name__ == '__main__':
    run()