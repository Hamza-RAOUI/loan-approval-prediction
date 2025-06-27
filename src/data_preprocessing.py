"""
Data preprocessing module for loan approval prediction.
Handles data loading, cleaning, encoding, and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    def __init__(self):
        self.label_encoder = preprocessing.LabelEncoder()
        self.encoded_columns = []
        
    def load_data(self, file_path):
        """Load the dataset from CSV file."""
        try:
            data = pd.read_csv(file_path)
            print(f"Dataset loaded successfully. Shape: {data.shape}")
            return data
        except FileNotFoundError:
            print(f"Error: File {file_path} not found.")
            return None
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def explore_data(self, data):
        """Basic data exploration."""
        print("\n=== Data Overview ===")
        print(f"Dataset shape: {data.shape}")
        print(f"\nFirst 5 rows:")
        print(data.head())
        
        print(f"\nData types:")
        print(data.dtypes)
        
        print(f"\nMissing values:")
        print(data.isnull().sum())
        
        # Count categorical variables
        obj = (data.dtypes == 'object')
        print(f"\nCategorical variables: {len(list(obj[obj].index))}")
        
        return data
    
    def clean_data(self, data):
        """Clean the dataset by removing unnecessary columns and handling missing values."""
        # Drop Loan_ID as it's not useful for prediction
        if 'Loan_ID' in data.columns:
            data = data.drop(['Loan_ID'], axis=1)
            print("Dropped Loan_ID column")
        
        # Handle missing values by filling with mean for numeric columns
        for col in data.columns:
            if data[col].dtype in ['int64', 'float64']:
                data[col] = data[col].fillna(data[col].mean())
            else:
                # For categorical columns, fill with mode
                data[col] = data[col].fillna(data[col].mode()[0])
        
        print("Missing values handled")
        return data
    
    def encode_categorical_features(self, data):
        """Encode categorical features using Label Encoder."""
        obj = (data.dtypes == 'object')
        object_cols = list(obj[obj].index)
        
        for col in object_cols:
            data[col] = self.label_encoder.fit_transform(data[col].astype(str))
            self.encoded_columns.append(col)
        
        print(f"Encoded categorical columns: {object_cols}")
        
        # Verify no object columns remain
        obj_after = (data.dtypes == 'object')
        print(f"Categorical variables after encoding: {len(list(obj_after[obj_after].index))}")
        
        return data
    
    def prepare_features_target(self, data, target_column='Loan_Status'):
        """Separate features and target variable."""
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        X = data.drop([target_column], axis=1)
        y = data[target_column]
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Feature columns: {list(X.columns)}")
        
        return X, y
    
    def preprocess_pipeline(self, file_path, target_column='Loan_Status'):
        """Complete preprocessing pipeline."""
        print("Starting data preprocessing pipeline...")
        
        # Load data
        data = self.load_data(file_path)
        if data is None:
            return None, None
        
        # Explore data
        data = self.explore_data(data)
        
        # Clean data
        data = self.clean_data(data)
        
        # Encode categorical features
        data = self.encode_categorical_features(data)
        
        # Prepare features and target
        X, y = self.prepare_features_target(data, target_column)
        
        print("Data preprocessing completed successfully!")
        return X, y, data


def create_sample_data():
    """Create a sample dataset for testing when actual data is not available."""
    np.random.seed(42)
    n_samples = 600
    
    data = {
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Married': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.choice(['0', '1', '2', '3+'], n_samples),
        'Education': np.random.choice(['Graduate', 'Not Graduate'], n_samples),
        'Self_Employed': np.random.choice(['Yes', 'No'], n_samples),
        'ApplicantIncome': np.random.normal(5000, 2000, n_samples).astype(int),
        'CoapplicantIncome': np.random.normal(2000, 1500, n_samples).astype(int),
        'LoanAmount': np.random.normal(150, 50, n_samples).astype(int),
        'Loan_Amount_Term': np.random.choice([360, 240, 180, 120], n_samples),
        'Credit_History': np.random.choice([1.0, 0.0], n_samples, p=[0.8, 0.2]),
        'Property_Area': np.random.choice(['Rural', 'Urban', 'Semiurban'], n_samples),
    }
    
    # Create target based on some logic
    loan_status = []
    for i in range(n_samples):
        score = 0
        if data['Credit_History'][i] == 1.0:
            score += 3
        if data['Education'][i] == 'Graduate':
            score += 1
        if data['ApplicantIncome'][i] > 4000:
            score += 1
        if data['Married'][i] == 'Yes':
            score += 1
        
        # Add some randomness
        score += np.random.choice([-1, 0, 1])
        
        loan_status.append('Y' if score >= 3 else 'N')
    
    data['Loan_Status'] = loan_status
    
    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = DataPreprocessor()
    
    # Try to load real data, if not available create sample data
    try:
        X, y, processed_data = preprocessor.preprocess_pipeline("../data/LoanApprovalPrediction.csv")
    except:
        print("Creating sample data for testing...")
        sample_data = create_sample_data()
        sample_data.to_csv("../data/LoanApprovalPrediction.csv", index=False)
        X, y, processed_data = preprocessor.preprocess_pipeline("../data/LoanApprovalPrediction.csv")
