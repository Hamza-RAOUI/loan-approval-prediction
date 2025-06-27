"""
Prediction module for loan approval prediction.
Handles single predictions and batch predictions.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')


class LoanPredictor:
    def __init__(self, model_path="models/best_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.model_name = None
        self.feature_names = None
        self.label_encoders = {}
        self.feature_info = self._get_feature_info()
        
    def _get_feature_info(self):
        """Define feature information for encoding and validation."""
        return {
            'Gender': {'type': 'categorical', 'values': ['Male', 'Female']},
            'Married': {'type': 'categorical', 'values': ['Yes', 'No']},
            'Dependents': {'type': 'categorical', 'values': ['0', '1', '2', '3+']},
            'Education': {'type': 'categorical', 'values': ['Graduate', 'Not Graduate']},
            'Self_Employed': {'type': 'categorical', 'values': ['Yes', 'No']},
            'ApplicantIncome': {'type': 'numerical', 'min': 0, 'max': 100000},
            'CoapplicantIncome': {'type': 'numerical', 'min': 0, 'max': 100000},
            'LoanAmount': {'type': 'numerical', 'min': 0, 'max': 1000},
            'Loan_Amount_Term': {'type': 'categorical', 'values': [360, 240, 180, 120, 480, 300]},
            'Credit_History': {'type': 'categorical', 'values': [1.0, 0.0]},
            'Property_Area': {'type': 'categorical', 'values': ['Rural', 'Urban', 'Semiurban']}
        }
    
    def load_model(self):
        """Load the trained model."""
        try:
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.model_name = model_data['model_name']
            self.feature_names = model_data['feature_names']
            print(f"Model ({self.model_name}) loaded successfully")
            return True
        except FileNotFoundError:
            print(f"Model file not found: {self.model_path}")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def _encode_categorical_value(self, feature, value):
        """Encode a categorical value."""
        if feature not in self.feature_info:
            return value
        
        feature_info = self.feature_info[feature]
        if feature_info['type'] == 'categorical':
            # Simple encoding for common categorical values
            encoding_maps = {
                'Gender': {'Male': 1, 'Female': 0},
                'Married': {'Yes': 1, 'No': 0},
                'Dependents': {'0': 0, '1': 1, '2': 2, '3+': 3},
                'Education': {'Graduate': 1, 'Not Graduate': 0},
                'Self_Employed': {'Yes': 1, 'No': 0},
                'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
                'Credit_History': {1.0: 1, 0.0: 0, '1.0': 1, '0.0': 0, 1: 1, 0: 0}
            }
            
            if feature in encoding_maps:
                return encoding_maps[feature].get(value, value)
        
        return value
    
    def preprocess_input(self, input_data):
        """Preprocess input data for prediction."""
        if isinstance(input_data, dict):
            # Single prediction
            processed_data = {}
            for feature, value in input_data.items():
                if feature in self.feature_info:
                    processed_data[feature] = self._encode_categorical_value(feature, value)
                else:
                    processed_data[feature] = value
            
            # Convert to DataFrame
            df = pd.DataFrame([processed_data])
        
        elif isinstance(input_data, pd.DataFrame):
            # Batch prediction
            df = input_data.copy()
            for feature in df.columns:
                if feature in self.feature_info:
                    df[feature] = df[feature].apply(
                        lambda x: self._encode_categorical_value(feature, x)
                    )
        
        else:
            raise ValueError("Input data must be a dictionary or pandas DataFrame")
        
        # Ensure all required features are present
        if self.feature_names:
            missing_features = set(self.feature_names) - set(df.columns)
            if missing_features:
                print(f"Warning: Missing features: {missing_features}")
                # Fill missing features with default values
                for feature in missing_features:
                    df[feature] = 0
            
            # Reorder columns to match training data
            df = df[self.feature_names]
        
        return df
    
    def predict_single(self, applicant_data):
        """Make prediction for a single applicant."""
        if self.model is None:
            if not self.load_model():
                return None
        
        try:
            # Preprocess input
            processed_data = self.preprocess_input(applicant_data)
            
            # Make prediction
            prediction = self.model.predict(processed_data)[0]
            
            # Get probability if available
            probability = None
            if hasattr(self.model, 'predict_proba'):
                prob_scores = self.model.predict_proba(processed_data)[0]
                probability = {
                    'approval_probability': prob_scores[1] if len(prob_scores) > 1 else prob_scores[0],
                    'rejection_probability': prob_scores[0] if len(prob_scores) > 1 else 1 - prob_scores[0]
                }
            
            # Convert prediction to readable format
            loan_status = 'Approved' if prediction == 1 or prediction == 'Y' else 'Rejected'
            
            result = {
                'prediction': loan_status,
                'prediction_code': prediction,
                'confidence': probability,
                'model_used': self.model_name
            }
            
            return result
        
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def predict_batch(self, data):
        """Make predictions for multiple applicants."""
        if self.model is None:
            if not self.load_model():
                return None
        
        try:
            # Preprocess input
            processed_data = self.preprocess_input(data)
            
            # Make predictions
            predictions = self.model.predict(processed_data)
            
            # Get probabilities if available
            probabilities = None
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(processed_data)
            
            # Convert predictions to readable format
            loan_statuses = ['Approved' if pred == 1 or pred == 'Y' else 'Rejected' 
                           for pred in predictions]
            
            results = pd.DataFrame({
                'Prediction': loan_statuses,
                'Prediction_Code': predictions
            })
            
            if probabilities is not None:
                results['Approval_Probability'] = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
                results['Rejection_Probability'] = probabilities[:, 0] if probabilities.shape[1] > 1 else 1 - probabilities[:, 0]
            
            return results
        
        except Exception as e:
            print(f"Error making batch predictions: {e}")
            return None
    
    def validate_input(self, input_data):
        """Validate input data."""
        errors = []
        
        for feature, value in input_data.items():
            if feature in self.feature_info:
                feature_info = self.feature_info[feature]
                
                if feature_info['type'] == 'categorical':
                    if value not in feature_info['values']:
                        errors.append(f"{feature}: '{value}' not in allowed values {feature_info['values']}")
                
                elif feature_info['type'] == 'numerical':
                    try:
                        num_value = float(value)
                        if num_value < feature_info['min'] or num_value > feature_info['max']:
                            errors.append(f"{feature}: {value} not in range [{feature_info['min']}, {feature_info['max']}]")
                    except ValueError:
                        errors.append(f"{feature}: '{value}' is not a valid number")
        
        return errors
    
    def get_feature_requirements(self):
        """Get information about required features and their valid values."""
        return self.feature_info
    
    def explain_prediction(self, applicant_data):
        """Provide explanation for the prediction (if model supports it)."""
        result = self.predict_single(applicant_data)
        
        if result is None:
            return None
        
        explanation = {
            'prediction': result['prediction'],
            'confidence': result.get('confidence', {}),
            'key_factors': []
        }
        
        # Add feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            
            for feature, importance in top_features:
                value = applicant_data.get(feature, 'N/A')
                explanation['key_factors'].append({
                    'feature': feature,
                    'value': value,
                    'importance': importance
                })
        
        return explanation


def create_sample_prediction():
    """Create a sample prediction for testing."""
    predictor = LoanPredictor()
    
    # Sample applicant data
    sample_applicant = {
        'Gender': 'Male',
        'Married': 'Yes',
        'Dependents': '1',
        'Education': 'Graduate',
        'Self_Employed': 'No',
        'ApplicantIncome': 5849,
        'CoapplicantIncome': 0,
        'LoanAmount': 146,
        'Loan_Amount_Term': 360,
        'Credit_History': 1.0,
        'Property_Area': 'Urban'
    }
    
    print("Sample Applicant Data:")
    for key, value in sample_applicant.items():
        print(f"  {key}: {value}")
    
    # Validate input
    errors = predictor.validate_input(sample_applicant)
    if errors:
        print(f"\nValidation Errors: {errors}")
        return
    
    # Make prediction
    result = predictor.predict_single(sample_applicant)
    
    if result:
        print(f"\nPrediction Result:")
        print(f"  Loan Status: {result['prediction']}")
        print(f"  Model Used: {result['model_used']}")
        
        if result['confidence']:
            print(f"  Approval Probability: {result['confidence']['approval_probability']:.2%}")
            print(f"  Rejection Probability: {result['confidence']['rejection_probability']:.2%}")
        
        # Get explanation
        explanation = predictor.explain_prediction(sample_applicant)
        if explanation and explanation['key_factors']:
            print(f"\nKey Factors:")
            for factor in explanation['key_factors']:
                print(f"  {factor['feature']}: {factor['value']} (importance: {factor['importance']:.3f})")
    
    else:
        print("Prediction failed")


if __name__ == "__main__":
    create_sample_prediction()
