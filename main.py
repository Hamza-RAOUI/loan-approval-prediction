"""
Main execution script for loan approval prediction project.
Orchestrates the complete machine learning pipeline.
"""

import os
import sys
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import DataPreprocessor, create_sample_data
from src.visualization import DataVisualizer
from src.model_training import LoanApprovalModel
from src.prediction import LoanPredictor


def check_data_availability():
    """Check if dataset is available, create sample if not."""
    data_path = "data/LoanApprovalPrediction.csv"
    
    if not os.path.exists(data_path):
        print("Dataset not found. Creating sample data...")
        os.makedirs("data", exist_ok=True)
        
        # Create sample data
        sample_data = create_sample_data()
        sample_data.to_csv(data_path, index=False)
        print(f"Sample dataset created at {data_path}")
        print(f"Dataset shape: {sample_data.shape}")
        
        return data_path
    else:
        print(f"Dataset found at {data_path}")
        return data_path


def run_data_preprocessing(data_path):
    """Run data preprocessing pipeline."""
    print("\n" + "="*50)
    print("STEP 1: DATA PREPROCESSING")
    print("="*50)
    
    preprocessor = DataPreprocessor()
    X, y, processed_data = preprocessor.preprocess_pipeline(data_path)
    
    if X is None or y is None:
        print("Error: Data preprocessing failed")
        return None, None, None
    
    print(f"Preprocessing completed successfully!")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    return X, y, processed_data


def run_data_visualization(processed_data):
    """Run data visualization."""
    print("\n" + "="*50)
    print("STEP 2: DATA VISUALIZATION")
    print("="*50)
    
    try:
        visualizer = DataVisualizer()
        visualizer.create_comprehensive_report(processed_data)
        print("Visualization completed successfully!")
    except Exception as e:
        print(f"Visualization error: {e}")
        print("Continuing with model training...")


def run_model_training(X, y):
    """Run model training and evaluation."""
    print("\n" + "="*50)
    print("STEP 3: MODEL TRAINING AND EVALUATION")
    print("="*50)
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    model_trainer = LoanApprovalModel()
    results = model_trainer.train_and_evaluate_pipeline(X, y)
    
    if results:
        print(f"\nBest Model: {results['best_model_name']}")
        best_accuracy = results['results'][results['best_model_name']]['test_accuracy']
        print(f"Best Test Accuracy: {best_accuracy:.4f}")
        
        # Print summary of all models
        print(f"\nModel Performance Summary:")
        print("-" * 60)
        print(f"{'Model':<20} {'Train Acc':<12} {'Test Acc':<12} {'CV Score':<12}")
        print("-" * 60)
        
        for name, result in results['results'].items():
            print(f"{name:<20} {result['train_accuracy']:<12.4f} "
                  f"{result['test_accuracy']:<12.4f} {result['cv_mean']:<12.4f}")
        
        print("-" * 60)
        
        return results
    else:
        print("Error: Model training failed")
        return None


def run_sample_predictions():
    """Run sample predictions to test the system."""
    print("\n" + "="*50)
    print("STEP 4: SAMPLE PREDICTIONS")
    print("="*50)
    
    predictor = LoanPredictor()
    
    # Sample test cases
    test_cases = [
        {
            'name': 'High Income Graduate',
            'data': {
                'Gender': 'Male',
                'Married': 'Yes',
                'Dependents': '0',
                'Education': 'Graduate',
                'Self_Employed': 'No',
                'ApplicantIncome': 8000,
                'CoapplicantIncome': 2000,
                'LoanAmount': 200,
                'Loan_Amount_Term': 360,
                'Credit_History': 1.0,
                'Property_Area': 'Urban'
            }
        },
        {
            'name': 'Low Income, Poor Credit',
            'data': {
                'Gender': 'Female',
                'Married': 'No',
                'Dependents': '2',
                'Education': 'Not Graduate',
                'Self_Employed': 'Yes',
                'ApplicantIncome': 2500,
                'CoapplicantIncome': 0,
                'LoanAmount': 150,
                'Loan_Amount_Term': 360,
                'Credit_History': 0.0,
                'Property_Area': 'Rural'
            }
        },
        {
            'name': 'Medium Profile',
            'data': {
                'Gender': 'Male',
                'Married': 'Yes',
                'Dependents': '1',
                'Education': 'Graduate',
                'Self_Employed': 'No',
                'ApplicantIncome': 5000,
                'CoapplicantIncome': 1500,
                'LoanAmount': 120,
                'Loan_Amount_Term': 360,
                'Credit_History': 1.0,
                'Property_Area': 'Semiurban'
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['name']}")
        print("-" * 30)
        
        # Validate input
        errors = predictor.validate_input(test_case['data'])
        if errors:
            print(f"Validation Errors: {errors}")
            continue
        
        # Make prediction
        result = predictor.predict_single(test_case['data'])
        
        if result:
            print(f"Prediction: {result['prediction']}")
            
            if result['confidence']:
                approval_prob = result['confidence']['approval_probability']
                print(f"Approval Probability: {approval_prob:.2%}")
                
                # Determine confidence level
                if approval_prob > 0.8:
                    confidence_level = "Very High"
                elif approval_prob > 0.6:
                    confidence_level = "High"
                elif approval_prob > 0.4:
                    confidence_level = "Medium"
                else:
                    confidence_level = "Low"
                
                print(f"Confidence Level: {confidence_level}")
        else:
            print("Prediction failed")


def print_project_summary():
    """Print project summary and next steps."""
    print("\n" + "="*50)
    print("PROJECT SUMMARY")
    print("="*50)
    
    print("""
This loan approval prediction project includes:

✅ Data Preprocessing Pipeline
   - Automatic data loading and cleaning
   - Categorical encoding
   - Missing value handling

✅ Data Visualization
   - Correlation heatmaps
   - Feature distributions
   - Target analysis

✅ Machine Learning Models
   - Random Forest Classifier
   - Logistic Regression
   - K-Nearest Neighbors
   - Support Vector Machine

✅ Model Evaluation
   - Train/Test accuracy
   - Cross-validation scores
   - Precision, Recall, F1-score
   - Feature importance analysis

✅ Prediction System
   - Single prediction capability
   - Batch prediction support
   - Input validation
   - Confidence scores

NEXT STEPS:
1. Run the Gradio web interface: python app.py
2. Replace sample data with real dataset
3. Experiment with hyperparameter tuning
4. Try ensemble methods for better performance

FILES CREATED:
- README.md - Project documentation
- requirements.txt - Dependencies
- src/data_preprocessing.py - Data preprocessing
- src/visualization.py - Data visualization
- src/model_training.py - Model training
- src/prediction.py - Prediction system
- main.py - Main execution script
- app.py - Gradio web interface (to be created)
""")


def main():
    """Main execution function."""
    print("LOAN APPROVAL PREDICTION PROJECT")
    print("=" * 50)
    
    try:
        # Step 1: Check data availability
        data_path = check_data_availability()
        
        # Step 2: Data preprocessing
        X, y, processed_data = run_data_preprocessing(data_path)
        if X is None:
            return
        
        # Step 3: Data visualization
        run_data_visualization(processed_data)
        
        # Step 4: Model training
        results = run_model_training(X, y)
        if results is None:
            return
        
        # Step 5: Sample predictions
        run_sample_predictions()
        
        # Step 6: Project summary
        print_project_summary()
        
        print("\n🎉 Project execution completed successfully!")
        print("You can now run 'python app.py' to start the web interface.")
        
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user.")
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
