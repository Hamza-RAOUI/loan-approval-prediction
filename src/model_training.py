"""
Model training module for loan approval prediction.
Implements multiple classification algorithms and evaluates their performance.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib
import warnings
warnings.filterwarnings('ignore')


class LoanApprovalModel:
    def __init__(self):
        self.models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                criterion='entropy',
                random_state=42,
                max_depth=10
            ),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'SVM': SVC(kernel='rbf', random_state=42, probability=True),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
        }
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        self.results = {}
    
    def split_data(self, X, y, test_size=0.3, random_state=42):
        """Split data into training and testing sets."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """Train all models."""
        print("Training models...")
        trained_models = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            trained_models[name] = model
        
        print("All models trained successfully!")
        return trained_models
    
    def evaluate_models(self, trained_models, X_train, y_train, X_test, y_test):
        """Evaluate all trained models."""
        print("\n=== Model Evaluation Results ===")
        
        results = {}
        
        for name, model in trained_models.items():
            # Training predictions
            y_train_pred = model.predict(X_train)
            train_accuracy = accuracy_score(y_train, y_train_pred)
            
            # Test predictions
            y_test_pred = model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            
            # Additional metrics
            precision = precision_score(y_test, y_test_pred, average='weighted')
            recall = recall_score(y_test, y_test_pred, average='weighted')
            f1 = f1_score(y_test, y_test_pred, average='weighted')
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            results[name] = {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'model': model
            }
            
            print(f"\n{name} Results:")
            print(f"  Training Accuracy: {train_accuracy:.4f}")
            print(f"  Test Accuracy: {test_accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  CV Score: {cv_mean:.4f} (+/- {cv_std*2:.4f})")
        
        self.results = results
        return results
    
    def get_best_model(self):
        """Identify the best performing model based on test accuracy."""
        if not self.results:
            print("No results available. Train and evaluate models first.")
            return None
        
        best_name = max(self.results.keys(), 
                       key=lambda x: self.results[x]['test_accuracy'])
        self.best_model = self.results[best_name]['model']
        self.best_model_name = best_name
        
        print(f"\nBest Model: {best_name}")
        print(f"Test Accuracy: {self.results[best_name]['test_accuracy']:.4f}")
        
        return self.best_model, best_name
    
    def detailed_evaluation(self, X_test, y_test):
        """Provide detailed evaluation of the best model."""
        if self.best_model is None:
            print("No best model found. Run get_best_model() first.")
            return
        
        y_pred = self.best_model.predict(X_test)
        
        print(f"\n=== Detailed Evaluation: {self.best_model_name} ===")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Feature importance (if available)
        if hasattr(self.best_model, 'feature_importances_'):
            print("\nFeature Importance:")
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            print(feature_importance)
            
            return feature_importance
        
        return None
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning for the best models."""
        print("Performing hyperparameter tuning...")
        
        # Define parameter grids
        param_grids = {
            'RandomForest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'LogisticRegression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        }
        
        tuned_models = {}
        
        for model_name in ['RandomForest', 'LogisticRegression']:  # Focus on best performers
            if model_name in param_grids:
                print(f"Tuning {model_name}...")
                
                model = self.models[model_name]
                param_grid = param_grids[model_name]
                
                grid_search = GridSearchCV(
                    model, param_grid, cv=5, 
                    scoring='accuracy', n_jobs=-1
                )
                
                grid_search.fit(X_train, y_train)
                
                tuned_models[model_name] = {
                    'model': grid_search.best_estimator_,
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_
                }
                
                print(f"Best parameters for {model_name}: {grid_search.best_params_}")
                print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return tuned_models
    
    def save_model(self, model_path="../models/best_model.pkl"):
        """Save the best model to disk."""
        if self.best_model is None:
            print("No best model to save. Train models first.")
            return
        
        # Create models directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'feature_names': self.feature_names,
            'results': self.results
        }
        
        joblib.dump(model_data, model_path)
        print(f"Best model ({self.best_model_name}) saved to {model_path}")
    
    def load_model(self, model_path="../models/best_model.pkl"):
        """Load a saved model from disk."""
        try:
            model_data = joblib.load(model_path)
            self.best_model = model_data['model']
            self.best_model_name = model_data['model_name']
            self.feature_names = model_data['feature_names']
            self.results = model_data.get('results', {})
            
            print(f"Model ({self.best_model_name}) loaded successfully from {model_path}")
            return True
        except FileNotFoundError:
            print(f"Model file not found: {model_path}")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, X):
        """Make predictions using the best model."""
        if self.best_model is None:
            print("No model available for prediction. Train or load a model first.")
            return None
        
        predictions = self.best_model.predict(X)
        probabilities = None
        
        if hasattr(self.best_model, 'predict_proba'):
            probabilities = self.best_model.predict_proba(X)
        
        return predictions, probabilities
    
    def predict_single(self, features):
        """Make prediction for a single instance."""
        if self.best_model is None:
            print("No model available for prediction.")
            return None, None
        
        # Ensure features is a 2D array
        if len(features) == len(self.feature_names):
            features = np.array(features).reshape(1, -1)
        
        prediction = self.best_model.predict(features)[0]
        
        if hasattr(self.best_model, 'predict_proba'):
            probability = self.best_model.predict_proba(features)[0]
            confidence = max(probability)
        else:
            confidence = None
        
        return prediction, confidence
    
    def train_and_evaluate_pipeline(self, X, y):
        """Complete training and evaluation pipeline."""
        print("Starting complete training and evaluation pipeline...")
        
        # Store feature names
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Train models
        trained_models = self.train_models(X_train, y_train)
        
        # Evaluate models
        results = self.evaluate_models(trained_models, X_train, y_train, X_test, y_test)
        
        # Get best model
        best_model, best_name = self.get_best_model()
        
        # Detailed evaluation
        feature_importance = self.detailed_evaluation(X_test, y_test)
        
        # Save model
        self.save_model("models/best_model.pkl")
        
        print("\nPipeline completed successfully!")
        
        return {
            'results': results,
            'best_model': best_model,
            'best_model_name': best_name,
            'feature_importance': feature_importance,
            'test_data': (X_test, y_test)
        }


if __name__ == "__main__":
    # Test the model training
    from data_preprocessing import DataPreprocessor
    
    # Initialize preprocessor and model
    preprocessor = DataPreprocessor()
    model_trainer = LoanApprovalModel()
    
    # Load and preprocess data
    try:
        X, y, data = preprocessor.preprocess_pipeline("../data/LoanApprovalPrediction.csv")
    except:
        print("Creating sample data for testing...")
        from data_preprocessing import create_sample_data
        sample_data = create_sample_data()
        sample_data.to_csv("../data/LoanApprovalPrediction.csv", index=False)
        X, y, data = preprocessor.preprocess_pipeline("../data/LoanApprovalPrediction.csv")
    
    # Train and evaluate models
    if X is not None and y is not None:
        results = model_trainer.train_and_evaluate_pipeline(X, y)
        print("Model training completed!")
