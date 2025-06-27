"""
Gradio web interface for loan approval prediction.
Provides an interactive interface for users to input applicant data and get predictions.
"""

import gradio as gr
import pandas as pd
import numpy as np
import os
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.prediction import LoanPredictor
from src.data_preprocessing import create_sample_data


class LoanApprovalApp:
    def __init__(self):
        self.predictor = LoanPredictor()
        self.setup_model()
        
    def setup_model(self):
        """Setup the prediction model."""
        # Check if model exists, if not train it
        if not os.path.exists("models/best_model.pkl"):
            print("Model not found. Training model...")
            self.train_model()
        
        # Load the model
        if not self.predictor.load_model():
            print("Failed to load model. Training new model...")
            self.train_model()
            self.predictor.load_model()
    
    def train_model(self):
        """Train the model if it doesn't exist."""
        from src.data_preprocessing import DataPreprocessor
        from src.model_training import LoanApprovalModel
        
        # Create data if it doesn't exist
        data_path = "data/LoanApprovalPrediction.csv"
        if not os.path.exists(data_path):
            os.makedirs("data", exist_ok=True)
            sample_data = create_sample_data()
            sample_data.to_csv(data_path, index=False)
        
        # Train model
        preprocessor = DataPreprocessor()
        X, y, _ = preprocessor.preprocess_pipeline(data_path)
        
        if X is not None and y is not None:
            model_trainer = LoanApprovalModel()
            model_trainer.train_and_evaluate_pipeline(X, y)
            print("Model training completed!")
    
    def predict_loan_approval(self, gender, married, dependents, education, self_employed, 
                            applicant_income, coapplicant_income, loan_amount, 
                            loan_amount_term, credit_history, property_area):
        """Make loan approval prediction."""
        
        # Prepare input data
        applicant_data = {
            'Gender': gender,
            'Married': married,
            'Dependents': dependents,
            'Education': education,
            'Self_Employed': self_employed,
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_amount_term,
            'Credit_History': credit_history,
            'Property_Area': property_area
        }
        
        # Validate input
        errors = self.predictor.validate_input(applicant_data)
        if errors:
            return f"❌ Input Validation Errors:\n" + "\n".join(errors), "", ""
        
        # Make prediction
        result = self.predictor.predict_single(applicant_data)
        
        if result is None:
            return "❌ Prediction failed. Please try again.", "", ""
        
        # Format result
        prediction = result['prediction']
        model_name = result['model_used']
        
        # Create main result message
        if prediction == 'Approved':
            main_result = f"✅ **LOAN APPROVED**"
            emoji = "🎉"
        else:
            main_result = f"❌ **LOAN REJECTED**"
            emoji = "😞"
        
        # Format confidence information
        confidence_info = ""
        if result['confidence']:
            approval_prob = result['confidence']['approval_probability']
            confidence_info = f"""
**Prediction Confidence:**
- Approval Probability: {approval_prob:.1%}
- Rejection Probability: {1-approval_prob:.1%}
"""
        
        # Get explanation
        explanation = self.predictor.explain_prediction(applicant_data)
        explanation_text = ""
        if explanation and explanation['key_factors']:
            explanation_text = "\n**Key Factors Considered:**\n"
            for factor in explanation['key_factors'][:3]:  # Top 3 factors
                explanation_text += f"- {factor['feature']}: {factor['value']} (importance: {factor['importance']:.3f})\n"
        
        # Format complete response
        detailed_result = f"""
{confidence_info}

**Model Used:** {model_name}
{explanation_text}
"""
        
        return main_result, detailed_result, emoji
    
    def create_interface(self):
        """Create the Gradio interface."""
        
        # Custom CSS for better styling
        css = """
        .main-header {
            text-align: center;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .prediction-output {
            font-size: 24px;
            font-weight: bold;
            text-align: center;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .approved {
            background-color: #d4edda;
            color: #155724;
            border: 2px solid #c3e6cb;
        }
        .rejected {
            background-color: #f8d7da;
            color: #721c24;
            border: 2px solid #f5c6cb;
        }
        """
        
        with gr.Blocks(css=css, title="Loan Approval Prediction") as interface:
            
            gr.HTML("""
            <div class="main-header">
                <h1>🏦 Loan Approval Prediction System</h1>
                <p>AI-powered loan approval prediction using machine learning</p>
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("## 📝 Applicant Information")
                    
                    with gr.Row():
                        gender = gr.Dropdown(
                            choices=["Male", "Female"],
                            label="Gender",
                            value="Male"
                        )
                        married = gr.Dropdown(
                            choices=["Yes", "No"],
                            label="Married",
                            value="Yes"
                        )
                    
                    with gr.Row():
                        dependents = gr.Dropdown(
                            choices=["0", "1", "2", "3+"],
                            label="Number of Dependents",
                            value="0"
                        )
                        education = gr.Dropdown(
                            choices=["Graduate", "Not Graduate"],
                            label="Education",
                            value="Graduate"
                        )
                    
                    self_employed = gr.Dropdown(
                        choices=["Yes", "No"],
                        label="Self Employed",
                        value="No"
                    )
                    
                    gr.Markdown("## 💰 Financial Information")
                    
                    with gr.Row():
                        applicant_income = gr.Number(
                            label="Applicant Income (₹)",
                            value=5000,
                            minimum=0
                        )
                        coapplicant_income = gr.Number(
                            label="Coapplicant Income (₹)",
                            value=0,
                            minimum=0
                        )
                    
                    with gr.Row():
                        loan_amount = gr.Number(
                            label="Loan Amount (₹ thousands)",
                            value=150,
                            minimum=0
                        )
                        loan_amount_term = gr.Dropdown(
                            choices=[120, 180, 240, 300, 360, 480],
                            label="Loan Amount Term (months)",
                            value=360
                        )
                    
                    with gr.Row():
                        credit_history = gr.Dropdown(
                            choices=[1.0, 0.0],
                            label="Credit History",
                            value=1.0,
                            info="1.0 = Good Credit, 0.0 = Poor Credit"
                        )
                        property_area = gr.Dropdown(
                            choices=["Rural", "Semiurban", "Urban"],
                            label="Property Area",
                            value="Urban"
                        )
                    
                    predict_btn = gr.Button(
                        "🔍 Predict Loan Approval",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("## 🎯 Prediction Result")
                    
                    # Output components
                    main_output = gr.Markdown(
                        "Click 'Predict Loan Approval' to get results",
                        elem_classes=["prediction-output"]
                    )
                    
                    detailed_output = gr.Markdown("")
                    
                    emoji_output = gr.Markdown(
                        "",
                        elem_classes=["emoji-output"]
                    )
            
            # Example section
            gr.Markdown("## 📋 Sample Test Cases")
            
            examples = [
                ["Male", "Yes", "0", "Graduate", "No", 8000, 2000, 200, 360, 1.0, "Urban"],
                ["Female", "No", "2", "Not Graduate", "Yes", 2500, 0, 150, 360, 0.0, "Rural"],
                ["Male", "Yes", "1", "Graduate", "No", 5000, 1500, 120, 360, 1.0, "Semiurban"],
                ["Female", "Yes", "0", "Graduate", "Yes", 6000, 0, 180, 240, 1.0, "Urban"],
                ["Male", "No", "3+", "Not Graduate", "No", 3000, 500, 100, 360, 0.0, "Rural"]
            ]
            
            gr.Examples(
                examples=examples,
                inputs=[gender, married, dependents, education, self_employed,
                       applicant_income, coapplicant_income, loan_amount,
                       loan_amount_term, credit_history, property_area],
                label="Click on any example to auto-fill the form"
            )
            
            # Information section
            with gr.Accordion("ℹ️ About This System", open=False):
                gr.Markdown("""
                ### How it works:
                1. **Data Input**: Enter applicant's personal and financial information
                2. **AI Processing**: Machine learning model analyzes the data
                3. **Prediction**: Get approval/rejection decision with confidence scores
                
                ### Model Information:
                - **Algorithm**: Random Forest Classifier (Best performing model)
                - **Accuracy**: ~82.5% on test data
                - **Features**: 11 key features including income, credit history, education, etc.
                - **Training Data**: Based on historical loan approval patterns
                
                ### Key Factors:
                - **Credit History**: Most important factor
                - **Income**: Both applicant and coapplicant income matter
                - **Education**: Graduate status influences approval
                - **Loan Amount**: Should be reasonable relative to income
                
                ### Notes:
                - This is a demonstration system
                - Results are predictions, not guarantees
                - Real loan approval involves additional factors and human review
                """)
            
            # Set up the prediction function
            predict_btn.click(
                fn=self.predict_loan_approval,
                inputs=[gender, married, dependents, education, self_employed,
                       applicant_income, coapplicant_income, loan_amount,
                       loan_amount_term, credit_history, property_area],
                outputs=[main_output, detailed_output, emoji_output]
            )
            
            gr.Markdown("""
            ---
            <div style="text-align: center; color: #666;">
                <p>🔒 Built with machine learning • 🚀 Powered by Gradio • 💡 Educational Purpose</p>
            </div>
            """)
        
        return interface


def main():
    """Main function to launch the app."""
    try:
        print("🚀 Starting Loan Approval Prediction System...")
        
        # Create app instance
        app = LoanApprovalApp()
        
        # Create and launch interface
        interface = app.create_interface()
        
        print("✅ System ready!")
        print("🌐 Launching web interface...")
        
        # Launch with custom settings
        interface.launch(
            server_name="127.0.0.1",  # Use localhost for Windows
            server_port=7860,
            share=False,  # Set to True if you want a public link
            debug=False,
            show_error=True,
            quiet=False,
            favicon_path=None,
            auth=None  # Add authentication if needed: auth=("username", "password")
        )
        
    except Exception as e:
        print(f"❌ Error launching application: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
