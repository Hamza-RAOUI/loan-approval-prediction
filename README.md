# Loan Approval Prediction using Machine Learning

A machine learning project that predicts loan approval status based on applicant information using various classification algorithms.

## Project Overview

Banks receive a major portion of their profit from loans, but deciding whether an applicant's profile is relevant for loan approval requires analyzing many aspects. This project uses machine learning algorithms to predict whether a candidate's profile is suitable for loan approval based on key features like Marital Status, Education, Applicant Income, Credit History, etc.

## Dataset Features

The dataset contains 13 features:

| Feature | Description |
|---------|-------------|
| Loan_ID | A unique identifier |
| Gender | Gender of the applicant (Male/Female) |
| Married | Marital Status (Yes/No) |
| Dependents | Number of dependents |
| Education | Education level (Graduate/Not Graduate) |
| Self_Employed | Employment status (Yes/No) |
| ApplicantIncome | Applicant's income |
| CoapplicantIncome | Co-applicant's income |
| LoanAmount | Loan amount (in thousands) |
| Loan_Amount_Term | Loan term (in months) |
| Credit_History | Credit history (1.0/0.0) |
| Property_Area | Property area (Rural/Urban/Semiurban) |
| Loan_Status | Target variable (Y/N) |

## Models Used

- **Random Forest Classifier** (Best performing - 82.5% accuracy)
- **Logistic Regression** (80.83% accuracy)
- **K-Nearest Neighbors** (63.75% accuracy)
- **Support Vector Machine** (69.17% accuracy)

## Project Structure

```
loan_approval_prediction/
├── data/
│   └── LoanApprovalPrediction.csv
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── visualization.py
│   └── prediction.py
├── models/
│   └── best_model.pkl
├── notebooks/
│   └── exploratory_data_analysis.ipynb
├── app.py                 # Gradio interface
├── requirements.txt
├── main.py               # Main execution script
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd loan_approval_prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the dataset and place it in the `data/` folder.

## Usage

### Exploratory Data Analysis

Open and run the Jupyter notebook for comprehensive data exploration:

```bash
jupyter notebook notebooks/exploratory_data_analysis.ipynb
```

### Training the Model

```bash
python main.py
```

### Running the Web Interface

```bash
python app.py
```

This will launch a Gradio interface where you can input applicant details and get loan approval predictions.

## Results

The Random Forest Classifier achieved the best performance with 82.5% accuracy on the test set. The model shows that Credit History has the highest impact on loan approval decisions.

## Key Insights

- Credit History is the most important feature for loan approval
- There's a positive correlation between Applicant Income and Loan Amount
- Married applicants with good credit history have higher approval rates

## Future Improvements

- Implement ensemble learning techniques (Bagging and Boosting)
- Feature engineering for better performance
- Cross-validation for more robust evaluation
- Hyperparameter tuning for optimal results
