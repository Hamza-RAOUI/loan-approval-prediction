"""
Visualization module for loan approval prediction.
Creates various plots for data exploration and analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class DataVisualizer:
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_categorical_distributions(self, data):
        """Plot distribution of categorical variables."""
        # Get categorical columns (object type)
        obj = (data.dtypes == 'object')
        object_cols = list(obj[obj].index)
        
        if not object_cols:
            print("No categorical variables found for plotting")
            return
        
        # Calculate subplot dimensions
        n_cols = 3
        n_rows = (len(object_cols) + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 5 * n_rows))
        
        for idx, col in enumerate(object_cols, 1):
            plt.subplot(n_rows, n_cols, idx)
            value_counts = data[col].value_counts()
            sns.barplot(x=value_counts.index, y=value_counts.values)
            plt.title(f'Distribution of {col}')
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for i, v in enumerate(value_counts.values):
                plt.text(i, v + 0.5, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_heatmap(self, data):
        """Plot correlation heatmap of numerical features."""
        plt.figure(figsize=self.figsize)
        
        # Select only numerical columns
        numeric_data = data.select_dtypes(include=[np.number])
        
        sns.heatmap(numeric_data.corr(), 
                   annot=True, 
                   cmap='BrBG', 
                   fmt='.2f',
                   linewidths=0.5, 
                   center=0)
        plt.title('Correlation Heatmap of Numerical Features')
        plt.tight_layout()
        plt.show()
    
    def plot_target_distribution(self, data, target_col='Loan_Status'):
        """Plot distribution of target variable."""
        plt.figure(figsize=(8, 6))
        
        if target_col in data.columns:
            value_counts = data[target_col].value_counts()
            
            # Create pie chart
            plt.subplot(1, 2, 1)
            plt.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
            plt.title('Loan Status Distribution')
            
            # Create bar chart
            plt.subplot(1, 2, 2)
            sns.barplot(x=value_counts.index, y=value_counts.values)
            plt.title('Loan Status Count')
            plt.ylabel('Count')
            
            for i, v in enumerate(value_counts.values):
                plt.text(i, v + 5, str(v), ha='center', va='bottom')
        else:
            print(f"Target column '{target_col}' not found")
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_vs_target(self, data, feature_cols, target_col='Loan_Status'):
        """Plot features against target variable."""
        if target_col not in data.columns:
            print(f"Target column '{target_col}' not found")
            return
        
        n_features = len(feature_cols)
        n_cols = 2
        n_rows = (n_features + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 5 * n_rows))
        
        for idx, feature in enumerate(feature_cols, 1):
            if feature in data.columns:
                plt.subplot(n_rows, n_cols, idx)
                
                if data[feature].dtype == 'object' or data[feature].nunique() < 10:
                    # Categorical or discrete feature
                    sns.countplot(data=data, x=feature, hue=target_col)
                    plt.xticks(rotation=45)
                else:
                    # Continuous feature
                    for status in data[target_col].unique():
                        subset = data[data[target_col] == status]
                        plt.hist(subset[feature], alpha=0.7, label=f'{target_col}={status}')
                    plt.xlabel(feature)
                    plt.legend()
                
                plt.title(f'{feature} vs {target_col}')
        
        plt.tight_layout()
        plt.show()
    
    def plot_income_analysis(self, data):
        """Analyze income-related features."""
        plt.figure(figsize=(15, 10))
        
        # Income distribution
        plt.subplot(2, 3, 1)
        if 'ApplicantIncome' in data.columns:
            plt.hist(data['ApplicantIncome'], bins=30, edgecolor='black')
            plt.title('Applicant Income Distribution')
            plt.xlabel('Income')
            plt.ylabel('Frequency')
        
        plt.subplot(2, 3, 2)
        if 'CoapplicantIncome' in data.columns:
            plt.hist(data['CoapplicantIncome'], bins=30, edgecolor='black')
            plt.title('Coapplicant Income Distribution')
            plt.xlabel('Income')
            plt.ylabel('Frequency')
        
        # Income vs Loan Amount
        plt.subplot(2, 3, 3)
        if 'ApplicantIncome' in data.columns and 'LoanAmount' in data.columns:
            plt.scatter(data['ApplicantIncome'], data['LoanAmount'], alpha=0.6)
            plt.xlabel('Applicant Income')
            plt.ylabel('Loan Amount')
            plt.title('Income vs Loan Amount')
        
        # Combined income
        plt.subplot(2, 3, 4)
        if 'ApplicantIncome' in data.columns and 'CoapplicantIncome' in data.columns:
            total_income = data['ApplicantIncome'] + data['CoapplicantIncome']
            plt.hist(total_income, bins=30, edgecolor='black')
            plt.title('Total Income Distribution')
            plt.xlabel('Total Income')
            plt.ylabel('Frequency')
        
        # Income vs Loan Status
        plt.subplot(2, 3, 5)
        if 'ApplicantIncome' in data.columns and 'Loan_Status' in data.columns:
            sns.boxplot(data=data, x='Loan_Status', y='ApplicantIncome')
            plt.title('Income vs Loan Status')
        
        # Loan Amount vs Loan Status
        plt.subplot(2, 3, 6)
        if 'LoanAmount' in data.columns and 'Loan_Status' in data.columns:
            sns.boxplot(data=data, x='Loan_Status', y='LoanAmount')
            plt.title('Loan Amount vs Loan Status')
        
        plt.tight_layout()
        plt.show()
    
    def plot_credit_history_impact(self, data):
        """Analyze credit history impact on loan approval."""
        if 'Credit_History' not in data.columns or 'Loan_Status' not in data.columns:
            print("Credit_History or Loan_Status column not found")
            return
        
        plt.figure(figsize=(12, 5))
        
        # Credit history distribution
        plt.subplot(1, 2, 1)
        credit_counts = data['Credit_History'].value_counts()
        plt.pie(credit_counts.values, labels=['Good Credit', 'Poor Credit'], autopct='%1.1f%%')
        plt.title('Credit History Distribution')
        
        # Credit history vs loan approval
        plt.subplot(1, 2, 2)
        cross_tab = pd.crosstab(data['Credit_History'], data['Loan_Status'])
        cross_tab.plot(kind='bar')
        plt.title('Credit History vs Loan Approval')
        plt.xlabel('Credit History (0: Poor, 1: Good)')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.legend(title='Loan Status')
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        approval_rates = data.groupby('Credit_History')['Loan_Status'].apply(
            lambda x: (x == 'Y').mean() if 'Y' in x.values else (x == 1).mean()
        )
        print("\nLoan Approval Rates by Credit History:")
        for credit, rate in approval_rates.items():
            status = "Good Credit" if credit == 1 else "Poor Credit"
            print(f"{status}: {rate:.2%}")
    
    def create_comprehensive_report(self, data, target_col='Loan_Status'):
        """Create a comprehensive visualization report."""
        print("Creating comprehensive visualization report...")
        
        # Target distribution
        self.plot_target_distribution(data, target_col)
        
        # Categorical distributions
        self.plot_categorical_distributions(data)
        
        # Correlation heatmap
        self.plot_correlation_heatmap(data)
        
        # Income analysis
        self.plot_income_analysis(data)
        
        # Credit history impact
        self.plot_credit_history_impact(data)
        
        # Feature vs target for key features
        key_features = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
        available_features = [f for f in key_features if f in data.columns]
        if available_features:
            self.plot_feature_vs_target(data, available_features, target_col)


if __name__ == "__main__":
    # Test visualization with sample data
    try:
        # Try to load actual data
        data = pd.read_csv("../data/LoanApprovalPrediction.csv")
    except:
        # Create sample data for testing
        from data_preprocessing import create_sample_data
        data = create_sample_data()
    
    visualizer = DataVisualizer()
    visualizer.create_comprehensive_report(data)
