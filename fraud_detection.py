#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Online Payment Fraud Detection using Machine Learning
This script analyzes transaction data and builds machine learning models
to detect fraudulent transactions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import os

def load_and_prepare_data(file_path):
    """Load and prepare the dataset."""
    data = pd.read_csv(file_path)
    return data

def analyze_data(data):
    """Perform initial data analysis."""
    print("\nDataset Info:")
    print(data.info())
    
    print("\nDataset Statistics:")
    print(data.describe())
    
    # Analyze column types
    obj = (data.dtypes == 'object')
    object_cols = list(obj[obj].index)
    print("\nCategorical variables:", len(object_cols))
    
    int_ = (data.dtypes == 'int')
    num_cols = list(int_[int_].index)
    print("Integer variables:", len(num_cols))
    
    fl = (data.dtypes == 'float')
    fl_cols = list(fl[fl].index)
    print("Float variables:", len(fl_cols))
    
    return object_cols, num_cols, fl_cols

def save_plot(fig, filename):
    """Save the plot to the plots directory."""
    plt.savefig(os.path.join('plots', filename), bbox_inches='tight', dpi=300)
    plt.close(fig)

def visualize_data(data):
    """Create visualizations for data analysis."""
    # Transaction type distribution
    fig = plt.figure(figsize=(10, 6))
    sns.countplot(x='type', data=data)
    plt.title('Distribution of Transaction Types')
    save_plot(fig, 'transaction_types_distribution.png')
    
    # Transaction amount by type
    fig = plt.figure(figsize=(10, 6))
    sns.barplot(x='type', y='amount', data=data)
    plt.title('Average Transaction Amount by Type')
    save_plot(fig, 'transaction_amount_by_type.png')
    
    # Fraud distribution
    print("\nFraud Distribution:")
    print(data['isFraud'].value_counts())
    
    # Step distribution
    fig = plt.figure(figsize=(15, 6))
    sns.distplot(data['step'], bins=50)
    plt.title('Distribution of Transaction Steps')
    save_plot(fig, 'transaction_steps_distribution.png')
    
    # Correlation heatmap
    fig = plt.figure(figsize=(12, 6))
    sns.heatmap(data.apply(lambda x: pd.factorize(x)[0]).corr(),
                cmap='BrBG',
                fmt='.2f',
                linewidths=2,
                annot=True)
    plt.title('Correlation Heatmap')
    save_plot(fig, 'correlation_heatmap.png')

def prepare_features(data):
    """Prepare features for model training."""
    # One-hot encode transaction types
    type_new = pd.get_dummies(data['type'], drop_first=True)
    data_new = pd.concat([data, type_new], axis=1)
    
    # Prepare X and y
    X = data_new.drop(['isFraud', 'type', 'nameOrig', 'nameDest'], axis=1)
    y = data_new['isFraud']
    
    return X, y

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple models."""
    models = [
        LogisticRegression(),
        XGBClassifier(),
        RandomForestClassifier(n_estimators=7,
                             criterion='entropy',
                             random_state=7)
    ]
    
    for model in models:
        print(f'\nTraining {model.__class__.__name__}:')
        model.fit(X_train, y_train)
        
        # Training accuracy
        train_preds = model.predict_proba(X_train)[:, 1]
        print('Training ROC-AUC Score:', roc_auc_score(y_train, train_preds))
        
        # Validation accuracy
        y_preds = model.predict_proba(X_test)[:, 1]
        print('Validation ROC-AUC Score:', roc_auc_score(y_test, y_preds))
        
        # Plot confusion matrix for XGBoost model
        if isinstance(model, XGBClassifier):
            fig = plt.figure(figsize=(8, 6))
            cm = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
            cm.plot(cmap='Blues')
            plt.title('Confusion Matrix - XGBoost Model')
            save_plot(fig, 'confusion_matrix_xgboost.png')

def main():
    """Main function to run the fraud detection analysis."""
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Load data
    data = load_and_prepare_data('new_file.csv')
    
    # Analyze data
    analyze_data(data)
    
    # Visualize data
    visualize_data(data)
    
    # Prepare features
    X, y = prepare_features(data)
    print("\nFeature shape:", X.shape)
    print("Target shape:", y.shape)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train and evaluate models
    train_and_evaluate_models(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main() 