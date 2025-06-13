# Online Payment Fraud Detection

This project implements a machine learning-based system for detecting fraudulent online payment transactions. The system uses various machine learning algorithms to classify transactions as either legitimate or fraudulent.

## Features

- Data analysis and visualization of transaction patterns
- Multiple machine learning models (Logistic Regression, XGBoost, Random Forest)
- Performance evaluation using ROC-AUC scores
- Confusion matrix visualization
- Comprehensive data visualization including:
  - Transaction type distribution
  - Transaction amount analysis
  - Correlation heatmap
  - Transaction steps distribution

## Requirements

The project requires the following Python packages:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- xgboost

You can install all required packages using:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your transaction data in a CSV file named 'new_file.csv' in the project root directory
2. Run the main script:
```bash
python fraud_detection.py
```

The script will:
- Load and analyze the data
- Generate visualizations (saved in the 'plots' directory)
- Train multiple machine learning models
- Evaluate model performance
- Generate a confusion matrix for the best performing model

## Project Structure

```
.
├── README.md
├── requirements.txt
├── fraud_detection.py
├── .gitignore
└── plots/
    ├── transaction_types_distribution.png
    ├── transaction_amount_by_type.png
    ├── transaction_steps_distribution.png
    ├── correlation_heatmap.png
    └── confusion_matrix_xgboost.png
```

## Data Format

The input CSV file should contain the following columns:
- step: Integer representing the time step
- type: Transaction type
- amount: Transaction amount
- nameOrig: Origin account name
- oldbalanceOrg: Original balance of origin account
- newbalanceOrig: New balance of origin account
- nameDest: Destination account name
- oldbalanceDest: Original balance of destination account
- newbalanceDest: New balance of destination account
- isFraud: Binary indicator of fraud (1 for fraud, 0 for legitimate)

## Model Performance

The project implements three different models with their respective ROC-AUC scores:

1. Logistic Regression
   - Training ROC-AUC Score: 0.9015
   - Validation ROC-AUC Score: 0.8996

2. XGBoost
   - Training ROC-AUC Score: 0.7493
   - Validation ROC-AUC Score: 0.7567

3. Random Forest
   - Training ROC-AUC Score: 0.9999
   - Validation ROC-AUC Score: 0.9654

### Performance Analysis
- The Random Forest model shows the best performance with a validation ROC-AUC score of 0.9654
- Logistic Regression performs well with a validation score of 0.8996
- XGBoost shows relatively lower performance but still maintains good discrimination power

### Dataset Statistics
- Total number of transactions: 6,362,620
- Number of fraudulent transactions: 8,213 (0.13% of total)
- Number of legitimate transactions: 6,354,407 (99.87% of total)

## License

This project is open source and available under the MIT License. 