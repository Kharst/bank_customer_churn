# Bank Customer Churn Prediction

This repository contains a machine learning model designed to predict bank customer churn. The model was trained using a Tuned Random Forest algorithm, which provides robust predictions by optimizing hyperparameters. The project aims to help banks identify customers who are at risk of leaving, enabling proactive retention strategies.

## Project Details

- **Model**: Tuned Random Forest
- **Objective**: Predict whether a customer will churn.
- **Key Features**: Customer demographics, account balance, transactions, and more.
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1 Score, AUC-ROC.

## How to Use

1. **Data**: The dataset used for training was sourced from Kaggle.
2. **Model Training**: The model was tuned using grid search and cross-validation to select the best parameters.
3. **Prediction**: The model can predict if a customer will churn based on input features like tenure, number of products, and balance.

## Dependencies

To run the project, install the required Python packages:

```bash
pip install -r requirements.txt
