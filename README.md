# Diabetes Readmission Prediction — End-to-End ML Pipeline

## About this project

In this internship milestone, I built a machine learning pipeline to predict whether a patient with diabetes may be readmitted to the hospital within 30 days. I used the public UCI diabetes hospital dataset and trained a logistic regression model.

## Topics covered in the notebook

- Loading and exploring a real dataset
- Handling missing values and basic data cleaning
- Feature engineering and creating a binary classification target
- Class imbalance and balanced class weights
- Patient-level train/test splitting to reduce data leakage
- Numerical preprocessing: median imputation and standardization
- Categorical preprocessing: most-frequent imputation and one-hot encoding
- `Pipeline` and `ColumnTransformer` in scikit-learn
- Logistic regression and patient-aware cross-validation
- Model evaluation: accuracy, precision, recall, F1, ROC-AUC, average precision, and confusion matrix
- ROC and precision-recall curves
- Comparing performance with a baseline model
- Inspecting transformed features and model coefficients
- Making predictions and saving the complete pipeline with Joblib

## What I did

I cleaned the dataset, created a medication-count feature, prepared numerical and categorical preprocessing steps, and trained a logistic regression model. I made sure encounters from the same patient stayed in the same data split. I then evaluated the model using different metrics and saved the pipeline for reuse.

## Tools used

Python, Pandas, NumPy, Matplotlib, Scikit-learn, Joblib, and Google Colab.

## How to run

1. Open `Diabetes_Readmission_End_to_End_Reviewed.ipynb` in Google Colab.
2. Choose **Runtime → Run all**.
3. The notebook downloads the UCI dataset and runs the ML pipeline.
4. Check the results and the saved `diabetes_readmission_pipeline.joblib` file.

The dataset download requires internet access. Model training and cross-validation can take some time.

## Dataset

[UCI Diabetes 130-US Hospitals (1999–2008)](https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999+2008)

## Note

This is a learning project, not a model for medical decisions. The reviewed notebook was smoke-tested with synthetic records, but the final results need verification using the real dataset in Colab.
