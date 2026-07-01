# i2i-Academy-Applied_Data_Science-3
Machine learning project to predict customer churn using Random Forest and logistic Regression 
# Bank Customer Churn Prediction

This project predicts if a bank customer will leave the bank (churn) or stay. We use a dataset from Kaggle and compare two different machine learning models to see which one performs better.

## Tools and Technologies
* **Python**: The main programming language.
* **Pandas**: Used to load and clean the dataset.
* **Scikit-learn**: Used for splitting data, feature scaling, training models, and evaluation.

## Data Cleaning and The Leakage Problem
We prepared and cleaned the data with these steps:
1. We removed `RowNumber`, `CustomerId`, and `Surname` because they are unique IDs and do not help the model learn real patterns.
2. **Fixing Data Leakage**: At first, both models gave a 99.9% accuracy score. We investigated the data and found that the `Complain` column was causing data leakage. Every customer who complained was automatically marked as exited. We dropped the `Complain` column so the models could learn from real customer behavior instead of cheating.
3. We converted categorical text data into numeric variables using dummy variables.

## Models and Results
We split the data into 80% training and 20% testing sets. We trained two different classification algorithms:

* **Random Forest Classifier**: 86.6% Accuracy
* **Logistic Regression**: 81.1% Accuracy

### Conclusion
The Random Forest model performed better than Logistic Regression. This is because the dataset has complex and non-linear relationships between features like age, number of products, and balance. Logistic Regression uses a simple straight line, which cannot capture these complex customer behaviors.

## How to Run the Project
1. Install the required libraries:
   pip install pandas scikit-learn

2. Place the `churn.csv` file in the project folder.

3. Run the script:
   python main.py
