import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Load the dataset
df = pd.read_csv("churn.csv")

# 2. Data Cleaning: Drop useless columns and data leakage sources
df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'Complain'])

# Convert categorical text columns into numeric dummy variables
df = pd.get_dummies(df, drop_first=True)

# Separate features (X) and target variable (y)
X = df.drop(columns=['Exited'])
y = df['Exited']

# Split the dataset into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Feature Scaling to normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Initialize two different classification algorithms
rf_model = RandomForestClassifier(random_state=42)
lr_model = LogisticRegression(random_state=42)

# Train both models on the exact same training data
rf_model.fit(X_train_scaled, y_train)
lr_model.fit(X_train_scaled, y_train)

# Make predictions on the unseen test data
rf_preds = rf_model.predict(X_test_scaled)
lr_preds = lr_model.predict(X_test_scaled)

# 4. Print Evaluation Metrics for Model 1 (Random Forest)
print("====================================")
print("MODEL 1: RANDOM FOREST RESULTS")
print("====================================")
print("Accuracy Score:", accuracy_score(y_test, rf_preds))
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_preds))

# Print Evaluation Metrics for Model 2 (Logistic Regression)
print("\n====================================")
print("MODEL 2: LOGISTIC REGRESSION RESULTS")
print("====================================")
print("Accuracy Score:", accuracy_score(y_test, lr_preds))
print("Confusion Matrix:")
print(confusion_matrix(y_test, lr_preds))