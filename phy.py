# lab# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('C:\Users\viarbat\Downloads\Lead+Scoring+Case+Study.zip\Lead Scoring Assignment.Lead.excel')

# Display the first few rows of the dataset
print(data.head())

# Handle missing values - 'Select' is treated as a null value
data.replace('Select', np.nan, inplace=True)

# Check for missing values
print(data.isnull().sum())

# Drop rows with missing target 'Converted'
data.dropna(subset=['Converted'], inplace=True)

# Fill missing values for numerical columns with median and categorical columns with mode
numerical_cols = data.select_dtypes(include=[np.number]).columns
categorical_cols = data.select_dtypes(include=[object]).columns

# Fill missing numerical data with median
data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].median())

# Fill missing categorical data with mode
data[categorical_cols] = data[categorical_cols].apply(lambda x: x.fillna(x.mode()[0]))

# Encode categorical variables using Label Encoding (for simplicity)
label_encoder = LabelEncoder()
for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col].astype(str))

# Split the dataset into features (X) and target variable (y)
X = data.drop(columns=['Converted'])  # All columns except target
y = data['Converted']  # Target variable

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features using StandardScaler (important for logistic regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the Logistic Regression model
log_reg = LogisticRegression(random_state=42)

# Train the model on the training set
log_reg.fit(X_train, y_train)

# Predict on the test set
y_pred = log_reg.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Print the classification report for detailed metrics
print(classification_report(y_test, y_pred))

# Calculate the ROC-AUC score
roc_auc = roc_auc_score(y_test, log_reg.predict_proba(X_test)[:, 1])
print(f'ROC-AUC: {roc_auc:.2f}')

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Converted', 'Converted'], yticklabels=['Not Converted', 'Converted'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Generate lead scores (probabilities of conversion) for each lead
lead_scores = log_reg.predict_proba(X_test)[:, 1] * 100  # Multiply by 100 to get scores between 0 and 100

# Display a few of the lead scores
print(f'Lead Scores: {lead_scores[:10]}')

# Add lead scores to the test set for better visualization
test_results = pd.DataFrame(X_test, columns=X.columns)
test_results['Actual'] = y_test
test_results['Lead_Score'] = lead_scores
print(test_results.head())

# Optionally, save the results as a CSV
test_results.to_csv('lead_scores.csv', index=False)
