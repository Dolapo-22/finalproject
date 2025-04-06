# Random Forest Classifier

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold

# Load dataset (use your own dataset here)
data = pd.read_csv('your_dataset.csv')
X = data.drop(columns='target')  # Features
y = data['target']  # Target variable

# Split data into training and testing sets (not needed for k-fold, just for reference)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Initialize KFold for cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Store metrics for each fold
metrics_list = []

# KFold cross-validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fit the model
    rf.fit(X_train, y_train)
    
    # Predict the labels
    y_pred = rf.predict(X_test)
    
    # Calculate performance metrics
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate additional metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    metrics_list.append({
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn, 'Accuracy': accuracy, 'Precision': precision,
        'Recall': recall, 'F1 Score': f1_score
    })

# Create a DataFrame to display the metrics
metrics_df = pd.DataFrame(metrics_list)
print(metrics_df)

# Calculate the average metrics across all folds
average_metrics = metrics_df.mean()
print("Average Metrics: ")
print(average_metrics)
