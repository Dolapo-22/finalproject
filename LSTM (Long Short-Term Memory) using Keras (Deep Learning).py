# LSTM Classifier

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset (use your own dataset here)
data = pd.read_csv('your_dataset.csv')
X = data.drop(columns='target')  # Features
y = data['target']  # Target variable

# Encode the labels for classification
encoder = LabelEncoder()
y = encoder.fit_transform(y)
y = to_categorical(y)  # One-hot encode for multi-class classification

# Reshape X to be 3D as required by LSTM [samples, time steps, features]
X = np.array(X)
X = X.reshape((X.shape[0], 1, X.shape[1]))

# Initialize KFold for cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Store metrics for each fold
metrics_list = []

# KFold cross-validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Create LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))  # 2 output classes (binary classification)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Fit the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    
    # Predict the labels
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Calculate performance metrics
    cm = confusion_matrix(y_true, y_pred)
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
