"""
Final Term Project - Classification Model Comparison
Author: Idowu Dolapo
Dataset: Amazon Reviews (Effective Java)
Models: Random Forest, Naive Bayes, LSTM (Deep Learning)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Sklearn libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# TensorFlow/Keras libraries
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("TensorFlow is not installed. LSTM model will be skipped.")
    TENSORFLOW_AVAILABLE = False

# Step 1: Load dataset or create fallback
def load_dataset():
    try:
        df = pd.read_csv("amazon_effective_java_dataset.csv")
        print("Dataset loaded from CSV.")
    except FileNotFoundError:
        print("CSV not found. Creating fallback dataset.")
        data = {
            "Review": [
                "This book is a must-read for Java developers.",
                "Terrible quality, nothing useful.",
                "Effective Java helped me write cleaner code.",
                "I didn't find this helpful at all.",
                "Highly recommended for object-oriented programming.",
                "Not about Java at all, misleading title.",
                "Great insights from Joshua Bloch!",
                "Waste of money, not recommended.",
                "The best Java reference out there.",
                "Too advanced for beginners."
            ],
            "Label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        }
        df = pd.DataFrame(data)
    return df

df = load_dataset()
X = df["Review"]
y = df["Label"]

# Step 2: Preprocess for TF-IDF models (Random Forest & Naive Bayes)
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Step 3: Random Forest with 10-fold cross-validation
rf = RandomForestClassifier(n_estimators=100, random_state=42)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_scores = cross_val_score(rf, X_tfidf, y, cv=kfold, scoring="f1_weighted")

# Step 4: Naive Bayes with 10-fold cross-validation
nb = MultinomialNB()
nb_scores = cross_val_score(nb, X_tfidf, y, cv=kfold, scoring="f1_weighted")

# Step 5: Prepare for LSTM
if TENSORFLOW_AVAILABLE:
    max_words = 1000
    max_len = 100

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X)
    X_seq = tokenizer.texts_to_sequences(X)
    X_pad = pad_sequences(X_seq, maxlen=max_len)
    y_dl = np.array(y)

    # Step 6: LSTM model
    model = Sequential([
        Embedding(max_words, 64, input_length=max_len),
        SpatialDropout1D(0.2),
        LSTM(64, dropout=0.2, recurrent_dropout=0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_pad, y_dl, epochs=10, batch_size=2, validation_split=0.2, verbose=0)

# Step 7: Manual performance metrics
def get_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) else 0
    fnr = fn / (fn + tp) if (fn + tp) else 0
    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "FPR": fpr,
        "FNR": fnr
    }

# Step 8: Generate predictions and print metrics
results = {}

rf.fit(X_tfidf, y)
y_pred_rf = rf.predict(X_tfidf)
results["Random Forest"] = get_metrics(y, y_pred_rf)

nb.fit(X_tfidf, y)
y_pred_nb = nb.predict(X_tfidf)
results["Naive Bayes"] = get_metrics(y, y_pred_nb)

if TENSORFLOW_AVAILABLE:
    y_pred_dl = model.predict(X_pad).flatten()
    y_pred_dl = np.where(y_pred_dl > 0.5, 1, 0)
    results["LSTM"] = get_metrics(y, y_pred_dl)

# Display the results
for model_name, metrics in results.items():
    print(f"\n{model_name} Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
