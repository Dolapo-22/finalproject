# Final Term Project - Classification Model Comparison

**Author**: Idowu Dolapo  
**Course**: Machine Learning Final Project  
**Title**: Comparative Analysis of Classification Models on Amazon Book Reviews

---

## Project Overview

This project compares the performance of three different classification models for sentiment analysis on reviews of the book *Effective Java*. The models used are:

- Random Forest (Scikit-learn)
- Naive Bayes (Scikit-learn)
- LSTM (Deep Learning with TensorFlow/Keras)

Each model's performance is evaluated using precision, recall, F1 score, and additional derived metrics.

---

## Dataset

- The dataset used is a collection of Amazon reviews about the book *Effective Java*.
- If the dataset file `amazon_effective_java_dataset.csv` is not found in the working directory, a small hardcoded fallback dataset is automatically used for demonstration purposes.

---

## Requirements

Make sure you have Python 3 installed along with the following libraries:

- pandas
- numpy
- scikit-learn
- tensorflow (optional, only required for LSTM model)

You can install all dependencies using pip:

```bash
pip install pandas numpy scikit-learn tensorflow
