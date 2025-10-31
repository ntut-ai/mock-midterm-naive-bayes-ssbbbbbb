#!/usr/bin/env python
# coding: utf-8

# Naive Bayes Classifier (with optional PCA)
# Reference: https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/

import pandas as pd
import numpy as np
import argparse
from math import sqrt, exp, pi

# 如果使用 PCA
try:
    from sklearn.decomposition import PCA
except ImportError:
    PCA = None
    print("⚠️  Warning: sklearn 未安裝，無法使用 PCA 模式。")

# ---------- Utility Functions ----------

def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
        print('[%s] => %d' % (value, i))
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

def separate_by_class(dataset):
    separated = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if class_value not in separated:
            separated[class_value] = list()
        separated[class_value].append(vector)
    return separated

def mean(numbers):
    return sum(numbers) / float(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x - avg) ** 2 for x in numbers]) / float(len(numbers) - 1)
    return sqrt(variance)

def summarize_dataset(dataset):
    summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
    del summaries[-1]  # remove label column
    return summaries

def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = dict()
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
    return summaries

def calculate_probability(x, mean, stdev):
    if stdev == 0:
        return 1.0 if x == mean else 0.0
    exponent = exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent

def calculate_class_probabilities(summaries, row):
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][2] / float(total_rows)
        for i in range(len(class_summaries)):
            mean, stdev, _ = class_summaries[i]
            probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
    return probabilities

# ---------- Naive Bayes Train / Predict ----------

def nb_train(train_data):
    model = summarize_by_class(train_data)
    return model

def nb_predict(summaries, row):
    probabilities = calculate_class_probabilities(summaries, row)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label

# ---------- Main Program ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Naive Bayes Classifier with optional PCA")
    parser.add_argument("--train-csv", required=True, help="Training data CSV file (labels in last column)")
    parser.add_argument("--test-csv", required=True, help="Test data CSV file")
    parser.add_argument("--use-pca", action="store_true", help="Enable PCA (Principal Component Analysis)")
    parser.add_argument("--pca-components", type=int, default=2, help="Number of PCA components (default=2)")
    args = parser.parse_args()

    # Load data
    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    train_data = train_df.to_numpy()
    test_data = test_df.iloc[:, :-1].to_numpy()
    test_label = test_df.iloc[:, -1:].to_numpy()

    # Label encoding
    label_id_dict = str_column_to_int(train_data, len(train_data[0]) - 1)
    id_label_dict = {v: k for k, v in label_id_dict.items()}

    # ----- Optional PCA -----
    if args.use_pca:
        if PCA is None:
            raise ImportError("scikit-learn is required for PCA. Install via `pip install scikit-learn`.")
        print(f"🔹 Using PCA (n_components={args.pca_components}) ...")

        # Separate X, y
        X_train = np.array([row[:-1] for row in train_data])
        y_train = np.array([row[-1] for row in train_data])

        # Fit PCA
        pca = PCA(n_components=args.pca_components)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(test_data)

        # Combine back into [X|y] format for training
        train_data = np.hstack((X_train_pca, y_train.reshape(-1, 1)))
        test_data = X_test_pca

    # Train model
    model = nb_train(train_data)

    # Validate
    row = [5.7, 2.9, 4.2, 1.3]
    if args.use_pca:
        row = pca.transform([row])[0]
    label = nb_predict(model, row)
    print('Validation on Data=%s, Predicted: %s' % (row, label))

    # Predict on test set
    predictions = []
    rows, _ = test_data.shape
    for i in range(rows):
        y_pred = nb_predict(model, test_data[i])
        predictions.append([id_label_dict[y_pred]])

    # Compute accuracy
    result = np.array(predictions) == test_label
    accuracy = sum(result == True) / len(result)
    print('✅ Evaluate Naive Bayes on Iris dataset. Accuracy = %.2f' % accuracy)
