#############################################################
## ASSIGNMENT 2 CODE SKELETON
## RELEASED: 2/2/2020
## DUE: 2/12/2020
## DESCRIPTION: In this assignment, you will explore the
## text classification problem of identifying complex words.
## We have provided the following skeleton for your code,
## with several helper functions, and all the required
## functions you need to write.
#############################################################

from collections import defaultdict
import gzip
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

#### 1. Evaluation Metrics ####

def get_comparison_report(y_pred, y_true):
    result = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for pred, true in zip(y_pred, y_true):
        if pred == 1:
            if true == 1:
                result["tp"] += 1
            else:
                result["fp"] += 1
        else:
            if true == 1:
                result["tn"] += 1
            else:
                result["fn"] += 1
    return result


## Input: y_pred, a list of length n with the predicted labels,
## y_true, a list of length n with the true labels

## Calculates the precision of the predicted labels
def get_precision(y_pred, y_true):
    ## YOUR CODE HERE...
    report = get_comparison_report(y_pred, y_true)
    total_pred_pos = report["tp"] + report["fp"]
    true_pos = report["tp"]
    return true_pos / total_pred_pos if total_pred_pos else 1


## Calculates the recall of the predicted labels
def get_recall(y_pred, y_true):
    ## YOUR CODE HERE...
    report = get_comparison_report(y_pred, y_true)
    total_true_pos = report["tp"] + report["fn"]
    true_pos = report["tp"]
    return true_pos / total_true_pos if total_true_pos else 1


## Calculates the f-score of the predicted labels
def get_fscore(y_pred, y_true):
    ## YOUR CODE HERE...
    precision = get_precision(y_pred, y_true)
    recall = get_recall(y_pred, y_true)
    return 2 * precision * recall / (precision + recall)


def get_metrics(y_pred, y_true):
    precision = get_precision(y_pred, y_true)
    recall = get_recall(y_pred, y_true)
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 and recall != 0 else 0
    return precision, recall, f1


def test_predictions(y_pred, y_true):
    precision, recall, f1 = get_metrics(y_pred, y_true)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)


#### 2. Complex Word Identification ####

## Loads in the words and labels of one of the datasets
def load_file(data_file):
    words = []
    labels = []
    with open(data_file, 'rt', encoding="utf8") as f:
        i = 0
        for line in f:
            if i > 0:
                line_split = line[:-1].split("\t")
                words.append(line_split[0].lower())
                labels.append(int(line_split[1]))
            i += 1
    return words, labels


### 2.1: A very simple baseline

## Makes feature matrix for all complex
def all_complex_feature(words):
    return [1] * len(words)


## Labels every word complex
def all_complex(data_file):
    words, labels = load_file(data_file)
    pred = all_complex_feature(words)
    performance = get_metrics(pred, labels)
    test_predictions(pred, labels)
    return performance


def majority_class(training_file, development_file):
    print("1. TRAIN DATA")
    all_complex(training_file)
    print("2. DEV DATA")
    all_complex(development_file)


### 2.2: Word length thresholding

## Makes feature matrix for word_length_threshold
def length_threshold_feature(words, threshold):
    return [1 if len(word) >= threshold else 0 for word in words]


## Finds the best length threshold by f-score, and uses this threshold to
## classify the training and development set
def word_length_threshold(training_file, development_file):
    # find best threshold on training data
    print("1. TRAIN DATA")
    train_words, train_labels = load_file(training_file)
    train_metrics = {}  # used for drawing
    max_f1 = -1
    best_threshold = None
    for thres in range(4, 21):
        pred = length_threshold_feature(train_words, thres)
        precision, recall, f1 = get_metrics(pred, train_labels)
        train_metrics[thres] = precision, recall, f1
        print("==Testing on training data with length threshold:", thres)
        test_predictions(pred, train_labels)
        if max_f1 < f1:
            max_f1 = f1
            best_threshold = thres

    print("Best threshold is:", best_threshold)
    training_performance = train_metrics[best_threshold]

    # apply on development data
    print("2. DEV DATA")
    dev_words, dev_labels = load_file(development_file)
    dev_pred = length_threshold_feature(dev_words, best_threshold)
    development_performance = get_metrics(dev_pred, dev_labels)
    test_predictions(dev_pred, dev_labels)

    return training_performance, development_performance


### 2.3: Word frequency thresholding

## Loads Google NGram counts
def load_ngram_counts(ngram_counts_file):
    counts = defaultdict(int)
    with gzip.open(ngram_counts_file, 'rt') as f:
        for line in f:
            token, count = line.strip().split('\t')
            if token[0].islower():
                counts[token] = int(count)
    return counts


# Finds the best frequency threshold by f-score, and uses this threshold to
## classify the training and development set

## Make feature matrix for word_frequency_threshold
def frequency_threshold_feature(words, threshold, counts):
    return [1 if counts[word] >= threshold else 0 for word in words]


def word_frequency_threshold(training_file, development_file, counts):
    # find best threshold with training data
    print("1. TRAIN DATA")
    train_words, train_labels = load_file(training_file)
    train_word_freq = sorted([counts[word] for word in train_words])
    train_metrics = {}  # used for drawing
    max_f1 = -1
    best_threshold = None
    for perc in range(5, 100, 5):
        thres = int(np.percentile(train_word_freq, perc))
        pred = frequency_threshold_feature(train_words, thres, counts)
        precision, recall, f1 = get_metrics(pred, train_labels)
        train_metrics[thres] = precision, recall, f1
        print("==Testing on training data with frequency threshold:", thres)
        test_predictions(pred, train_labels)
        if max_f1 < f1:
            max_f1 = f1
            best_threshold = thres
    training_performance = train_metrics[best_threshold]
    print("\nBest threshold:", best_threshold, training_performance)

    # apply on dev data
    print("2. DEV DATA")
    dev_words, dev_labels = load_file(development_file)
    dev_pred = frequency_threshold_feature(dev_words, best_threshold, counts)
    development_performance = get_metrics(dev_pred, dev_labels)
    test_predictions(dev_pred, dev_labels)

    return training_performance, development_performance


### 2.4: Naive Bayes

def get_features(words, counts, standardize=False):
    length = [len(word) for word in words]
    frequency = [counts[word] for word in words]
    features = np.array([*zip(length, frequency)])
    if standardize:
        fmean = np.mean(features, axis=0)
        fstd = np.std(features, axis=0)
        return (features - fmean) / fstd, fmean, fstd
    return features


def apply_sk_model(model, training_file, development_file, counts):
    # train
    print("1. TRAIN DATA")
    train_words, train_labels = load_file(training_file)
    X_train, train_mean, train_std = get_features(train_words, counts, standardize=True)
    y_train = np.array(train_labels)
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)

    training_performance = get_metrics(train_pred, train_labels)
    test_predictions(train_pred, train_labels)

    # apply on dev
    print("2. DEV DATA")
    dev_words, dev_labels = load_file(development_file)
    X_dev = get_features(dev_words, counts)
    standardized_X_dev = (X_dev - train_mean) / train_std
    dev_pred = model.predict(standardized_X_dev)

    development_performance = get_metrics(dev_pred, dev_labels)
    test_predictions(dev_pred, dev_labels)
    return training_performance, development_performance


## Trains a Naive Bayes classifier using length and frequency features
def naive_bayes(training_file, development_file, counts):
    return apply_sk_model(GaussianNB(), training_file, development_file, counts)


### 2.5: Logistic Regression

## Trains a Naive Bayes classifier using length and frequency features
def logistic_regression(training_file, development_file, counts):
    return apply_sk_model(LogisticRegression(), training_file, development_file, counts)


### 2.7: Build your own classifier

## Trains a classifier of your choosing, predicts labels for the test dataset
## and writes the predicted labels to the text file 'test_labels.txt',
## with ONE LABEL PER LINE


if __name__ == "__main__":
    training_file = "data/complex_words_training.txt"
    development_file = "data/complex_words_development.txt"
    test_file = "data/complex_words_test_unlabeled.txt"

    # train_data = load_file(training_file)

    ngram_counts_file = "ngram_counts.txt.gz"
    counts = load_ngram_counts(ngram_counts_file)

    models = [
        ("Majority Class", lambda train_file, dev_file, counts: majority_class(train_file, dev_file)),
        ("Word Length", lambda train_file, dev_file, counts: word_length_threshold(train_file, dev_file)),
        ("Word Frequency", word_frequency_threshold),
        ("Naive Bayes", naive_bayes),
        ("Logistic Regression", logistic_regression),
    ]

    print("Available models:")
    for idx, model in enumerate(models):
        print(f"{idx+1}. {model[0]}")
    selections = [*map(
        lambda sel: int(sel.strip()) - 1,
        input("Enter your classifier selections separated by comma, e.g. 1,2,3: ").split(","))]

    for selection in selections:
        model_name, model_caller = models[selection]
        print(f"\n================\nEvaluating Model: {model_name}")
        model_caller(training_file, development_file, counts)
