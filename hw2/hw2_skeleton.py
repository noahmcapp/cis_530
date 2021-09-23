#!/usr/local/bin/python3.9

# system modules
import os
import sys
import time
import gzip
import argparse
from collections import defaultdict, Counter

# installed modules
import numpy as np
from nltk.corpus import wordnet as wn
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


# plotting modules
from matplotlib import pyplot as plt

# custom modules
from syllables import count_syllables

#### 1. Evaluation Metrics ####

## Input: y_pred, a list of length n with the predicted labels,
## y_true, a list of length n with the true labels

# converts predictions into TP, FN, FP, and TN
# NOTE: based on https://en.wikipedia.org/wiki/Receiver_operating_characteristic
def score(y_pred, y_true):
    results = {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
    for hyp, ref in zip(y_pred, y_true):
        if ref == 1:
            if hyp == 1:
                results["tp"] += 1
            else:
                results["fn"] += 1
        else:
            if hyp == 1:
                results["fp"] += 1
            else:
                results["tn"] += 1
    return results


#
# end of score

# calculates the precision of the predicted labels
def get_precision(y_pred, y_true):
    results = score(y_pred, y_true)
    total_pred_pos = float(results["tp"] + results["fp"])
    return results["tp"] / total_pred_pos if total_pred_pos else 1


#
# end of get_precision

## calculates the recall of the predicted labels
def get_recall(y_pred, y_true):
    results = score(y_pred, y_true)
    total_pos = results["tp"] + results["fn"]
    return results["tp"] / total_pos if total_pos else 1


#
# end of get_recall

## calculates the f-score of the predicted labels
def get_fscore(y_pred, y_true):
    precision = get_precision(y_pred, y_true)
    recall = get_recall(y_pred, y_true)
    den = precision + recall
    return 2 * (precision * recall) / den if den else 0


#
# end of get_fscore

# generates precision, recall, and fscore metrics
def get_metrics(y_pred, y_true):
    precision = get_precision(y_pred, y_true)
    recall = get_recall(y_pred, y_true)
    fscore = get_fscore(y_pred, y_true)
    return precision, recall, fscore


#
# end of get_metrics

# prints out the precision, recall, and f-score of the predictions
def test_predictions(y_pred, y_true):
    precision, recall, fscore = get_metrics(y_pred, y_true)
    print("** Results **")
    print("\tPrecision: %.4f" % precision)
    print("\tRecall: %.4f" % recall)
    print("\tF1 Score: %.4f" % fscore)
    print("** End of Results ** \n")


#
# end of test_predictions


#### 2. Complex Word Identification ####

## Loads in the words and labels of one of the datasets
def load_file(data_file, feat_set=0):
    # if feat_set = 1, we also return the sentence
    words = []
    labels = []
    sentences = []
    with open(data_file, 'rt', encoding="utf8") as f:
        i = 0
        for line in f:
            if i > 0:
                line_split = line[:-1].split("\t")
                words.append(line_split[0].lower())
                labels.append(int(line_split[1]))
                sentences.append(line_split[3].lower())
            i += 1
    if feat_set == 0:
        return words, labels
    return words, labels, sentences

def load_test_file(data_file, feat_set=0):
    words, sentences = [], []
    with open(data_file, 'rt', encoding='utf-8') as f:
        i = 0
        for line in f:
            if i > 0:
                line_split = line[:-1].split('\t')
                words.append(line_split[0].lower())
                sentences.append(line_split[1])
            i += 1
    if feat_set == 0:
        return words
    else:
        return words, sentences

### 2.1: A very simple baseline

## Labels every word complex

# generates list of length n of 1's
def all_complex_feature(words):
    feats = [1] * len(words)
    return feats


#
# end of all_complex_feature

# classifies all words as complex
def all_complex(data_file, pr=False):
    # load the data
    words, y_true = load_file(data_file)

    # generate the features
    y_pred = all_complex_feature(words)

    # calculate the results
    test_predictions(y_pred, y_true)
    performance = get_metrics(y_pred, y_true)

    # generate the PR curve
    if pr:
        gen_pr('majority_pr', np.array(y_pred), y_true)

    return performance


#
# end of all_complex

# algorithm for just selecting the majority class as a baseline
def majority_class(training_file, development_file):
    print(' ---> Processing Training Data <--- ')
    all_complex(training_file)
    print(' ---> Processing Development Data <--- ')
    all_complex(development_file, pr=True)


#
# end of majority_class

### 2.2: Word length thresholding

# gets whether or not each word length is above the threshold
def length_threshold_feature(words, threshold):
    feats = [1 if len(word) >= threshold else 0 for word in words]
    return feats


#
# end of length_threshold_feature

# classifies each word based on the length, assuming longer means more complex
# NOTE: the threshold used for dev data is the threshold which yields the
#       highest F1 Score for the training data
def word_length_threshold(training_file, development_file):
    # load the training and development data
    words_trn, y_true_trn = load_file(training_file)
    words_dev, y_true_dev = load_file(development_file)

    # loop through thresholds
    mx_thresh = max([len(x) for x in words_trn])
    mi_thresh = min([len(x) for x in words_trn])
    pr_x, pr_y, scores = [], [], []
    thresholds = list(range(mx_thresh - 1, mi_thresh - 1, -1))
    print(' ---> Testing Thresholds Between %d and %d <--- ' % (mi_thresh, mx_thresh))
    for thresh in thresholds:
        # generate predictions for each threshold
        y_pred = length_threshold_feature(words_trn, thresh)

        # calculate and store the performance
        p, r, f = get_metrics(y_pred, y_true_trn)

        # store the data for a PR curve
        pr_x.append(r)
        pr_y.append(p)
        scores.append(f)
    #
    # end of threshold loop

    # get the optimal threshold from the training results
    optimal_ind = np.argmax(scores)
    thresh = thresholds[optimal_ind]
    print(' ---> Calculated Optimal Length Threshold: %d <--- ' % thresh)

    # write out the pr curve
    write_data('length_pr', pr_x, pr_y)

    # generate the final predictions
    print(' -----> Generating Training Predictions <----- ')
    y_pred_trn = length_threshold_feature(words_trn, thresh)
    test_predictions(y_pred_trn, y_true_trn)
    print(' -----> Generating Development Predictions <----- ')
    y_pred_dev = length_threshold_feature(words_dev, thresh)
    test_predictions(y_pred_dev, y_true_dev)

    # calculate the results
    training_performance = get_metrics(y_pred_trn, y_true_trn)
    development_performance = get_metrics(y_pred_dev, y_true_dev)

    return training_performance, development_performance


#
# end of word_length_threshold

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

# gets whether or not each word frequency is below the threshold
def frequency_threshold_feature(words, threshold, counts):
    feats = [1 if counts[word] < threshold else 0 for word in words]
    return feats


#
# end of frequency_threshold_feature

def frequency_cdf(words, counts, base):
    freqs = np.array([counts[x] for x in words])
    freqs = np.sort(freqs)
    cdf = np.array(range(len(freqs))) / float(len(freqs))
    fig, ax = plt.subplots(1, 1)
    ax.plot(freqs, cdf)
    ax.set_xscale('log', base=base)
    ax.set_xlabel('Log Spaced Word Frequencies')
    ax.set_title('CDF of Unigram Frequencies')
    plt.savefig('out/plots/frequency_cdf.png')


# classifies each word based on the frequency of it's unigram, assuming less frequent = more complex
# NOTE: the threshold used for development data is the threshold which yields the
#       highest F1 Score for the training data
def word_frequency_threshold(training_file, development_file, counts, n_thresh):
    # load the training and development data
    words_trn, y_true_trn = load_file(training_file)
    words_dev, y_true_dev = load_file(development_file)

    # plot the cdf to inform the threshold sampling
    base = 10
    frequency_cdf(words_trn, counts, base)

    # generate a log spaced set of thresholds to test
    mx_thresh = max(counts.values())
    thresholds = np.logspace(np.log(1) / np.log(base),
                             np.log(mx_thresh) / np.log(base),
                             num=n_thresh, base=base)

    # loop through thresholds
    print(' ---> Testing Thresholds up to %.2e <--- ' % (mx_thresh))
    t0 = time.time()
    pr_x, pr_y, scores = [], [], []
    for thresh in thresholds:
        # generate predictions for each threshold
        y_pred = frequency_threshold_feature(words_trn, thresh, counts)

        # calculate and store the peprformance
        p, r, f = get_metrics(y_pred, y_true_trn)

        # store the data for a PR curve
        pr_x.append(r)
        pr_y.append(p)
        scores.append(f)
    #
    # end of threshold loop

    t1 = time.time()
    print(' ---> Tested %.2e Thresholds in %.2f Seconds <--- ' % \
          (n_thresh, t1 - t0))

    # write out the pr curve
    write_data('freq_pr', pr_x, pr_y)

    # get the optimal threshold from the training results
    optimal_ind = np.argmax(scores)
    thresh = thresholds[optimal_ind]
    print(' ---> Calculated Optimal Frequency Threshold: %d <--- ' % thresh)

    # generate the final predictions
    print(' -----> Generating Training Predictions <----- ')
    y_pred_trn = frequency_threshold_feature(words_trn, thresh, counts)
    test_predictions(y_pred_trn, y_true_trn)
    print(' -----> Generating Development Predictions <----- ')
    y_pred_dev = frequency_threshold_feature(words_dev, thresh, counts)
    test_predictions(y_pred_dev, y_true_dev)

    # calculate the results
    training_performance = get_metrics(y_pred_trn, y_true_trn)
    development_performance = get_metrics(y_pred_dev, y_true_dev)

    return training_performance, development_performance


#
# end of word_frequency_threshold

# generates feature vectors for various word features
def length_feature(words):
    return np.array([len(x) for x in words])[:, None].astype(float)


def frequency_feature(words, counts):
    return np.array([counts[x] for x in words])[:, None].astype(float)


def syllables_feature(words):
    return np.array([count_syllables(x) for x in words])[:, None].astype(float)


def synonyms_feature(words):
    verb_counts = [len(wn.synsets(x, wn.VERB)) for x in words]
    noun_counts = [len(wn.synsets(x, wn.NOUN)) for x in words]
    adj_counts = [len(wn.synsets(x, wn.ADJ)) for x in words]
    return np.array([verb_counts, noun_counts, adj_counts]).transpose().astype(float)


def sentence_length(sentences):
    return np.array([len(sent) for sent in sentences])[:, None].astype(float)


def sentence_word_count(sentences):
    return np.array([len(sent.split()) for sent in sentences])[:, None].astype(float)


def sentence_avg_word_length(sentences):
    def avg_word_length(sentence):
        tokens = sentence.split()
        return sum([*map(len, tokens)]) / len(tokens)
    return np.array([*map(avg_word_length, sentences)])[:, None].astype(float)


def sentence_avg_word_freq(sentences):
    def avg_word_freq(sentence):
        tokens = sentence.split()
        counter = Counter(tokens)
        return sum(counter.values()) / len(counter)
    return np.array([*map(avg_word_freq, sentences)])[:, None].astype(float)

#
# end of feature vectors

# generates a set of features built from various feature vectors
def gen_feats(words, counts, feat_set=0, sentences=None):
    length = length_feature(words)
    frequency = frequency_feature(words, counts)
    if feat_set == 0:
        feats = np.concatenate([length, frequency], axis=1)
    else:
        syllables = syllables_feature(words)
        synonyms = synonyms_feature(words)

        sent_len = sentence_length(sentences)
        sent_word_count = sentence_word_count(sentences)
        sent_avg_freq = sentence_avg_word_freq(sentences)
        sent_avg_word_length = sentence_avg_word_length(sentences)

        feats = np.concatenate([
            length, frequency, syllables, synonyms,
            sent_len, sent_avg_freq, sent_word_count, sent_avg_word_length],
                               axis=1)
    #
    # end of feature set selection

    return feats


#
# end of load_feats

# standardizes the trn and dev features by the trn mean and stdev
def norm(trn, dev):
    mu = np.mean(trn, axis=0)
    sigma = np.std(trn, axis=0)
    trn = (trn - mu) / sigma
    dev = (dev - mu) / sigma
    return trn, dev


#
# end of norm

# loads the trn and dev data, then generates a feature set
def load_feats(training_file, development_file, counts, feat_set):
    # load the training and development data
    if feat_set == 0:
        words_trn, y_true_trn = load_file(training_file, feat_set)
        words_dev, y_true_dev = load_file(development_file, feat_set)

        # generate the feature set
        feats_trn = gen_feats(words_trn, counts, feat_set)
        feats_dev = gen_feats(words_dev, counts, feat_set)
    else:  # feat_set = 1
        words_trn, y_true_trn, sent_trn = load_file(training_file, feat_set)
        words_dev, y_true_dev, sent_dev = load_file(development_file, feat_set)

        # generate the feature set
        feats_trn = gen_feats(words_trn, counts, feat_set, sent_trn)
        feats_dev = gen_feats(words_dev, counts, feat_set, sent_dev)

    # normalize the features
    feats_trn, feats_dev = norm(feats_trn, feats_dev)

    return feats_trn, y_true_trn, feats_dev, y_true_dev


#
# end of load_feats

# trains and decodes an sklearn model
def run_sk_model(clf, feats_trn, y_true_trn, feats_dev, y_true_dev, name):
    # train the model
    print(' ---> Training the Model <--- ')
    clf.fit(feats_trn, y_true_trn)

    # generate the final predictions
    print(' -----> Generating Training Predictions <----- ')
    y_pred_trn = clf.predict(feats_trn)
    test_predictions(y_pred_trn, y_true_trn)
    print(' -----> Generating Development Predictions <----- ')
    y_pred_dev = clf.predict(feats_dev)
    test_predictions(y_pred_dev, y_true_dev)

    if isinstance(clf, GridSearchCV):
        print("Model best params:", clf.best_params_)

    # calculate the results
    training_performance = get_metrics(y_pred_trn, y_true_trn)
    development_performance = get_metrics(y_pred_dev, y_true_dev)

    # generate and write the pr curve
    probs_dev = clf.predict_proba(feats_dev)[:, 1]

    return training_performance, development_performance


#
# end of run_sk_model

## Trains a Naive Bayes classifier using length and frequency features
def naive_bayes(training_file, development_file, counts, feat_set=0):
    # load the features
    feats_trn, y_true_trn, feats_dev, y_true_dev = load_feats(
        training_file, development_file, counts, feat_set)

    # initialize the model
    clf = GaussianNB()

    # train and decode the model
    return run_sk_model(
        clf, feats_trn, y_true_trn, feats_dev, y_true_dev, 'naive_bayes_pr')


#
# end of naive_bayes

### 2.5: Logistic Regression

## Trains a Naive Bayes classifier using length and frequency features
def logistic_regression(training_file, development_file, counts, feat_set=0):
    # load the features
    feats_trn, y_true_trn, feats_dev, y_true_dev = load_feats(
        training_file, development_file, counts, feat_set)

    # initialize the model
    clf = LogisticRegression()

    # train and decode the model
    return run_sk_model(
        clf, feats_trn, y_true_trn, feats_dev, y_true_dev,
        'logistic_regression_pr')


#
# end of logistic_regression

#
# end of file
