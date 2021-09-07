#!/usr/local/bin/python3.9

# system modules
import os
import sys
import gzip
import argparse
from collections import defaultdict

# installed modules
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()
sns.set_style("whitegrid")
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=sns.color_palette("Set2"))

#### 1. Evaluation Metrics ####

## Input: y_pred, a list of length n with the predicted labels,
## y_true, a list of length n with the true labels

# converts predictions into TP, FN, FP, and TN
# NOTE: based on https://en.wikipedia.org/wiki/Receiver_operating_characteristic
def score(y_pred, y_true):
    tp, fn, fp, tn = 0, 0, 0, 0
    for hyp, ref in zip(y_pred, y_true):
        if ref == 1:
            if hyp == 1:
                tp += 1
            else:
                fn += 1
        else:
            if hyp == 1:
                fp += 1
            else:
                tn += 1
    return tp, fn, fp, tn
#
# end of score

# calculates the precision of the 
def get_precision(y_pred, y_true):
    tp, fn, fp, tn = score(y_pred, y_true)
    if tp + fp == 0:
        return 1
    else:
        return (tp) / float(tp + fp)
#
# end of get_precision
    
## Calculates the recall of the predicted labels
def get_recall(y_pred, y_true):
    tp, fn, fp, tn = score(y_pred, y_true)
    if tp + fn == 0:
        return 1
    else:
        return (tp) / float(tp + fn)
#
# end of get_precision

# TODO: this is bad coding, since we call score a bunch of times
#        however functions might need these arguments for gradescope?
## Calculates the f-score of the predicted labels
def get_fscore(y_pred, y_true):
    tp, fn, fp, tn = score(y_pred, y_true)    
    precision = get_precision(y_pred, y_true)
    recall = get_recall(y_pred, y_true)
    if precision + recall == 0:
        return 0
    else:
        return 2 * (precision * recall) / (precision + recall)
#
# end of get_fscore

def get_metrics(y_pred, y_true):
    precision = get_precision(y_pred, y_true)
    recall = get_recall(y_pred, y_true)
    fscore = get_fscore(y_pred, y_true)
    return precision, recall, fscore
#
# end of get_metrics

# calculates the avg. precision (AP) or area under the curve (AUC)
#  of the PR curve
def get_ap(pr_x, pr_y):
    pr_x, pr_y = np.array(pr_x), np.array(pr_y)
    return np.sum(pr_y[1:] * (pr_x[1:] - pr_x[:-1]))
#
# end of get_ap

# prints out the precision, recall, and f-score of the predictions
def test_predictions(y_pred, y_true):
    precision, recall, fscore = get_metrics(y_pred, y_true)
    print("** Results **")
    print("Precision: %.4f" % precision)
    print("Recall: %.4f" % recall)
    print("F1 Score: %.4f" % fscore)
    print("** End of Results ** \n")
#
# end of test_predictions


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

## Labels every word complex

# generates list of length n of 1's
def all_complex_feature(words):
    feats = [1] * len(words)
    return feats
#
# end of all_complex_feature

# classifies all words as complex
def all_complex(data_file):

    # load the data
    words, y_true = load_file(data_file)

    # generate the features
    y_pred = all_complex_feature(words)

    # calculate the results
    test_predictions(y_pred, y_true)
    performance = get_metrics(y_pred, y_true)
    return performance
#
# end of all_complex

### 2.2: Word length thresholding

# calculates the length of each word as the feature vector
def length_threshold_feature(words, threshold):
    feats = [1 if len(word) >= threshold else 0 for word in words]
    return feats
#
# end of length_threshold_feature

# classifies each word based on the length, assuming longer means more complex
# NOTE: the threshold used for development data is the threshold which yields the
#       highest F1 Score for the training data
def word_length_threshold(training_file, development_file, thresh=0):

    # load the training and development data
    words_trn, y_true_trn = load_file(training_file)
    words_dev, y_true_dev = load_file(development_file)

    # loop through thresholds
    mx_thresh = max([len(x) for x in words_trn])
    mi_thresh = min([len(x) for x in words_trn])    
    pr_x, pr_y, scores = [], [], []
    thresholds = list(range(mx_thresh-1, mi_thresh-1, -1))
    print(' ---> Testing Thresholds Between %d and %d' % (mi_thresh, mx_thresh))
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

    # calculate the area under the curve
    ap = get_ap(pr_x, pr_y)
    
    # get the optimal threshold from the training results
    mx_rec_ind = np.argmax(pr_x)    
    mx_prec_ind = np.argmax(pr_y)
    optimal_ind = np.argmax(scores)
    thresh = thresholds[optimal_ind]
    print(' ---> Calculated Optimal Length Threshold: %d <--- ' % thresh)

    # plot the PR curve
    fig, axs = plt.subplots(1,1)
    ax = axs
    ax.plot(pr_x, pr_y)
    ax.plot(pr_x[optimal_ind], pr_y[optimal_ind], '.', markersize=10,
               label='Max. F1 -- Threshold = %d' % thresh)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Length Threshold Sweep -- AUC = %.4f' % ap)
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.legend()
    plt.savefig('plots/length_threshold.png')
    
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

## Make feature matrix for word_frequency_threshold
def frequency_threshold_feature(words, threshold, counts):
    pass
def word_frequency_threshold(training_file, development_file, counts):
    ## YOUR CODE HERE
    training_performance = [tprecision, trecall, tfscore]
    development_performance = [dprecision, drecall, dfscore]
    return training_performance, development_performance

### 2.4: Naive Bayes
        
## Trains a Naive Bayes classifier using length and frequency features
def naive_bayes(training_file, development_file, counts):
    ## YOUR CODE HERE
    training_performance = (tprecision, trecall, tfscore)
    development_performance = (dprecision, drecall, dfscore)
    return development_performance

### 2.5: Logistic Regression

## Trains a Naive Bayes classifier using length and frequency features
def logistic_regression(training_file, development_file, counts):
    ## YOUR CODE HERE    
    training_performance = (tprecision, trecall, tfscore)
    development_performance = (dprecision, drecall, dfscore)
    return development_performance

### 2.7: Build your own classifier

## Trains a classifier of your choosing, predicts labels for the test dataset
## and writes the predicted labels to the text file 'test_labels.txt',
## with ONE LABEL PER LINE

def main(argv):

    # define location of data
    training_file = "data/complex_words_training.txt"
    development_file = "data/complex_words_development.txt"
    test_file = "data/complex_words_test_unlabeled.txt"
    ngram_counts_file = "ngram_counts.txt.gz"

    # parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-algo', type=str, required=True,
                        help="name of algorithm to run",
                        choices=['basic', 'length', 'ngram'])
    args = parser.parse_args()

    # make sure the directory is setup properly
    if not os.path.exists('plots/'):
        os.makedirs('plots')
    
    if args.algo == 'basic':
        print(' -> Running Simple All Complex Baseline <- ')
        print(' ---> Processing Training Data <--- ')
        all_complex(training_file)
        print(' ---> Processing Development Data <--- ')
        all_complex(development_file)
    elif args.algo == 'length':
        print(' -> Running Word Length Thresholder <- ')
        word_length_threshold(training_file, development_file)
    elif args.algo == 'ngram':
        counts = load_ngram_counts(ngram_counts_file)
        print(len(counts))
        word_frequency_threshold(training_file, development_file, counts)
    else:
        pass
    #
    # end of algorithms
#
# end of main

if __name__ == "__main__":
    main(sys.argv)
    
#
# end of file
