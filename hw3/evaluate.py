#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
from argparse import ArgumentParser
from matplotlib import pyplot as plt


def warn(*args, **kwargs): pass
import warnings; warnings.warn = warn


parser = ArgumentParser()

parser.add_argument("-p", "--predicted", dest = "pred_path",
    required = True, help = "path to your model's predicted labels file")

parser.add_argument("-d", "--development", dest = "dev_path",
    required = True, help = "path to the development labels file")

parser.add_argument("-c", "--confusion", dest = "show_confusion",
    action = "store_true", help = "show confusion matrix")
parser.add_argument('-u', '--unknown', dest='unknown',
                    action='store_true')
parser.add_argument('-e', '--error', dest='error',
                    action='store_true')
args = parser.parse_args()


pred = pd.read_csv(args.pred_path, index_col = "id")
dev  = pd.read_csv(args.dev_path,  index_col = "id")

pred.columns = ["predicted"]
dev.columns  = ["actual"]

data = dev.join(pred)


if args.show_confusion:
    
    data["count"] = 1
    counts = data.groupby(["actual", "predicted"]).count().reset_index()
    confusion = counts[counts.actual != counts.predicted].reset_index(drop = True)
    tags = sorted(list(set([l.rstrip().split(',', 1)[1].strip('"') for l in open('data/train_y.csv').readlines()[1:]]))) + ['<s>']

    cmat = np.zeros([len(tags),len(tags)], dtype=int)
    for index, row in confusion.iterrows():
        x = tags.index(row['actual'])
        y = tags.index(row['predicted'])
        cmat[x][y] = row['count']
        if tags[x] == 'NN':
            print(tags[y], row['count'])

    xtags = [x for x in tags]
    ytags = [x for x in tags]

    mi = 50
    val_cols = ~np.all(cmat < mi, axis=1)
    xtags = [tags[i] for i in range(len(tags)) if val_cols[i]]
    cmat = cmat[val_cols, :]
    val_rows = ~np.all(cmat < mi, axis=0)
    ytags = [tags[i] for i in range(len(tags)) if val_rows[i]]    
    cmat = cmat[:, val_rows]
    
    fig, ax = plt.subplots(1,1)
    ax.matshow(cmat, cmap='coolwarm', origin='lower')
    ax.set_xticks(np.arange(cmat.shape[1]))
    ax.set_yticks(np.arange(cmat.shape[0]))
    ax.set_xticklabels(ytags)
    ax.set_yticklabels(xtags)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix of Tags with Significant Error')
    plt.gca().xaxis.tick_bottom()
    plt.show()
    exit()
    
else:

    print("Mean F1 Score:", f1_score(
        data.actual,
        data.predicted,
        average = "weighted"
    ))

    
if args.unknown:
    trn_inds = [int(l.rstrip().split(',')[0]) for l in open('data/train_x.csv').readlines()[1:]]
    dev_inds = [int(l.rstrip().split(',')[0]) for l in open('data/dev_x.csv').readlines()[1:]]    
    trn_words = [l.rstrip().split(',')[1].strip('"') for l in open('data/train_x.csv').readlines()[1:]]
    dev_words = [l.rstrip().split(',')[1].strip('"') for l in open('data/dev_x.csv').readlines()[1:]]
    
    dev_tags = [l.rstrip().split(',')[1] for l in open('data/dev_y.csv').readlines()[1:]]
    pred_tags = [l.rstrip().split(',')[1] for l in open(args.pred_path).readlines()[1:]]
    
    trn_set = set(trn_words)
    dev_set = set(dev_words)
    
    known = []
    unknown = []
    for ind in dev_inds:
        if dev_words[ind] in trn_set:
            known.append(ind)
        else:
            unknown.append(ind)

    known_actual = [dev_tags[i] for i in known]
    known_pred = [pred_tags[i] for i in known]
    unknown_actual = [dev_tags[i] for i in unknown]
    unknown_pred = [pred_tags[i] for i in unknown]

    unknown_tags = sorted(list(set(unknown_actual)))
    freqs = []
    for t in unknown_tags:
        freqs.append(unknown_actual.count(t))
    print("Known F1 Score:", f1_score(
        known_actual,
        known_pred,
        average = "weighted"
    ))
    print("Unknown F1 Score:", f1_score(
        unknown_actual,
        unknown_pred,
        average = "weighted"
    ))

if args.error:
    trn_inds = [int(l.rstrip().split(',')[0]) for l in open('data/train_x.csv').readlines()[1:]]
    dev_inds = [int(l.rstrip().split(',')[0]) for l in open('data/dev_x.csv').readlines()[1:]]    
    trn_words = [l.rstrip().split(',')[1].strip('"') for l in open('data/train_x.csv').readlines()[1:]]
    dev_words = [l.rstrip().split(',')[1].strip('"') for l in open('data/dev_x.csv').readlines()[1:]]

    dev_tags = [l.rstrip().split(',')[1] for l in open('data/dev_y.csv').readlines()[1:]]
    pred_tags = [l.rstrip().split(',')[1] for l in open(args.pred_path).readlines()[1:]]

    trn_set = set(trn_words)
    dev_set = set(dev_words)
    
    known = []
    unknown = []
    for ind in dev_inds:
        if dev_words[ind] in trn_set:
            known.append(ind)
        else:
            unknown.append(ind)
    
    for i in dev_inds:
        if dev_tags[i] != pred_tags[i] and i in known:
            print(dev_words[i], dev_tags[i], pred_tags[i])
            
    
