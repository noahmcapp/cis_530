# main_classify.py
import codecs
import math
import glob
import random
import string
import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score
import unicodedata

'''
Don't change these constants for the classification task.
You may use different copies for the sentence generation model.
'''
languages = ["af", "cn", "de", "fi", "fr", "in", "ir", "pk", "za"]
all_letters = string.ascii_letters + " .,;'"

'''
Returns the words of the language specified by reading it from the data folder
Returns the validation data if train is false and the train data otherwise.
Return: A nx1 array containing the words of the specified language
'''


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def getWords(baseDir, lang, train=True):
    folder = "train" if train else "val"
    filename = baseDir + "/data/" + folder + "/" + lang + "_" + folder + ".txt"
    words = []
    with open(filename, "r", encoding='utf-8', errors='ignore') as f:
        for line in f:
            words.append(unicodeToAscii(line.strip()))
    return np.array(words, dtype='object')


'''
Returns a label corresponding to the language
For example it returns an array of 0s for af
Return: A nx1 array as integers containing index of the specified language in the "languages" array
'''


def getLabels(lang, length):
    idx = languages.index(lang)
    return np.array([idx] * length)


'''
Returns all the laguages and labels after reading it from the file
Returns the validation data if train is false and the train data otherwise.
You may assume that the files exist in baseDir and have the same names.
Return: X, y where X is nx1 and y is nx1
'''


def readData(baseDir, train=True):
    X = np.array([], dtype='object')
    y = np.array([], dtype='int')
    for lang in languages:
        words = getWords(baseDir, lang, train)
        X = np.concatenate([X, words], axis=None)
        y = np.concatenate([y, getLabels(lang, words.size)])
    return X, y


'''
Convert a line/word to a pytorch tensor of numbers
Refer the tutorial in the spec
Return: A tensor corresponding to the given line
'''


def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, len(all_letters))
    for idx, char in enumerate(line):
        tensor[idx][0][all_letters.find(char)] = 1
    return tensor


'''
Returns the category/class of the output from the neural network
Input: Output of the neural networks (class probabilities)
Return: A tuple with (language, language_index)
        language: "af", "cn", etc.
        language_index: 0, 1, etc.
'''


def category_from_output(output):
    top_n, top_i = output.topk(1)
    language_idx = top_i[0].item()
    return languages[language_idx], language_idx


'''
Get a random input output pair to be used for training 
Refer the tutorial in the spec
'''


def random_training_pair(X, y):
    assert X.size == y.size
    idx = random.randint(0, X.size - 1)
    return X[idx], y[idx]


'''
Input: trained model, a list of words, a list of class labels as integers
Output: a list of class labels as integers
'''


def evaluate(model, line_tensor):
    hidden = model.initHidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)
    return output


def predict(model, X, y):
    pred = []
    model.eval()
    for line in X:
        with torch.no_grad():
            output = evaluate(model, line_to_tensor(line))
            lang, lang_idx = category_from_output(output)
            pred.append(lang_idx)
    return pred


'''
Input: trained model, a list of words, a list of class labels as integers
Output: The accuracy of the given model on the given input X and target y
'''


def calculateAccuracy(model, X, y):
    y_pred = predict(model, X, None)
    correct = 0
    for pred, actual in zip(y_pred, y):
        if pred == actual:
            correct += 1
    return correct / len(y_pred)


'''
Train the model for one epoch/one training word.
Ensure that it runs within 3 seconds.
Input: X and y are lists of words as strings and classes as integers respectively
Returns: You may return anything
'''


def trainOneEpoch(model, criterion, optimizer, X, y):
    word, lang_idx = random_training_pair(X, y)
    hidden = model.initHidden()
    model.zero_grad()
    line_tensor = line_to_tensor(word)
    category_tensor = torch.tensor([lang_idx], dtype=torch.long)

    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)
    loss = criterion(output, category_tensor)
    loss.backward()
    optimizer.step()
    return loss.item()


'''
Use this to train and save your classification model. 
Save your model with the filename "model_classify"
'''


def run(lr=0.001, hidden_size=110, track_loss=False, track_confusion=False, save_model=False):
    baseDir = "/content/gdrive/My Drive/cis530_hw6"

    # read data
    X_train, y_train = readData(baseDir, True)
    X_val, y_val = readData(baseDir, False)

    # train
    n_iters = 100000

    model = CharRNNClassify(hidden_size=hidden_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    train_loss = []
    val_loss = []
    loss_sum = 0
    for iter in range(1, n_iters + 1):
        model.train()
        loss_sum += trainOneEpoch(model, criterion, optimizer, X_train, y_train)
        if track_loss and iter and iter % 1000 == 0:
            # save accumulated train loss
            train_loss.append(loss_sum / 1000)
            loss_sum = 0

            # calculate validation loss
            vloss = 0
            model.eval()
            with torch.no_grad():
                for idx in range(X_val.shape[0]):
                    word, lang_idx = X_val[idx], y_val[idx]
                    line_tensor = line_to_tensor(word)
                    output = evaluate(model, line_tensor)
                    category_tensor = torch.tensor([lang_idx], dtype=torch.long)
                    loss = criterion(output, category_tensor)
                    vloss += loss.item()
            val_loss.append(vloss / X_val.shape[0])

    # save model
    if save_model:
        torch.save(model.state_dict(), baseDir + "/model_classify")

    # evaluate
    model.eval()
    with torch.no_grad():
        # calculate accuracy on validation set
        val_acc = calculateAccuracy(model, X_val, y_val)
        print(val_acc)
        # create confusion matrix
        confusions = torch.zeros(len(languages), len(languages))
        if track_confusion:
            for _ in range(10000):
                word, lang_idx = random_training_pair(X_train, y_train)
                line_tensor = line_to_tensor(word)
                output = evaluate(model, line_tensor)
                _, guess_idx = category_from_output(output)
                confusions[lang_idx][guess_idx] += 1
            for i in range(len(languages)):
                confusions[i] = confusions[i] / confusions[i].sum()
    return train_loss, val_loss, val_acc, confusions


def plot_confusions_and_loss():
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    train_loss, val_loss, val_acc, confusions = run(track_loss=True, track_confusion=True)

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusions.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + languages, rotation=90)
    ax.set_yticklabels([''] + languages)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()


def plot_hidden_size_effect():
    import matplotlib.pyplot as plt

    hidden_size = [10, 50, 110, 150, 200, 250, 400]
    accuracy2 = []

    for hs in hidden_size:
        _, _, acc, _ = run(hidden_size=hs)
        accuracy2.append(acc)
    plt.xticks([*range(len(accuracy2))], hidden_size)
    plt.plot(accuracy2, label="Accuracy by hidden size")
    plt.show()


def plot_learning_rate_effect():
    import matplotlib.pyplot as plt

    learning_rate = [0.00001, 0.00005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    accuracy = []

    for lr in learning_rate:
        _, _, acc, _ = run(lr=lr)
        accuracy.append(acc)
    plt.xticks([*range(len(accuracy))], learning_rate)
    plt.plot(accuracy, label="Accuracy by learning rate")
    plt.show()
