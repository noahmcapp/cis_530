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


def run():
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    from models import CharRNNClassify
    
    baseDir = "/content/gdrive/My Drive/cis530_hw6"

    # read data
    X_train, y_train = readData(baseDir, True)
    X_val, y_val = readData(baseDir, False)

    # train
    n_iters = 100000

    model = CharRNNClassify()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters())

    train_loss = []
    val_loss = []
    iters = []
    loss_sum = 0
    for iter in range(1, n_iters + 1):
        model.train()
        loss_sum += trainOneEpoch(model, criterion, optimizer, X_train, y_train)
        if iter and iter % 1000 == 0:
            # save accumulated train loss
            iters.append(iter)
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
    torch.save(model.state_dict(), baseDir + "/model_classify")

    # evaluate
    model.eval()
    with torch.no_grad():
        # calculate accuracy on validation set
        val_acc = calculateAccuracy(model, X_val, y_val)
        print(val_acc)
        # create confusion matrix
        confusions = torch.zeros(len(languages), len(languages))
        for _ in range(10000):
            word, lang_idx = random_training_pair(X_train, y_train)
            line_tensor = line_to_tensor(word)
            output = evaluate(model, line_tensor)
            _, guess_idx = category_from_output(output)
            confusions[lang_idx][guess_idx] += 1
        for i in range(len(languages)):
            confusions[i] = confusions[i] / confusions[i].sum()

    ### plotting
    # Set up confusion plot
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

    # Set up loss plot
    iters = [i for i in range(200, 100000, 1000)]
    # plotting train loss
    plt.plot(iters, train_loss, label="Train loss")
    # plotting validation loss
    plt.plot(iters, val_loss, label="Validation loss")
    plt.xlabel('iterations')
    # Set the y axis label of the current axis.
    plt.ylabel('Loss')
    # Set a title of the current axes.
    plt.title('Train and Validation Loss')
    # show a legend on the plot
    plt.legend()
    # Display a figure.
    plt.show()

    # load saved model to predict test

    model = CharRNNClassify()
    model.load_state_dict(torch.load(baseDir + '/model_classify'))
    model.eval()

    test_cities = []
    with open("/content/gdrive/My Drive/cis530_hw6/cities_test.txt", "r", errors='ignore') as f:
        for line in f:
            test_cities.append(unicodeToAscii(line.strip()))
    test_cities = np.array(test_cities, dtype='object')

    test_pred = predict(model, test_cities, None)
    lang_pred = [languages[idx] for idx in test_pred]
    with open(baseDir + "/labels.txt", "w") as f:
        f.write('\n'.join(lang_pred))

    return train_loss, val_loss, val_acc, confusions