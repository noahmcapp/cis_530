#!/usr/bin/env python3

# system modules
import os
import sys
import random
import string
import time
import math
import argparse

# installed modules
import unidecode
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

# CONSTANTS
ALL_CHARS = string.printable
NCHARS = len(ALL_CHARS)

TRN_SPLIT = 0.8
LR = 0.002

# this model takes in a character t-1 and generates a probability distribution for character t
# it uses an embedding to encode the character which connects a GRU (possible to have multiple layers)
# the GRU connects to the decoder which outputs the probability distribution of character t
class RNN(nn.Module): 
    def __init__(self, isize, hsize, osize, nlayers=1):
        super(RNN, self).__init__()

        self.isize = isize
        self.hsize = hsize
        self.osize = osize
        self.nlayers = nlayers

        self.lr = LR
        
        self.encoder = nn.Embedding(self.isize, self.hsize)
        self.gru = nn.GRU(self.hsize, self.hsize, self.nlayers)
        self.decoder = nn.Linear(self.hsize, self.osize)

        # initialize the optimizer and loss criterion
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
    #
    # end of constructor
    
    def forward(self, i, h):
        i = self.encoder(i.view(1,-1))
        o, h = self.gru(i.view(1,1,-1), h)
        o = self.decoder(o.view(1,-1))
        return o, h

    def init_hidden(self):
        return Variable(torch.zeros(self.nlayers, 1, self.hsize))

    # this function uses a single chunk of data to update the parameters
    #   and returns the loss of this chunk
    def train(self, x, y):

        # initialize the hidden layer and optimizer
        h = self.init_hidden()
        self.zero_grad()
        loss = 0

        # loop through the chunk
        for i in range(x.shape[0]):

            # accumulate the loss over the forward algorithm
            o, h = self(x[i], h)
            loss += self.criterion(o, y[i])

        # perform backprop and update parameters
        loss.backward()
        self.optimizer.step()

        # return the loss
        return loss.item() / x.shape[0]
    #
    # end of train

    def decode(self, primer, npred, temp):
        h = self.init_hidden()
        pred = primer        
        primer = to_tensor(primer)


        for i in range(len(primer) - 1):
            _, h = self(primer[i], h)
        #
        # end of primer
        
        x = primer[-1]
        for i in range(npred):
            o, h = self(x, h)

            o = o.data.view(-1).div(temp).exp()
            ind = torch.multinomial(o, 1)[0]

            p = ALL_CHARS[ind]
            pred += p
            x = to_tensor(p)
        #
        # end of prediction
        
        return pred
    #
    # end of decode
#
# end of RNN

def to_tensor(x):
    y = torch.zeros(len(x), 1).long()
    for i in range(len(x)):
        y[i] = ALL_CHARS.index(x[i])
    return Variable(y)
#
# end of to_tensor

class ChunkLoader:
    def __init__(self, chunk_len, data, nchunks, is_training=False):
        self.chunk_len = chunk_len
        self.data = data
        self.is_training = is_training
        self.get_inds()

    # TODO: chunks might need to be 0, 100, 200, 300 in validation?
    def get_inds(self):
        self.chunks = list(range(len(self.data)))
        self.n = len(self.chunks)        
        if self.is_training:
            random.shuffle(self.chunks)
                        
    def __next__(self):
        ind = self.chunks.pop()
        chunk = self.data[ind:ind + self.chunk_len]
        x = to_tensor(chunk[:-1])
        y = to_tensor(chunk[1:])
        return x, y

# loads all text from the given dataset .txt file
def load_data(fname):
    return unidecode.unidecode(open(fname).read())

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def run_train(model, nepochs, nchunks, chunk_loader, lr, ofname):

    # start training over the epochs
    start = time.time()
    losses = []
    avg_loss = 0
    width = 20
    for epoch in range(1, nepochs+1):
        print('Progress: [%s]' % (' ' * width), end="", flush=True)
        print('\b' * (width + 1), end="", flush=True)
        for i in range(nchunks):
            x, y = next(chunk_loader)

            # get the loss over this chunk
            loss = train(x, y, model, criterion, optimizer)

            # accumulate the loss over the epoch
            avg_loss += loss

            if i % int(nchunks/width) == 0:
                print('■', end='', flush=True)
        #
        # end of epoch

        # display the results of this epoch
        print('] - [%s (%d %d%%) %.4f]' % \
              (time_since(start), epoch, epoch / nepochs * 100, avg_loss/nchunks), flush=True)        
        
        # store the loss over this epoch
        losses.append(avg_loss / nchunks)
        avg_loss = 0
    #
    # end of training

    torch.save(model.state_dict(), ofname)

def main(argv):

    # TODO: parse the args
    chunk_len = 200
    nchunks = 100
    nepochs = 100
    
    hsize = 100
    nlayers = 1

    lr = 0.002
    
    fname = 'input.txt'

    ofname = 'models/model.dat'
    
    # load the data
    data = load_data(fname)

    # TODO: neeed to split in a more intelligent way??
    ind = int(len(data) * TRN_SPLIT)
    trn_data = data[:ind]
    val_data = data[ind:]

    # initialize the model
    trn_loader = ChunkLoader(chunk_len, trn_data, nchunks, is_training=True)
    val_loader = ChunkLoader(chunk_len, val_data, nchunks, is_training=False)
    model = RNN(NCHARS, hsize, NCHARS, nlayers)

    args = parse_args()
    
    # load or train the model
    if args.saved_model:
        model.load_state_dict(torch.load(ofname))        
    else:
        run_train(model, nepochs, nchunks, trn_loader, lr, ofname)

    # decode from the user's input
    while True:
        s = input("Enter primer string (leave blank to continue): ")
        if not s:
            break
        try:
            o = model.decode(s, args.npred, args.temp)
            print(o)            
        except Exception as e:
            print(e)
            print('Input Failed, try again.')
    #
    # end of decoding
#
# end of main

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True,
                        help="path to a .txt file to use as the dataset, this file is split into train/test datasets")    
    parser.add_argument('--saved_model', type=str, required=False, default='',
                        help="path to a saved *.dat model to use instead of training")
    parser.add_argument('--decode', action='store_true', default=False,
                        help="flag to decide if you want to decode the model on user input")
    parser.add_argument('--npred', type=int, default=100,
                        help="number of characters to decode after the primer")
    parser.add_argument('--temp', type=float, default=0.8,
                        help="temperature of model, higher means it takes more risks")
    return parser.parse_args()
#
# end of parse_args

if __name__ == '__main__':
    main(sys.argv)
