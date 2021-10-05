#!/usr/bin/env python

""" Contains the part of speech tagger class. """
import re
import numpy as np

from param_model_factory import get_emission_model, get_transition_model, ADD_K_EMISSION, ADD_K_TRANSITION
from pos_sentence import POSSentence


def load_data(sentence_file, tag_file=None):
    """Loads data from two files: one containing sentences and one containing tags.

    tag_file is optional, so this function can be used to load the test data.

    Suggested to split the data by the document-start symbol.

    """
    # Read lines from provided files
    word_lines = []
    tag_lines = []
    with open(sentence_file, 'r') as f:
        for line in f:
            word_lines.append(line.strip())
    if tag_file:
        with open(tag_file, 'r') as f:
            for line in f:
                tag_lines.append(line.strip())
    else:
        tag_lines = [None] * len(word_lines)

    # extract words and tags into sentences from the lines
    curr_words = []
    curr_tags = []
    sentences = []

    for word_line, tag_line in zip(word_lines, tag_lines):
        if word_line == 'id,word':
            continue
        word = re.sub(r'^\d+,', '', word_line)[1:-1]
        tag = re.sub(r'^\d+,', '', tag_line)[1:-1] if tag_file else None
        if word == '-DOCSTART-':
            if len(curr_words) != 0:
                sentence = POSSentence([*curr_words])
                if tag_file:
                    sentence.tags = [*curr_tags]
                sentences.append(sentence)
                curr_words = ['-DOCSTART-']
                curr_tags = [tag]
            else:
                curr_words.append('-DOCSTART-')
                if tag_file:
                    curr_tags.append(tag)
        else:
            curr_words.append(word)
            if tag_file:
                curr_tags.append(tag)

    sentence = POSSentence(curr_words)
    if tag_file:
        sentence.tags = curr_tags
    sentences.append(sentence)
    return sentences


def evaluate(data, model):
    """Evaluates the POS model on some sentences and gold tags.

    This model can compute a few different accuracies:
        - whole-sentence accuracy
        - per-token accuracy
        - compare the probabilities computed by different styles of decoding

    You might want to refactor this into several different evaluation functions.
    
    """
    q = model.q_mat()
    for sequence in data:
        e = model.e_mat(sequence)

        # TODO: all thse are giving the exact same results
        #       1) something wrong with decoding
        #       2) they all get same results cause of poor MLE?
        greedy_tags, greedy_score = model.inference(sequence, q, e, method='greedy')
        beam_tags, beam_score = model.inference(sequence, q, e, method='beam', k=4)        
        vit_tags, vit_score = model.inference(sequence, q, e, method='viterbi')        
        true_tags = sequence.tags

class POSTagger:
    def __init__(self):
        """Initializes the tagger model parameters and anything else necessary. """
        self.em = get_emission_model(ADD_K_EMISSION, **{
            'cutoff_percentile': 0.05,
            'k': 3
        })
        self.tm = get_transition_model(ADD_K_TRANSITION, **{
            'ngram': 3,
            'k': 3
        })

    def train(self, data):
        """Trains the model by computing transition and emission probabilities.

        You should also experiment:
            - smoothing.
            - N-gram models with varying N.
        
        """
        self.em.train(data)
        self.tm.train(data)
        self.tags = ['<s>'] + sorted(list(self.tm.avail_tags))
        
    def q_mat(self):
        n = len(self.tags)
        q = np.log(np.full((n,n,n), np.finfo(float).eps, dtype=float))
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    q[i,j,k] = self.tm.log_transit(self.tags[i], prev_tags=tuple((self.tags[k],self.tags[j])))
        return q
    
    def e_mat(self, sequence):
        n = len(sequence.words)
        m = len(self.tags)
        e = np.log(np.full((n,m), np.finfo(float).eps, dtype=float))                
        for i in range(n):
            for j in range(m):
                e[i][j] = self.em.log_emit(sequence.words[i], self.tags[j])
        return e

    def sequence_probability(self, sequence, tags):
        """Computes the probability of a tagged sequence given the emission/transition
        probabilities.
        """
        return 0.

    def inference(self, sequence, q, e, method='greedy', k=1):
        """Tags a sequence with part of speech tags.

        You should implement different kinds of inference (suggested as separate
        methods):

            - greedy decoding
            - decoding with beam search
            - viterbi
        """
        if method == 'greedy':
            inds, score = self.greedy(sequence, q, e)
        if method == 'beam':
            inds, score = self.beam(sequence, q, e, k)
        if method == 'viterbi':
            inds, score = self.viterbi(sequence, q, e)
        return [self.tags[i] for i in inds], score
    
    # generate a (len(sequence), self.ntags) array
    def get_emissions(sequence):
        return e

    def greedy(self, sequence, q, e):

        # initialize the trellis
        nseq = len(sequence.words)
        pi = np.log(np.full((nseq+2, 1), np.finfo(float).eps, dtype=float))
        bp = np.zeros_like(pi, dtype=int)
        bp[0] = 0 # NOTE: this assume <s> is at self.tags[0]
        bp[1] = 0
        pi[0] = 1
        pi[1] = 1
        for t in range(2, nseq+2):
            seq_ind = t-2
            x = (
                e[seq_ind,:][...,np.newaxis] + \
                q[:, bp[t-1], bp[t-2]] + \
                pi[t-1]
            )
            bp[t] = np.argmax(x)
            pi[t] = np.max(x)
        #
        # end of sequence

        # decode the bp and pi arrays to get the hidden sequence
        hidden_seq = np.zeros(nseq, dtype=int)
        hidden_seq[-1] = bp[nseq+2-1]
        for t in range(nseq-1, 0, -1):
            hidden_seq[t-1] = bp[t+2-1]
        return hidden_seq, pi[nseq+2-1]
    #
    # end of greedy

    def n_best(self, x, n):
        return np.array(np.unravel_index(np.argpartition(x.ravel(), -n)[-n:], x.shape))
    
    def beam(self, sequence, q, e, k=1):

        # initialize the trellis
        # TODO: check the paper, this def needs to be nseq, k, k
        nseq = len(sequence.words)
        pi = np.log(np.full((nseq+2, k, k), np.finfo(float).eps, dtype=float))
        bp = np.zeros_like(pi, dtype=int)
        bp[0,:] = 0 # NOTE: this assume <s> is at self.tags[0]
        bp[1,:] = 0
        pi[0,:] = 1
        pi[1,:] = 1
        for t in range(2, nseq+2):
            seq_ind = t-2
            x = (
                e[seq_ind,:][...,np.newaxis,np.newaxis] + \
                q[:, bp[t-1], bp[t-2]] + \
                pi[t-1]
            )
            inds = self.n_best(x, k)
            bp[t] = inds[0,:]
            for i in range(k):
                pi[t][i] = x[inds[:, i][0], inds[:, i][1]]
        #
        # end of sequence

        # decode the bp and pi arrays to get the hidden sequence
        hidden_seq = np.zeros(nseq, dtype=int)
        hidden_seq[-1] = bp[nseq+2-1,np.argmax(pi[nseq+2-1])][0]
        for t in range(nseq-1, 0, -1):
            hidden_seq[t-1] = bp[t+2-1,np.argmax(pi[t+2-1])][0]
        return hidden_seq, np.max(pi[nseq+2-1])
    #
    # end of beam

    def viterbi(self, sequence, q, e):

        # initialize the trellis
        # TODO: check the paper, this def needs to be nseq, k, k
        nseq = len(sequence.words)
        ntags = len(self.tags)
        pi = np.log(np.full((nseq+2, ntags, ntags), np.finfo(float).eps, dtype=float))
        bp = np.zeros_like(pi, dtype=int)
        bp[0,:] = 0 # NOTE: this assume <s> is at self.tags[0]
        bp[1,:] = 0
        pi[0,:] = 1
        pi[1,:] = 1
        for t in range(2, nseq+2):
            seq_ind = t-2
            x = (
                e[seq_ind,:][...,np.newaxis,np.newaxis] + \
                q[:, bp[t-1], bp[t-2]] + \
                pi[t-1]
            )
            inds = self.n_best(x, ntags)
            bp[t] = inds[0,:]
            for i in range(ntags):
                pi[t][i] = x[inds[:, i][0], inds[:, i][1]]
        #
        # end of sequence

        # decode the bp and pi arrays to get the hidden sequence
        hidden_seq = np.zeros(nseq, dtype=int)
        hidden_seq[-1] = bp[nseq+2-1,np.argmax(pi[nseq+2-1])][0]
        for t in range(nseq-1, 0, -1):
            hidden_seq[t-1] = bp[t+2-1,np.argmax(pi[t+2-1])][0]
        return hidden_seq, np.max(pi[nseq+2-1])
    #
    # end of beam
        
if __name__ == "__main__":
    pos_tagger = POSTagger()

    train_data = load_data("data/train_x.csv", "data/train_y.csv")
    dev_data = load_data("data/dev_x.csv", "data/dev_y.csv")
    test_data = load_data("data/test_x.csv")

    pos_tagger.train(train_data)

    # Experiment with your decoder using greedy decoding, beam search, viterbi...

    # Here you can also implement experiments that compare different styles of decoding,
    # smoothing, n-grams, etc.
    evaluate(dev_data, pos_tagger)
    exit()
    
    # Predict tags for the test set
    test_predictions = []
    for sentence in test_data:
        test_predictions.extend(pos_tagger.inference(sentence))
    
    # Write them to a file to update the leaderboard
    # TODO
