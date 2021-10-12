#!/usr/bin/env python

""" Contains the part of speech tagger class. """

# system modules
import os
import re
import sys
import argparse
from itertools import permutations

# installed modules
import numpy as np

# custom modules
from param_model_factory import get_emission_model, get_transition_model, ADD_K_EMISSION, ADD_K_TRANSITION, \
    KN_TRANSITION
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

    # loop through words and tags
    for word_line, tag_line in zip(word_lines, tag_lines):

        # ignore header
        if word_line == 'id,word':
            continue

        # get the word and tag from the file lines
        word = re.sub(r'^\d+,', '', word_line)[1:-1]
        tag = re.sub(r'^\d+,', '', tag_line)[1:-1] if tag_file else None

        # document separator
        if word == '-DOCSTART-':

            # start of the first document, store the document separator
            if len(curr_words) == 0:
                curr_words.append('-DOCSTART-')
                if tag_file:
                    curr_tags.append(tag)
                    
            # start of the next document, store the accumulated words/tags as a sentence
            else:
                sentence = POSSentence([*curr_words])
                if tag_file:
                    sentence.tags = [*curr_tags]
                sentences.append(sentence)
                curr_words = ['-DOCSTART-']
                curr_tags = [tag]
            #
            # end of document checking

        # accumulate words/tags inbetween document separators
        else:
            curr_words.append(word)
            if tag_file:
                curr_tags.append(tag)
        #
        # end of document separator checking

    # store the final set of words/tags as a sentence
    sentence = POSSentence(curr_words)
    if tag_file:
        sentence.tags = curr_tags
    sentences.append(sentence)

    # finally, return the data as a list of sentences
    return sentences
#
# end of load_data

def evaluate(data, model, f='output/pred_y.csv', method='greedy', k=1):
    """Evaluates the POS model on some sentences and gold tags.

    This model can compute a few different accuracies:
        - whole-sentence accuracy
        - per-token accuracy
        - compare the probabilities computed by different styles of decoding

    You might want to refactor this into several different evaluation functions.
    
    """

    # generate the transition matrix
    q = model.q_mat()

    # start the progress bar
    width = 50
    sys.stdout.write(('Decoding Dev Dataset: [%s]') % (" "*(width)))
    sys.stdout.flush()
    sys.stdout.write('\b'*(width+1))

    # loop through the sequences    
    pred_tags = []
    for i, sequence in enumerate(data):

        # report progress
        if i % (len(data)//width) == 0:
            sys.stdout.write("-")
            sys.stdout.flush()

        # generate the emission matrix for each sequence
        e = model.e_mat(sequence)
        
        # perform the inference
        tags, ll = model.inference(sequence, q, e, method=method, k=k)
        x = model.inference(sequence, q, e, method='greedy')[0]

        # store the predicted tags
        pred_tags += tags
    #
    # end of sequences

    # store the predicted tags in a file
    write_preds(f, pred_tags)    

    # end the progress bar
    sys.stdout.write("]\nSuccessfully Wrote Predictions -- %s\n" % (f))    
#
# end of evaluate

# writes a list of tags in the same format as the train/dev_y.csv files
def write_preds(f, tags):
    fp = open(f, 'w')
    fp.write('id,tag\n')
    ids = np.arange(len(tags))
    for id, tag in zip(ids, tags):
        fp.write('%s,\"%s\"\n' % (id, tag))
    fp.close()
#
# end of write_preds

# uses MLE to generate probability distributions of the tags and words
# then uses viterbi to decode the trigram HMM model
class POSTagger:
    def __init__(self, ngram):
        self.ngram = ngram
        
        """Initializes the tagger model parameters and anything else necessary. """
        self.em = get_emission_model(ADD_K_EMISSION, **{
            'k': 1e-4
        })
        self.tm = get_transition_model(KN_TRANSITION, **{
            'ngram': self.ngram
        })
    #
    # end of constructor
    
    def train(self, data):
        """Trains the model by computing transition and emission probabilities.

        You should also experiment:
            - smoothing.
            - N-gram models with varying N.
        
        """
        self.em.train(data)
        self.tm.train(data)

        # generate a list of tags
        # NOTE: sorted and adding start tag so that q and e are consistent between runs
        self.tags = ['<s>'] + sorted(list(self.tm.avail_tags))
    #
    # end of train

    # generates a [ntags,ntags,ntags] matrix
    #  e.g. q[i,j,k] = Pr{tag[k] | (tag[i],tag[j])}
    #  e.g. trigram: V D N -> q[V,D,N] = Pr{N | (V,D)}
    def q_mat(self):

        # initialize the array
        n = len(self.tags)
        q = np.empty([n]*self.ngram, dtype=float)

        # loop over all possible ngrams
        for i in range(n**self.ngram):
            ngram = []
            d, r = i, 0
            for j in range(0, self.ngram):
                d, r = divmod(d, n)
                ngram.append(r)
            ngram = tuple(ngram[::-1])
            
            # get the transition probability the ngram
            prev_tags = tuple([self.tags[ind] for ind in ngram[:-1]])
            q[ngram] = self.tm.transit(self.tags[ngram[-1]], prev_tags=prev_tags)
        #
        # end of trigrams loop

        # generate the log-transition probability matrix
        return np.log(q)
    #
    # end of q_mat

    # generates a [nseq,ntags] matrix
    #  e.g. e[i,j] = Pr{seq[i] | tags[j]}
    #  e.g. word: cat, tag: N -> q[cat,N] = Pr{cat | N}
    def e_mat(self, sequence):

        # initialize the matrix
        n = len(sequence.words)
        m = len(self.tags)
        e = np.empty((n,m), dtype=float)

        # loop over all word/tag combinations
        for i in range(n):
            for j in range(m):

                # get the emission probability -- Pr{word|tag}
                e[i][j] = self.em.emit(sequence.words[i], self.tags[j])
            #
            # end of tags
        #
        # end of words
            
        # generate the log-emission probability matrix
        return np.log(e)
    #
    # end of e_mat

    # TODO: need to generalize to ngram
    # generates the log-likelihood of the tag sequence given the q and e matrices
    def sequence_probability(self, tags, q, e):
        """Computes the probability of a tagged sequence given the emission/transition
        probabilities.
        """

        # get the indices of the tags
        tags = [self.tags.index(x) for x in tags]

        # initialize the score
        score = np.log(1)

        # loop through tags
        for t in range(len(tags)):

            # get the t-1 and t-2 tags, or <s> if extends past 0
            t_2 = tags[t-2] if t-2 >= 0 else 0
            t_1 = tags[t-1] if t-1 >= 0 else 0

            # accumulate the log-emission and log-tranistion probs
            score += e[t,tags[t]] + q[t_2,t_1,tags[t]]
            
        return score
    #
    # end of sequence_probability

    # decodes the trigram HMM using various algorithms
    def inference(self, sequence, q, e, method='greedy', k=1):
        """Tags a sequence with part of speech tags.

        You should implement different kinds of inference (suggested as separate
        methods):

            - greedy decoding
            - decoding with beam search
            - viterbi
        """
        if method == 'greedy':
            inds, ll = self.greedy(sequence, q, e)
        if method == 'beam':
            inds, ll = self.beam(sequence, q, e, k)
        if method == 'viterbi':
            inds, ll = self.viterbi(sequence, q, e)

        # convert the tag indices to tag strings
        return [self.tags[i] for i in inds], ll
    #
    # end of inference

    # performs a k=1 beam search over the data, at each word/tag in sequence it calculates
    # the log-likelihood of the word given 
    def greedy(self, sequence, q, e, ngram=3):

        # initialize the trellis
        # NOTE: uses nseq+2 to add 2 <s> at the beginning
        nseq = len(sequence.words)
        pi = np.full([nseq+(self.ngram-1)]+[1]*(self.ngram-1), float("-inf"), dtype=float)
        bp = np.full_like(pi, -1, dtype=int)
        
        # set the start probabilities, t=0 and t=1 can only be <s>
        # NOTE: this assumes <s> is at self.tags[0]
        bp[0:(self.ngram-1)] = 0 
        pi[0:(self.ngram-1)] = 0

        # loop through the sequence, starting at first word in sequence
        for t in range((self.ngram-1), nseq+(self.ngram-1)):

            # map t to the sequence, 2 less to handle the <s> tags
            seq_ind = t-(self.ngram-1)

            # q[tags,less_1_bigram] -> {ntags,} -> prob of seeing any of the tags
            #                                              given the most likely bigram from
            #                                              previous step            
            # pi[less_1_bigram] -> {1,1,1} -> prob of the most likely bigram from previous step
            # e[word,tags] -> {ntags,} -> prob of seeing word given any of the tags
            # TODO: the q part here is not generalized to ngrams
            bp_inds = [tuple([t-i] + [0]*(self.ngram-1)) for i in range((self.ngram-1), 0, -1)]
            x = (
                q[tuple([bp[ind] for ind in bp_inds])] + \
                pi[t-1][...,np.newaxis] + \
                e[seq_ind,:]
            )

            # get the most likely tag at this time step
            bp[t] = np.argmax(x)
            pi[t] = np.max(x)
        #
        # end of sequence

        # decode the trellis, take the most probable
        hidden_seq = list(np.squeeze(bp[(self.ngram-1):]))

        # finally, return the decoded sequence and the estimated log-likelihood 
        return hidden_seq, pi[-1]
    #
    # end of greedy

    def viterbi(self, sequence, q, e):

        # initialize the trellis
        # NOTE: uses nseq+2 to add 2 <s> at the beginning
        nseq = len(sequence.words)
        ntags = len(self.tags)
        pi = np.full([nseq+(self.ngram-1)]+[ntags]*(self.ngram-1), float("-inf"), dtype=float)
        bp = np.full_like(pi, -1, dtype=int)

        # set the start probabilities, t=0 and t=1 can only be <s>
        # NOTE: this assumes <s> is at self.tags[0]
        for i in range(self.ngram-1):
            bp[tuple([i]+[0]*(self.ngram-1))] = 0
            pi[tuple([i]+[0]*(self.ngram-1))] = 0             

        # loop through the sequence, starting at first word in sequence
        for t in range((self.ngram-1), nseq+(self.ngram-1)):

            # map t to the sequence, 2 less to handle the <s> tags
            seq_ind = t-(self.ngram-1)

            # q[tags,tags,tags] -> {ntags,ntags,ntags} -> prob of seeing any of the tags
            #                                             given any possible bigram
            # pi[less_1_bigrams] -> {ntags,ntags,1} -> probs of the bigrams from the previous step
            # e[word,tags] -> {ntags,} -> prob of seeing word given any of the tags
            x = (
                q + \
                pi[t-1][...,np.newaxis] + \
                e[seq_ind]
            )

            # get the bigram that gives each tag the highest log-likelihood
            bp[t] = np.argmax(x, axis=0)
            pi[t] = np.max(x, axis=0)
        #
        # end of sequence
        
        # decode the trellis, start with the most probable bigram in pi
        hidden_seq = list(np.unravel_index(np.argmax(pi[-1]), pi.shape[1:]))

        # loop through the rest of the trellis
        for t in range(nseq-1, (self.ngram-1)-1, -1):

            # get the tag at time step t, using bigram from hidden sequence
            bp_ind = tuple([t+(self.ngram-1)] + hidden_seq[:(self.ngram-1)])
            hidden_seq.insert(0,bp[bp_ind])

        # finally, return the decoded hidden sequence and the exact log-likelihood
        return hidden_seq, np.max(pi[-1])
    #
    # end of viterbi

    # TODO: not generalized to ngram
    def beam(self, sequence, q, e, k=1):

        # initialize the trellis
        # NOTE: uses nseq+2 to add 2 <s> at the beginning
        nseq = len(sequence.words)
        ntags = len(self.tags)
        pi = np.full((nseq+2, ntags, ntags), float("-inf"), dtype=float)
        bp = np.zeros_like(pi, dtype=int)

        # set the start probabilities, t=0 and t=1 can only be <s>
        # NOTE: this assumes <s> is at self.tags[0]
        for i in range(self.ngram-1):
            bp[tuple([i]+[0]*(self.ngram-1))] = 0
            pi[tuple([i]+[0]*(self.ngram-1))] = 0             

        for t in range(2, nseq+2):
            seq_ind = t-2
            x = (
                q + \
                pi[t-1][...,np.newaxis] + \
                e[seq_ind,:]
            )
            y = np.max(x, axis=0)
            inds = np.argpartition(y, -k, axis=-1)[...,-k:]

            # NOTE: this is makes it slower than viterbi, couldn't
            #       figure out how to do this with pi = {nseq, k, k}
            #       instead, just using pi = {nseq,ntags,ntags} and zeroing out
            #       the worst ntags-k paths
            # only keep the best k paths
            bp[t] = np.argmax(x, axis=0)            
            for i in range(ntags):
                for j in inds[i]:
                    pi[t,i,j] = y[i,j]
        #
        # end of sequence

        # decode the trellis, start with the most probable bigram in pi
        hidden_seq = list(np.unravel_index(np.argmax(pi[-1]), pi.shape[1:]))

        # loop through the rest of the trellis
        for t in range(nseq+1, (self.ngram-1)+1, -1):

            # get the tag at time step t, using bigram from hidden sequence
            bp_ind = tuple([t] + hidden_seq[:(self.ngram-1)])
            hidden_seq.insert(0,bp[bp_ind])
            
        # finally, return the decoded hidden sequence and the exact log-likelihood
        return hidden_seq, np.max(pi[-1])
    #
    # end of beam
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-method', type=str, choices=['viterbi','beam','greedy'], default='greedy')
    parser.add_argument('-beam_k', type=int, default=1)
    parser.add_argument('-ngram', type=int, default=3)
    args = parser.parse_args()
    
    pos_tagger = POSTagger(args.ngram)

    train_data = load_data("data/train_x.csv", "data/train_y.csv")
    dev_data = load_data("data/dev_x.csv", "data/dev_y.csv")
    test_data = load_data("data/test_x.csv")

    pos_tagger.train(train_data)

    # Experiment with your decoder using greedy decoding, beam search, viterbi...

    # Here you can also implement experiments that compare different styles of decoding,
    # smoothing, n-grams, etc.
    evaluate(dev_data, pos_tagger, method=args.method, k=args.beam_k)
    exit()
    
    # Predict tags for the test set
    test_predictions = []
    q_mat = model.q_mat()
    
    for sentence in test_data:
        e = model.e_mat(sequence)
        test_predictions.extend(pos_tagger.inference(sentence, q, e, method='viterbi'))
    
    # Write them to a file to update the leaderboard
    write_preds('output/test_pred_y.csv', test_predictions)
#
# end of main
