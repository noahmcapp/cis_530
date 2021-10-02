#!/usr/bin/env python

""" Contains the part of speech tagger class. """

def load_data(sentence_file, tag_file=None):
    """Loads data from two files: one containing sentences and one containing tags.

    tag_file is optional, so this function can be used to load the test data.

    Suggested to split the data by the document-start symbol.

    """
    return []

def evaluate(data, model):
    """Evaluates the POS model on some sentences and gold tags.

    This model can compute a few different accuracies:
        - whole-sentence accuracy
        - per-token accuracy
        - compare the probabilities computed by different styles of decoding

    You might want to refactor this into several different evaluation functions.
    
    """
    pass

class POSTagger():
    def __init__(self):
        """Initializes the tagger model parameters and anything else necessary. """
        pass

    def train(self, data):
        """Trains the model by computing transition and emission probabilities.

        You should also experiment:
            - smoothing.
            - N-gram models with varying N.
        
        """
        pass

    def sequence_probability(self, sequence, tags):
        """Computes the probability of a tagged sequence given the emission/transition
        probabilities.
        """
        return 0.

    def inference(self, sequence):
        """Tags a sequence with part of speech tags.

        You should implement different kinds of inference (suggested as separate
        methods):

            - greedy decoding
            - decoding with beam search
            - viterbi
        """
        return []

    # generate a (len(sequence), self.ntags) array
    def get_emissions(sequence):
        return e
    
    def viterbi(self, sequence, q=None, e=None):

        # get the transition and emissions matrices
        if q is None:
            q = self.q
        if e is None:
            e = self.get_emissions(sequence)

        # initialize the trellis
        nseq = len(sequence)
        pi = np.zeros((self.ntags, nseq), dtype=float)
        bp = np.array_like(pi, dtype=int)
        pi[0][0] = 1

        # loop over sequence
        for t in range(1, nseq):

            # loop over all possible tags
            for i in range(self.ntags):

                # [1] -- e[curr_token, tag] is prob of seeing token at this time step
                #                             given the tag being analyzed
                # [ntags] -- q[tag, tags] is probs of seeing the tag being analyzed
                #                           given all possible tags
                # [ntags] -- pi[tags, prev_token] is probs of any tag occuring
                #                                   at the previous time step
                # [ntags] -- x is probs of the current tag being associated with the
                #            current word and any of the tags at the previous time step
                x = e[sequence[t],i] * q[i,:] * pi[:,t-1]

                # get the tag at previous time step which yields the highest prob
                bp[i,t] = np.argmax(x)

                # store the highest prob at this tag
                pi[i,t] = x[bp[i,t]]
            #
            # end of tags
        #
        # end of sequence

        # use the back pointer to generate the most optimal sequence
        hidden_seq = np.zeros(nseq, dtype=int)
        hidden_seq[nseq-1] = np.argmax(pi[:,nseq-1])
        for t in range(nseq-1, -1, -1):
            hidden_seq[t-1] = bp[hidden_seq[t],t]
        return hidden_seq
    #
    # end of viterbi
    
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

    # Predict tags for the test set
    test_predictions = []
    for sentence in test_data:
        test_predictions.extend(pos_tagger.inference(sentence))
    
    # Write them to a file to update the leaderboard
    # TODO
