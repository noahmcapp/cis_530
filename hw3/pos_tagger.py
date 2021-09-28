#!./env/bin/python

""" Contains the part of speech tagger class. """
import re


class POSSentence:
    def __init__(self, words, tag_sequence=None):
        self.words = words
        self.tags = tag_sequence

    # TODO: Add suitable utility methods here if needed

    def __str__(self):
        if not self.tags:
            return ' '.join(self.words)
        return ' '.join([f"{word}/{tag}" for word, tag in zip(self.words, self.tags)])


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
        if word == '-DOCSTART-':
            if curr_words:
                sentence = POSSentence([*curr_words])
                if tag_file:
                    sentence.tags = [*curr_tags]
                sentences.append(sentence)
                curr_words = []
                curr_tags = []
        else:
            curr_words.append(word)
            if tag_file:
                tag = re.sub(r'^\d+,', '', tag_line)[1:-1]
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
