UNKNOWN = '<UNK>'
START_WORD = 'START'
STOP_WORD = 'STOP'
START_TAG = '^'
STOP_TAG = '$'
class POSSentence:
    def __init__(self, words, tag_sequence=None, ngram=3):
        self.words = [START_WORD]*(ngram-1) + words + [STOP_WORD]*(ngram-1)
        self.tags = [START_TAG]*(ngram-1) + tag_sequence + [STOP_TAG]*(ngram-1)

    def __str__(self):
        if not self.tags:
            return ' '.join(self.words)
        return ' '.join([f"{word}/{tag}" for word, tag in zip(self.words, self.tags)])
