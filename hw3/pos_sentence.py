class POSSentence:
    def __init__(self, words, tag_sequence=None):
        self.words = words
        self.tags = tag_sequence

    def __str__(self):
        if not self.tags:
            return ' '.join(self.words)
        return ' '.join([f"{word}/{tag}" for word, tag in zip(self.words, self.tags)])