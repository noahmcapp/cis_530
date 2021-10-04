import math
from typing import List, Tuple
from collections import defaultdict

from pos_sentence import POSSentence

UNKNOWN = '<UNK>'
START_TAG = "<s>"
ADD_K_EMISSION = 'add_k_emission'
ADD_K_TRANSITION = 'add_k_transition'


### start of Emission Model
class EmissionModel:
    def __init__(self):
        self.word_counts = defaultdict(int)
        self.word_tag_count = defaultdict(int)
        self.tag_count = defaultdict(int)

    def train(self, train_sentences: List[POSSentence]):
        for sentence in train_sentences:
            for word in sentence.words:
                self.word_counts[word] += 1
        for sentence in train_sentences:
            for word, tag in zip(sentence.words, sentence.tags):
                self.tag_count[tag] += 1
                self.word_tag_count[word, tag] += 1

    def emit(self, word: str, tag: str) -> float:
        raise NotImplemented

    def log_emit(self, word: str, tag: str) -> float:
        return math.log(self.emit(word, tag))


class AddKEmissionModel(EmissionModel):
    def __init__(self, cutoff_percentile: float = 0.05, k=3):
        super().__init__()
        self.cutoff_percentile = cutoff_percentile
        self.k = k

    def train(self, train_sentences: List[POSSentence]):
        super().train(train_sentences)
        cutoff_count = sorted(self.word_counts)[int(len(self.word_counts) * self.cutoff_percentile)]
        unknown_words = set()
        unknown_count = 0
        for word, count in self.word_counts.items():
            if count <= cutoff_count:
                unknown_words.add(word)
                unknown_count += count

        self.word_counts[UNKNOWN] = unknown_count
        for unk in unknown_words:
            self.word_counts.pop(unk)
        self.word_tag_count = defaultdict(int)
        self.tag_count = defaultdict(int)
        for sentence in train_sentences:
            for word, tag in zip(sentence.words, sentence.tags):
                token = UNKNOWN if word in unknown_words else word
                self.tag_count[tag] += 1
                self.word_tag_count[token, tag] += 1

    def emit(self, word: str, tag: str) -> float:
        token = UNKNOWN if self.word_tag_count.get(word) is None else word
        return (self.word_tag_count.get((token, tag), 0) + self.k) / (
                self.tag_count[tag] + len(self.word_counts) * self.k)


### End of emission model

### Start of Transition model


class TransitionModel:
    def __init__(self, ngram: int):
        self.ngram = ngram
        self.ngram_count = defaultdict(int)
        self.less_one_ngram_count = defaultdict(int)
        self.avail_tags = set()

    @staticmethod
    def get_ngram(length: int, idx: int, tokens: List[str]) -> Tuple[str]:
        if idx >= length - 1:
            return tuple(tokens[i] for i in range(idx - length + 1, idx + 1))
        return tuple([START_TAG] * (length - idx - 1) + tokens[:idx + 1])

    def train(self, train_sentences: List[POSSentence]):
        self.less_one_ngram_count[tuple([START_TAG] * (self.ngram - 1))] = len(train_sentences)
        for sentences in train_sentences:
            tags = sentences.tags
            for i in range(len(tags)):
                self.avail_tags.add(tags[i])
                self.ngram_count[self.get_ngram(self.ngram, i, tags)] += 1
                self.less_one_ngram_count[self.get_ngram(self.ngram - 1, i, tags)] += 1

    def transit(self, tag: str, prev_tags: Tuple[str]) -> float:
        raise NotImplemented

    def log_transit(self, tag: str, prev_tags: Tuple[str]) -> float:
        return math.log(self.transit(tag, prev_tags))


class AddKTransitionModel(TransitionModel):
    def __init__(self, ngram: int, k: int = 3):
        super().__init__(ngram)
        self.k = k

    def transit(self, tag: str, prev_tags: Tuple[str]) -> float:
        if len(prev_tags) != self.ngram - 1:
            raise ValueError("number of previous tokens is invalid")
        n_count = self.ngram_count.get(tuple([tag, *prev_tags]), 0)
        d_count = self.less_one_ngram_count.get(prev_tags, 0)
        return (n_count + self.k) / (d_count + self.k * len(self.avail_tags))


### End of Transition model


def get_emission_model(emission_model: str, **kwargs) -> EmissionModel:
    cls_ = None
    if emission_model == ADD_K_EMISSION:
        cls_ = AddKEmissionModel

    if cls_ is None:
        raise ValueError("invalid emission model")
    return cls_(**kwargs)


def get_transition_model(transition_model: str, **kwargs) -> TransitionModel:
    cls_ = None
    if transition_model == ADD_K_TRANSITION:
        cls_ = AddKTransitionModel

    if cls_ is None:
        raise ValueError("invalid unknown word handle type")
    return cls_(**kwargs)
