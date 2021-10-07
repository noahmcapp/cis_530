import math
from typing import List, Tuple
from collections import defaultdict

from pos_sentence import POSSentence

UNKNOWN = '<UNK>'
START_TAG = "<s>"
ADD_K_EMISSION = 'add_k_emission'
ADD_K_TRANSITION = 'add_k_transition'
KN_TRANSITION = "kn_transition"


### start of Emission Model
class EmissionModel:
    def __init__(self):
        self._word_counts = defaultdict(int)
        self._word_tag_count = defaultdict(int)
        self._tag_count = defaultdict(int)

    def train(self, train_sentences: List[POSSentence]):
        for sentence in train_sentences:
            for word in sentence.words:
                self._word_counts[word] += 1
        for sentence in train_sentences:
            for word, tag in zip(sentence.words, sentence.tags):
                self._tag_count[tag] += 1
                self._word_tag_count[word, tag] += 1

    def emit(self, word: str, tag: str) -> float:
        raise NotImplemented

    def log_emit(self, word: str, tag: str) -> float:
        return math.log(self.emit(word, tag))


class AddKEmissionModel(EmissionModel):
    def __init__(self, k=3):
        super().__init__()
        self.k = k

    def train(self, train_sentences: List[POSSentence]):
        super().train(train_sentences)
        cutoff_count = 1
        unknown_words = set()
        unknown_count = 0
        for word, count in self._word_counts.items():
            if count <= cutoff_count:
                unknown_words.add(word)
                unknown_count += count

        self._word_counts[UNKNOWN] = unknown_count
        for unk_word in unknown_words:
            self._word_counts.pop(unk_word)
            for tag in self._tag_count:
                self._word_tag_count[UNKNOWN, tag] += self._word_tag_count.get((unk_word, tag), 0)
                self._word_tag_count.pop((unk_word, tag), None)

    def emit(self, word: str, tag: str) -> float:
        token = UNKNOWN if self._word_counts.get(word) is None else word
        return (self._word_tag_count.get((token, tag), 0) + self.k) / (
                self._tag_count[tag] + len(self._word_counts) * self.k)


### End of emission model

### Start of Transition model


class TransitionModel:
    def __init__(self, ngram: int):
        self.ngram = ngram
        self._ngram_count = defaultdict(int)
        self._less_one_ngram_count = defaultdict(int)
        self.avail_tags = set()

    @staticmethod
    def _get_ngram(length: int, idx: int, tokens: List[str]) -> Tuple[str]:
        if idx >= length - 1:
            return tuple(tokens[i] for i in range(idx - length + 1, idx + 1))
        return tuple([START_TAG] * (length - idx - 1) + tokens[:idx + 1])

    def train(self, train_sentences: List[POSSentence]):
        self._less_one_ngram_count[tuple([START_TAG] * (self.ngram - 1))] = len(train_sentences)
        for sentences in train_sentences:
            tags = sentences.tags
            for i in range(len(tags)):
                self.avail_tags.add(tags[i])
                self._ngram_count[self._get_ngram(self.ngram, i, tags)] += 1
                self._less_one_ngram_count[self._get_ngram(self.ngram - 1, i - 1, tags)] += 1

    def transit(self, tag: str, prev_tags: Tuple[str]) -> float:
        raise NotImplemented

    def log_transit(self, tag: str, prev_tags: Tuple[str]) -> float:
        return math.log(self.transit(tag, prev_tags))


class KneserNeyTransitionModel(TransitionModel):
    PREFIX_CONTEXT_TYPE = 0
    SUFFIX_CONTEXT_TYPE = 1
    BOTH_CONTEXT_TYPE = 2

    def __init__(self, ngram: int, d=2):
        super().__init__(ngram)
        self.d = d
        self._unique_context_count = {}
        self._transit_memo = {}

    def train(self, train_sentences: List[POSSentence]):
        super().train(train_sentences)
        unique_context = defaultdict(set)
        for sentence in train_sentences:
            tags = sentence.tags
            for i, tag in enumerate(tags):
                seq = self._get_ngram(self.ngram, i, tags)
                for j in range(self.ngram):
                    # unique bigrams
                    if j < self.ngram - 1:
                        unique_context[tuple(), self.BOTH_CONTEXT_TYPE].add((seq[j], seq[j + 1]))
                    # unique context for other higher order grams
                    for k in range(j, self.ngram):
                        subseq = tuple(seq[idx] for idx in range(j, k + 1))
                        # prefix context type
                        if j > 0:
                            unique_context[subseq, self.PREFIX_CONTEXT_TYPE].add(seq[j - 1])
                        # suffix context type
                        if k < self.ngram - 1:
                            unique_context[subseq, self.SUFFIX_CONTEXT_TYPE].add(seq[k + 1])
                        # both prefix and suffix context type
                        if j > 0 and k < self.ngram - 1:
                            unique_context[subseq, self.BOTH_CONTEXT_TYPE].add((seq[j - 1], seq[k + 1]))
        for seq, context in unique_context.items():
            self._unique_context_count[seq] = len(context)

    def transit(self, tag: str, prev_tags: Tuple[str]) -> float:
        if self._transit_memo.get((tag, prev_tags)) is not None:
            return self._transit_memo[tag, prev_tags]
        # base case: unigram
        if not prev_tags:
            return self._unique_context_count[(tag,), self.PREFIX_CONTEXT_TYPE] / \
                   self._unique_context_count[tuple(), self.BOTH_CONTEXT_TYPE]

        # highest order, use normal count
        is_highest_order = len(prev_tags) == self.ngram - 1

        # start with a lower level before interpolating and addition of discounted probability
        p_cont = self.transit(tag, tuple([prev_tags[i] for i in range(1, len(prev_tags))]))

        normalized_sum = self._less_one_ngram_count.get(prev_tags, 0) if is_highest_order else \
            self._unique_context_count.get((prev_tags, self.BOTH_CONTEXT_TYPE), 0)
        if normalized_sum == 0:
            p = self.d * p_cont
        else:
            n_seq = tuple([*prev_tags, tag])
            normalized_d = self.d / normalized_sum
            if is_highest_order:
                p = max(self._ngram_count.get(n_seq, 0) - self.d, 0) / normalized_sum + \
                    normalized_d * self._unique_context_count[prev_tags, self.SUFFIX_CONTEXT_TYPE] * p_cont
            else:
                p = max(self._unique_context_count.get((n_seq, self.PREFIX_CONTEXT_TYPE), 0) - self.d, 0) / normalized_sum + \
                    normalized_d * self._unique_context_count[prev_tags, self.SUFFIX_CONTEXT_TYPE] * p_cont
        self._transit_memo[tag, prev_tags] = p
        return p


class AddKTransitionModel(TransitionModel):
    def __init__(self, ngram: int, k: int = 3):
        super().__init__(ngram)
        self.k = k

    def transit(self, tag: str, prev_tags: Tuple[str]) -> float:
        if len(prev_tags) != self.ngram - 1:
            raise ValueError("number of previous tokens is invalid")
        n_count = self._ngram_count.get(tuple([*prev_tags, tag]), 0)
        d_count = self._less_one_ngram_count.get(prev_tags, 0)
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
    elif transition_model == KN_TRANSITION:
        cls_ = KneserNeyTransitionModel

    if cls_ is None:
        raise ValueError("invalid unknown word handle type")
    return cls_(**kwargs)
