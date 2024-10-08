# -*- coding: utf-8 -*-

from collections import Counter
from parser.utils.vocab import Vocab
from parser.utils.fn import tohalfwidth
from parser.utils.common import bos, eos
import torch


class RawField(object):

    def __init__(self, name, fn=None):
        super(RawField, self).__init__()

        self.name = name
        self.fn = fn

    def __repr__(self):
        return f"({self.name}): {self.__class__.__name__}()"

    def preprocess(self, sequence):
        if self.fn is not None:
            sequence = self.fn(sequence)
        return sequence

    def transform(self, sequences):
        return [self.preprocess(sequence) for sequence in sequences]


class Field(RawField):

    def __init__(self, name, pad=None, unk=None, bos=None, eos=None,
                 lower=False, tohalfwidth=False, use_vocab=True, tokenize=None, fn=None,
                 labels=[]):
        self.name = name
        self.pad = pad
        self.unk = unk
        self.bos = bos
        self.eos = eos

        self.labels = labels
        self.label2id = {ne: index for index, ne in enumerate(self.labels)}
        self.label_num = len(self.labels)

        self.lower = lower
        self.tohalfwidth = tohalfwidth
        self.use_vocab = use_vocab
        self.tokenize = tokenize
        self.fn = fn

        self.specials = [token for token in [pad, unk, bos, eos]
                         if token is not None]

    def __repr__(self):
        s, params = f"({self.name}): {self.__class__.__name__}(", []
        if self.pad is not None:
            params.append(f"pad={self.pad}")
        if self.unk is not None:
            params.append(f"unk={self.unk}")
        if self.bos is not None:
            params.append(f"bos={self.bos}")
        if self.eos is not None:
            params.append(f"eos={self.eos}")
        if self.lower:
            params.append(f"lower={self.lower}")
        if not self.use_vocab:
            params.append(f"use_vocab={self.use_vocab}")
        if self.tohalfwidth:
            params.append(f"tohalfwidth={self.tohalfwidth}")
        s += f", ".join(params)
        s += f")"

        return s

    @property
    def pad_index(self):
        return self.specials.index(self.pad) if self.pad is not None else 0

    @property
    def unk_index(self):
        return self.specials.index(self.unk) if self.unk is not None else 0

    @property
    def bos_index(self):
        return self.specials.index(self.bos)

    @property
    def eos_index(self):
        return self.specials.index(self.eos)

    def preprocess(self, sequence):
        if self.fn is not None:
            sequence = self.fn(sequence)
        if self.tokenize is not None:
            sequence = self.tokenize(sequence)
        if self.lower:
            sequence = [str.lower(token) for token in sequence]
        if self.tohalfwidth:
            sequence = [tohalfwidth(token) for token in sequence]

        return sequence

    def build(self, corpus, min_freq=1, embed=None):
        sequences = getattr(corpus, self.name)
        counter = Counter(token
                          for sequence in sequences
                          for token in self.preprocess(sequence))
        self.vocab = Vocab(counter, min_freq, self.specials, self.unk_index)

        if not embed:
            self.embed = None
        else:
            tokens = self.preprocess(embed.tokens)
            # if the `unk` token has existed in the pretrained,
            # then replace it with a self-defined one
            if embed.unk:
                tokens[embed.unk_index] = self.unk

            self.vocab.extend(tokens)    # extend tokens in embed
            self.embed = torch.zeros(len(self.vocab), embed.dim)
            self.embed[self.vocab.token2id(tokens)] = embed.vectors
            self.embed /= torch.std(self.embed)

    def transform(self, sequences):
        sequences = [self.preprocess(sequence) for sequence in sequences]
        if self.use_vocab:
            sequences = [self.vocab.token2id(sequence)
                         for sequence in sequences]
        if self.bos:
            sequences = [[self.bos_index] + sequence for sequence in sequences]
        if self.eos:
            sequences = [sequence + [self.eos_index] for sequence in sequences]
        sequences = [torch.tensor(sequence) for sequence in sequences]

        return sequences


class NGramField(Field):
    def __init__(self, *args, **kwargs):
        self.n = kwargs.pop('n') if 'n' in kwargs else 1
        super(NGramField, self).__init__(*args, **kwargs)

    def build(self, corpus, min_freq=1, dict_file=None, embed=None):
        sequences = getattr(corpus, self.name)
        counter = Counter()
        sequences = [self.preprocess(sequence) for sequence in sequences]
        n_pad = self.n - 1
        for sequence in sequences:
            chars = list(sequence) + [eos] * n_pad
            bichars = ["".join(chars[i + s] for s in range(self.n))
                       for i in range(len(chars) - n_pad)]
            counter.update(bichars)
        if dict_file is not None:
            counter &= self.read_dict(dict_file)
        self.vocab = Vocab(counter, min_freq, self.specials, self.unk_index)
        if not embed:
            self.embed = None
        else:
            tokens = self.preprocess(embed.tokens)
            # if the `unk` token has existed in the pretrained,
            # then replace it with a self-defined one
            if embed.unk:
                tokens[embed.unk_index] = self.unk

            self.vocab.extend(tokens)
            self.embed = torch.zeros(len(self.vocab), embed.dim)
            self.embed[self.vocab.token2id(tokens)] = embed.vectors
            self.embed /= torch.std(self.embed)

    def read_dict(self, dict_file):
        word_list = dict()
        with open(dict_file, encoding='utf-8') as dict_in:
            for line in dict_in:
                line = line.split()
                if len(line) == 3:
                    word_list[line[0]] = 100
        return Counter(word_list)

    def __repr__(self):
        s, params = f"({self.name}): {self.__class__.__name__}(", []
        params.append(f"n={self.n}")
        if self.pad is not None:
            params.append(f"pad={self.pad}")
        if self.unk is not None:
            params.append(f"unk={self.unk}")
        if self.bos is not None:
            params.append(f"bos={self.bos}")
        if self.eos is not None:
            params.append(f"eos={self.eos}")
        if self.lower:
            params.append(f"lower={self.lower}")
        if not self.use_vocab:
            params.append(f"use_vocab={self.use_vocab}")
        if self.tohalfwidth:
            params.append(f"tohalfwidth={self.tohalfwidth}")
        s += f", ".join(params)
        s += f")"

        return s

    def transform(self, sequences):
        sequences = [self.preprocess(sequence) for sequence in sequences]
        n_pad = (self.n - 1)
        for sent_idx, sequence in enumerate(sequences):
            chars = list(sequence) + [eos] * n_pad
            sequences[sent_idx] = ["".join(chars[i + s] for s in range(self.n))
                                   for i in range(len(chars) - n_pad)]
        if self.use_vocab:
            sequences = [self.vocab.token2id(sequence)
                         for sequence in sequences]
        if self.bos:
            sequences = [[self.bos_index] + sequence for sequence in sequences]
        if self.eos:
            sequences = [sequence + [self.eos_index] for sequence in sequences]
        sequences = [torch.tensor(sequence) for sequence in sequences]

        return sequences


class SegmentField(Field):
    """[summary]

    Examples:
        >>> sentence = ["我", "喜欢", "这个", "游戏"]
        >>> sequence = [(0, 1), (1, 3), (3, 5), (5, 7)]
        >>> spans = field.transform([sequences])[0]  
        >>> spans
        tensor([[False, True, False, False,  False,  False, False, False],
                [False, False,  False, True,  False, False, False, False],
                [False, False, False,  False,  False, False, False, False],
                [False, False, False, False,  False, True, False, False],
                [False, False, False, False, False,  False, False, False],
                [False, False, False, False, False, False, False, True],
                [False, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, False]])
    """

    def build(self, corpus, min_freq=1):
        """do nothing

        """
        
        return

    def transform(self, sequences):
        sequences = [self.preprocess(sequence) for sequence in sequences]
        spans = []
        for sequence in sequences:
            seq_len = sequence[-1][1] + 1
            span_chart = torch.full((seq_len, seq_len), self.pad_index).bool()
            for i, j in sequence:
                span_chart[i, j] = 1
            spans.append(span_chart)
            
        return spans


class PosField(Field):
    """[summary]

    Examples:
        >>> sentence = ["我", "喜欢", "这个", "游戏"]
        # >>> sequence = ['pos1', 'pos2', 'pos3', 'pos4']
        >>> sequence = [(0, 1, 'pos1'), (1, 3, 'pos2'), (3, 5, 'pos3'), (5, 7, 'pos4')]
        >>> pos = field.transform([sequences])[0]
        >>> pos
        tensor([id1, id2, id3, id4])
    """

    def build(self, corpus, min_freq=1):
        """do nothing

        """

        return

    def transform(self, sequences):
        sequences = [self.preprocess(sequence) for sequence in sequences]
        spans = []
        for sequence in sequences:
            seq_len = sequence[-1][1] + 1
            span_chart = torch.full((seq_len, seq_len), self.pad_index)
            for i, j, pos in sequence:
                span_chart[i, j] = self.label2id[pos]
            spans.append(span_chart)

        return spans

class NeField(Field):
    """[summary]

    Examples:
        >>> sentence = ["我", "喜欢", "这个", "游戏"]
        # >>> sequence = ['pos1', 'pos2', 'pos3', 'pos4']
        >>> sequence = [(0, 1, 'ne1'), (1, 3, 'ne2'), (3, 5, 'ne3'), (5, 7, 'ne4')]
        >>> pos = field.transform([sequences])[0]
        >>> pos
        tensor([id1, id2, id3, id4])
    """

    def build(self, corpus, min_freq=1):
        """do nothing

        """

        return

    def transform(self, sequences):
        sequences = [self.preprocess(sequence) for sequence in sequences]
        spans = []
        for sequence in sequences:
            # print(sequence)
            seq_len = sequence[-1][1] + 1
            span_chart = torch.full((seq_len, seq_len), self.pad_index)
            for i, j, pos in sequence:
                span_chart[i, j] = self.label2id[pos]
            spans.append(span_chart)
            # print(torch.nonzero((span_chart != -1), as_tuple=True))
            # exit()
            # print(span_chart >= 0)
            # print(span_chart != -1)
            # print((span_chart >= 0) == (span_chart != -1))
            # exit()

        return spans

class BertField(Field):
    def transform(self, sequences):
        subwords, lens = [], []
        sequences = [list(sequence)
                     for sequence in sequences]

        for sequence in sequences:
            # TODO bert 
            sequence = self.preprocess(sequence)
            sequence = [piece if piece else self.preprocess(self.pad)
                        for piece in sequence]
            subwords.append(sequence)
        subwords = [torch.tensor(pieces) for pieces in subwords]
        mask = [torch.ones(len(pieces)).gt(0) for pieces in subwords]

        return list(zip(subwords, mask))
