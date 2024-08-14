# -*- coding: utf-8 -*-

from collections import namedtuple
from collections.abc import Iterable

from parser.utils.fn import tag2seg_ner
from parser.utils.field import Field
import sys
import json

CoNLL = namedtuple(typename='CoNLL',
                   field_names=['CHAR', 'SEG'],
                   defaults=[None] * 2)


class Sentence(object):

    def __init__(self, fields, values):
        for field, value in zip(fields, values):
            if isinstance(field, Iterable):  # field.name 与 char or seg 对应
                for j in range(len(field)):
                    setattr(self, field[j].name, value)
            else:
                setattr(self, field.name, value)
        self.fields = fields

    @property
    def values(self):
        for field in self.fields:
            if isinstance(field, Iterable):
                yield getattr(self, field[0].name)
            else:
                yield getattr(self, field.name)

    def __len__(self):
        return len(next(iter(self.values)))

    def __repr__(self):
        if hasattr(self, "labels"):  # 保存为BMES形式
            temp = list(self.values)
            temp[1] = self.labels
            return '\n'.join('\t'.join(map(str, line))
                             for line in zip(*temp)) + '\n'
        else:
            # 保存为 (i,j,ne,prob) 形式
            values = list(self.values)
            values[1] = [each for each in self.prob_labels if each[-2] != "O"]
            threshold = self.le_threshold_prob_labels[0]
            self.le_threshold_prob_labels = \
                [each for each in self.le_threshold_prob_labels[1:] if each[-2] != "O"]

            return f'{values[0]}\npred  :{values[1]}\np>={threshold}:{self.le_threshold_prob_labels}\n'

class Corpus(object):

    def __init__(self, fields, sentences):
        super(Corpus, self).__init__()

        self.fields = fields
        self.sentences = sentences  # all sentences

    def __len__(self):
        return len(self.sentences)

    def __repr__(self):
        return '\n'.join(str(sentence) for sentence in self)

    def __getitem__(self, index):
        return self.sentences[index]

    def __getattr__(self, name):
        if not hasattr(self.sentences[0], name):
            raise AttributeError
        for sentence in self.sentences:
            yield getattr(sentence, name)  # 生成器
        # res = []   # only for test
        # for sentence in self.sentences[:5]:
        #     res.append(getattr(sentence, name))
        # return res

    def __setattr__(self, name, value):
        if name in ['fields', 'sentences']:
            self.__dict__[name] = value
        else:
            for i, sentence in enumerate(self.sentences):
                setattr(sentence, name, value[i])

    @classmethod  # 无需实例化class可直接调用，cls = self
    def load(cls, path, fields):
        start, sentences = 0, []
        tags = set()
        fields = [field if field is not None else Field(str(i))
                  for i, field in enumerate(fields)]
        with open(path, 'r') as f:
            lines = [line.strip() for line in f]
        fn = tag2seg_ner
        for i, line in enumerate(lines):
            if not line:
                # [chars: ["我", "爱", ...], tags: [B, M, E, ...]], segment: [(0,1), (1,3)...]
                values = list(zip(*[l.split() for l in lines[start:i]]))  # char and tags
                values[-1] = fn(values[-1])  # tags to segment
                # print(values[-1])
                # print()
                for each in values[-1]:
                    tags.add(each[-1])
                sentences.append(Sentence(fields, values))  # field.name, char and segment
                start = i + 1
        # print(tags)
        # print(len(tags))
        # exit()
        return cls(fields, sentences)

    def save(self, path):
        with open(path, 'w') as f:
            f.write(f"{self}\n")

