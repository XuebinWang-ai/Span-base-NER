# -*- coding: utf-8 -*-
# @ModuleName: analyze
# @Function:
# @Author: Wxb
# @Time: 2023/4/27 10:20

from parser.utils.corpus import Corpus
from parser.utils.data import TextDataset, batchify
from parser.cmds.cmd import CMD
from parser.models import Model
from datetime import datetime


class Analyze(CMD):
    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Train a model.'
        )
        subparser.add_argument('--fdev', default='data/ctb51/dev.conll',
                               help='path to dev file')
        subparser.add_argument('--ftest', default='data/ctb51/test.conll',
                               help='path to test file')
        # subparser.add_argument('--embed', action='store_true',
        #                        help='whether to use pretrained embeddings')
        # subparser.add_argument('--unk', default=None,
        #                        help='unk token in pretrained embeddings')
        # subparser.add_argument('--dict-file', default=None,
        #                        help='path for dictionary')
        return subparser

    def __call__(self, args):
        super(Analyze, self).__call__(args)

        print('Load the dataset')
        # train = Corpus.load(args.ftrain, self.fields)
        dev = Corpus.load(args.fdev, self.fields)
        test = Corpus.load(args.ftest, self.fields)
        print(dev)
        # train = TextDataset(train, self.fields, args.buckets)
        dev = TextDataset(dev, self.fields, args.buckets)
        test = TextDataset(test, self.fields, args.buckets)
        # set the data loaders
        # train.loader = batchify(train, args.batch_size, False)
        dev.loader = batchify(dev, args.batch_size)
        test.loader = batchify(test, args.batch_size)
        print("dev set: "
              f"{len(dev)} sentences, "
              f"{len(dev.loader)} batches, "
              f"{len(dev.buckets)} buckets")
        print("test set: "
              f"{len(test)} sentences, "
              f"{len(test.loader)} batches, "
              f"{len(test.buckets)} buckets")

        print("Load the model")
        self.model = Model.load(args.model)
        print(f"{self.model}\n")

        print("Evaluate the dataset", args.fdev, args.ftest)
        start = datetime.now()

        loss, metric = self.evaluate(dev.loader)
        total_time = datetime.now() - start
        print(f"Loss: {loss:.4f} {metric}")
        print(f"{total_time}s elapsed, "
              f"{len(dev) / total_time.total_seconds():.2f} Sentences/s")
