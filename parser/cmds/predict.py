# -*- coding: utf-8 -*-

from datetime import datetime

import torch

from parser.models import Model, Span_NER_Model
from parser.cmds.cmd import CMD
from parser.utils.corpus import Corpus
from parser.utils.data import TextDataset, batchify
from parser.utils.common import onto4_label, onto5_label
from parser.utils.fn import seg2tag_ner


class Predict(CMD):

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Use a trained model to make predictions.'
        )
        subparser.add_argument('--fdata', default='data/ctb51/test.conll',
                               help='path to dataset')
        subparser.add_argument('--fpred', default='pred.conll',
                               help='path to predicted result')

        return subparser

    def __call__(self, args):
        super(Predict, self).__call__(args)
        
        labels = onto4_label if '4' in args.fdata else onto5_label
        
        print("Load the dataset")
        corpus = Corpus.load(args.fdata, self.fields)
        print(self.fields[:-1],)
        dataset = TextDataset(corpus,
                              self.fields,
                              args.buckets)
        # set the data loader
        dataset.loader = batchify(dataset, args.batch_size)
        print(f"{len(dataset)} sentences, "
              f"{len(dataset.loader)} batches"
              f"{len(dataset.buckets)} buckets")

        print("Load the model")
        # self.model = Model.load(args.model)
        self.model = Span_NER_Model.load(args.model)
        print(f"{self.model}\n")

        print("Make predictions on the dataset")
        start = datetime.now()
        all_segs_ner, metric = self.predict(dataset.loader)

        all_segs_ner = [
            seg2tag_ner([
                (i,j,labels[index]) 
                for i,j,index in sents
            ])
            for sents in all_segs_ner
        ]

        total_time = datetime.now() - start
        # restore the order of sentences in the buckets
        indices = torch.tensor([i
                                for bucket in dataset.buckets.values()
                                for i in bucket]).argsort()
        corpus.labels = [all_segs_ner[i] for i in indices]

        print(f"Save the predicted result to {args.fpred}")
        corpus.save(args.fpred)
        print(f'{args.fdata}: {metric}')
        print(f"{total_time}s elapsed, "
              f"{len(dataset) / total_time.total_seconds():.2f} Sentences/s")
