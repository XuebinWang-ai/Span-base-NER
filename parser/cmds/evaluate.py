# -*- coding: utf-8 -*-

import torch
from datetime import datetime
from parser.models import Model, Span_NER_Model
from parser.cmds.cmd import CMD
from parser.utils.corpus import Corpus
from parser.utils.data import TextDataset, batchify


class Evaluate(CMD):

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Evaluate the specified model and dataset.'
        )
        subparser.add_argument('--fdata', default='data/ctb51/test.conll',
                               help='path to dataset')
        subparser.add_argument('--fprob', default='prob.pred.txt',
                               help='path to save probabilty.')
        subparser.add_argument('--threshold', default='0.2',
                               help='the probabilty used to filter.')
        
        return subparser

    def __call__(self, args):
        super(Evaluate, self).__call__(args)

        print("Load the dataset")
        corpus = Corpus.load(args.fdata, self.fields)
        dataset = TextDataset(corpus, self.fields, args.buckets)
        # set the data loader
        dataset.loader = batchify(dataset, args.batch_size)
        print(f"{len(dataset)} sentences, "
              f"{len(dataset.loader)} batches, "
              f"{len(dataset.buckets)} buckets")

        print("Load the model")
        # self.model = Model.load(args.model)
        self.model = Span_NER_Model.load(args.model)
        print(f"{self.model}\n")

        print("Evaluate the dataset", args.fdata)
        start = datetime.now()
        loss, metric, (all_pred_ner_prob, le_threshold_probs) = \
            self.evaluate(dataset.loader)
        total_time = datetime.now() - start

        indices = torch.tensor([i
                                for bucket in dataset.buckets.values()
                                for i in bucket]).argsort()
        corpus.prob_labels = [all_pred_ner_prob[i] for i in indices]
        corpus.le_threshold_prob_labels = [[float(args.threshold)] + le_threshold_probs[i] 
                                           for i in indices]

        print(f"Save the predicted result to {args.fprob}")
        corpus.save(args.fprob)

        print(f"Loss: {loss:.4f} {metric}")
        print(f"{total_time}s elapsed, "
              f"{len(dataset) / total_time.total_seconds():.2f} Sentences/s")
