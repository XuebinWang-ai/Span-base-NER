# -*- coding: utf-8 -*-
import sys
from datetime import datetime, timedelta
from parser.models import Model, Span_NER_Model
from parser.cmds.cmd import CMD
from parser.utils.corpus import Corpus
from parser.utils.data import TextDataset, batchify
from parser.utils.metric import Metric

from torch.optim.lr_scheduler import ExponentialLR
from transformers import AdamW, get_linear_schedule_with_warmup



class Train(CMD):

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Train a model.'
        )
        subparser.add_argument('--ftrain', default='data/ctb51/train.conll',
                               help='path to train file')
        subparser.add_argument('--fdev', default='data/ctb51/dev.conll',
                               help='path to dev file')
        subparser.add_argument('--ftest', default='data/ctb51/test.conll',
                               help='path to test file')
        subparser.add_argument('--embed', action='store_true',
                               help='whether to use pretrained embeddings')
        subparser.add_argument('--unk', default=None,
                               help='unk token in pretrained embeddings')
        subparser.add_argument('--dict-file', default=None,
                               help='path for dictionary')
        return subparser

    def __call__(self, args):
        super(Train, self).__call__(args)

        train = Corpus.load(args.ftrain, self.fields)
        dev = Corpus.load(args.fdev, self.fields)
        test = Corpus.load(args.ftest, self.fields)

        train = TextDataset(train, self.fields, args.buckets)
        dev = TextDataset(dev, self.fields)
        test = TextDataset(test, self.fields)
        # set the data loaders
        train.loader = batchify(train, args.batch_size, True)
        dev.loader = batchify(dev, args.batch_size)
        test.loader = batchify(test, args.batch_size)
        print(f"{'train:':6} {len(train):5} sentences, "
              f"{len(train.loader):3} batches, "
              f"{len(train.buckets)} buckets")
        print(f"{'dev:':6} {len(dev):5} sentences, "
              f"{len(dev.loader):3} batches, "
              f"{len(train.buckets)} buckets")
        print(f"{'test:':6} {len(test):5} sentences, "
              f"{len(test.loader):3} batches, "
              f"{len(train.buckets)} buckets")

        print("Create the model")
        embed = {'embed': self.CHAR.embed}
        if hasattr(self, 'BIGRAM'):
            embed.update({
                'bi_embed': self.BIGRAM.embed,
            })
        if hasattr(self, 'TRIGRAM'):
            embed.update({
                'tri_embed': self.TRIGRAM.embed,
            })
        # self.model = Model(args).load_pretrained(embed)
        self.model = Span_NER_Model(args).load_pretrained(embed)
        print(f"{self.model}\n")
        self.model = self.model.to(args.device)

        # todo: 这里暂时注释掉
        # if torch.cuda.device_count() > 1:
        #     self.model = nn.DataParallel(self.model)
        # self.optimizer = AdamW(self.model.parameters(),
        #                        args.lr,
        #                        (args.mu, args.nu),
        #                        args.epsilon)
        # decay_steps = args.decay_epochs * len(train.loader)
        # self.scheduler = ExponentialLR(self.optimizer,
        #                                args.decay**(1/decay_steps))
        
        steps = len(train.loader) * args.epochs // args.update_steps
        # pdb.set_trace()
        self.optimizer = AdamW(
            [{'params': c, 'lr': args.lr}
                for n, c in self.model.named_parameters()],
            args.lr)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, int(steps*args.warmup), steps)

        total_time = timedelta()
        best_e, best_metric = 1, Metric()

        for epoch in range(1, args.epochs + 1):
            start = datetime.now()
            self.train(train.loader)
            print(f"Epoch {epoch} / {args.epochs}:")
            loss, train_metric, _ = self.evaluate(train.loader)
            print(f"{'train:':6} Loss: {loss:.4f} {train_metric}")
            loss, dev_metric, _ = self.evaluate(dev.loader)
            print(f"{'dev:':6} Loss: {loss:.4f} {dev_metric}")
            loss, test_metric, _ = self.evaluate(test.loader)
            print(f"{'test:':6} Loss: {loss:.4f} {test_metric}")

            t = datetime.now() - start
            # save the model if it is the best so far
            if dev_metric > best_metric:
                best_e, best_metric = epoch, dev_metric
                if hasattr(self.model, 'module'):
                    self.model.module.save(args.model)
                else:
                    self.model.save(args.model)
                print(f"{t}s elapsed (saved)\n")
            else:
                print(f"{t}s elapsed\n")
            total_time += t
            if epoch - best_e >= args.patience:
                break
            
            # exit()

        self.model = Span_NER_Model.load(args.model)
        loss, test_metric, _ = self.evaluate(test.loader)

        print(f"max span score of dev is {best_metric.score:.2%} at epoch {best_e}")
        print(f"the span score of test at epoch {best_e} is {test_metric.score:.2%}")
        print(f"average time of each epoch is {total_time / epoch}s")
        print(f"{total_time}s elapsed")
