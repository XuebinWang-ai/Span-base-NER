# -*- coding: utf-8 -*-

import os
import sys
from parser.utils import Embedding
from parser.utils.alg import neg_log_likelihood, directed_acyclic_graph, crf
from parser.utils.common import pad, unk, bos, eos, onto4_label, onto5_label
from parser.utils.corpus import CoNLL, Corpus
from parser.utils.field import BertField, Field, NGramField, SegmentField, PosField, NeField
from parser.utils.fn import get_spans, tensor2scalar
from parser.utils.metric import SegF1Metric
from parser.utils.span_semi_Markov import semi_Markov_loss, semi_Markov_y

import torch
import torch.nn as nn
from transformers import BertTokenizer


class CMD(object):

    def __call__(self, args):
        self.args = args
        path = args.fdata if hasattr(args, 'fdata') else args.ftrain
        labels = onto4_label if '4' in path else onto5_label
        if not os.path.exists(args.file):
            os.mkdir(args.file)
        if not os.path.exists(args.fields) or args.preprocess:
            print("Preprocess the data")

            self.CHAR = Field('chars', pad=pad, unk=unk,
                              bos=bos, eos=eos, lower=True)

            # TODO span as label, modify chartfield to spanfield
            self.SEG = NeField('segs', labels=labels)
            # self.POS = PosField('pos') if args.joint else None

            if args.feat == 'bert':
                tokenizer = BertTokenizer.from_pretrained(args.bert_model)
                self.FEAT = BertField('bert',
                                      pad='[PAD]',
                                      bos='[CLS]',
                                      eos='[SEP]',
                                      tokenize=tokenizer.encode)
                self.fields = CoNLL(CHAR=(self.CHAR, self.FEAT),
                                    SEG=self.SEG, 
                                    # POS=self.POS
                                    )
            elif args.feat == 'bigram':
                self.BIGRAM = NGramField(
                    'bichar', n=2, pad=pad, unk=unk, bos=bos, eos=eos, lower=True)
                # self.fields = CoNLL(CHAR=(self.CHAR, self.BIGRAM),
                #                     SEG=self.SEG)     # 原本的
                self.fields = CoNLL(CHAR=(self.CHAR, self.BIGRAM),
                                    SEG=self.SEG, POS=self.POS)
            elif args.feat == 'trigram':
                self.BIGRAM = NGramField(
                    'bichar', n=2, pad=pad, unk=unk, bos=bos, eos=eos, lower=True)
                self.TRIGRAM = NGramField(
                    'trichar', n=3, pad=pad, unk=unk, bos=bos, eos=eos, lower=True)
                self.fields = CoNLL(CHAR=(self.CHAR,
                                          self.BIGRAM,
                                          self.TRIGRAM),
                                    SEG=self.SEG,
                                    POS=self.POS)
            else:
                self.fields = CoNLL(CHAR=self.CHAR,
                                    SEG=self.SEG,
                                    POS=self.POS)

            train = Corpus.load(args.ftrain, self.fields)  # get: field.name, value
            embed = Embedding.load(
                'data/tencent.char.200.txt',
                args.unk) if args.embed else None
            self.CHAR.build(train, args.min_freq, embed)
            if hasattr(self, 'FEAT'):
                self.FEAT.build(train)
            if hasattr(self, 'BIGRAM'):
                embed = Embedding.load(
                    'data/tencent.bi.200.txt',
                    args.unk) if args.embed else None
                self.BIGRAM.build(train, args.min_freq,
                                  embed=embed,
                                  dict_file=args.dict_file)
            if hasattr(self, 'TRIGRAM'):
                embed = Embedding.load(
                    'data/tencent.tri.200.txt',
                    args.unk) if args.embed else None
                self.TRIGRAM.build(train, args.min_freq,
                                   embed=embed,
                                   dict_file=args.dict_file)
            # TODO
            self.SEG.build(train)
            # if self.POS:
            #     self.POS.build(train)  # 新加的
            torch.save(self.fields, args.fields)
        else:
            self.fields = torch.load(args.fields)
            if args.feat == 'bert':
                self.CHAR, self.FEAT = self.fields.CHAR
            elif args.feat == 'bigram':
                self.CHAR, self.BIGRAM = self.fields.CHAR
            elif args.feat == 'trigram':
                self.CHAR, self.BIGRAM, self.TRIGRAM = self.fields.CHAR
            else:
                self.CHAR = self.fields.CHAR
            # TODO
            self.SEG = self.fields.SEG   # 原本的
            # self.SEG, self.POS = self.fields.SEG, self.fields.POS

        self.interval = [0] * 11
        self.right = [0] * 11
        self.all = [0] * 11

        # TODO loss funciton 
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-1)
        # self.criterion = nn.CrossEntropyLoss(reduction='sum')

        args.update({
            'n_chars': self.CHAR.vocab.n_init,
            'pad_index': self.CHAR.pad_index,
            'unk_index': self.CHAR.unk_index,
            'ner_labels': len(labels)
        })

        # TODO
        vocab = f"{self.CHAR}\n"
        if hasattr(self, 'FEAT'):
            args.update({
                'n_feats': self.FEAT.vocab.n_init,
            })
            vocab += f"{self.FEAT}\n"
        if hasattr(self, 'BIGRAM'):
            args.update({
                'n_bigrams': self.BIGRAM.vocab.n_init,
            })
            vocab += f"{self.BIGRAM}\n"
        if hasattr(self, 'TRIGRAM'):
            args.update({
                'n_trigrams': self.TRIGRAM.vocab.n_init,
            })
            vocab += f"{self.TRIGRAM}\n"

        print(f"Override the default configs\n{args}")
        print(vocab[:-1])

    def train(self, loader):
        self.model.train()
        # torch.set_grad_enabled(True)
        for data in loader:
            # TODO label
            pos = None
            if self.args.feat == 'bert':
                chars, feats, segs = data
                feed_dict = {"chars": chars, "feats": feats}
            elif self.args.feat == 'bigram':
                chars, bigram, segs = data
                feed_dict = {"chars": chars, "bigram": bigram}
            elif self.args.feat == 'trigram':
                chars, bigram, trigram, segs = data
                feed_dict = {"chars": chars,
                             "bigram": bigram, "trigram": trigram}
            else:
                chars, segs = data
                feed_dict = {"chars": chars}

            self.optimizer.zero_grad()

            batch_size, seq_len = chars.shape
            # fenceposts length: (B)
            lens = chars.ne(self.args.pad_index).sum(1) - 1
            # TODO purpose
            # (B, 1, L-1)
            mask = lens.new_tensor(range(seq_len - 1)) < lens.view(-1, 1, 1)
            # TODO purpose
            # for example, seq_len=10, fenceposts=7, pad=2
            # for each sentence, get a L-1*L-1 matrix
            # span (i, i) and pad are masked 
            # [[False,  True,  True,  True,  True,  True,  True, False, False],
            #  [False, False,  True,  True,  True,  True,  True, False, False],
            #  [False, False, False,  True,  True,  True,  True, False, False],
            #  [False, False, False, False,  True,  True,  True, False, False],
            #  [False, False, False, False, False,  True,  True, False, False],
            #  [False, False, False, False, False, False,  True, False, False],
            #  [False, False, False, False, False, False, False, False, False],
            #  [False, False, False, False, False, False, False, False, False],
            #  [False, False, False, False, False, False, False, False, False]]
            # (B, L-1, L-1)
            mask = mask & mask.new_ones(seq_len - 1, seq_len - 1).triu_(1)
            # (B, L-1, L-1), (B, L-1, 1)
            s_span = self.model(feed_dict)

            loss, _ = self.get_loss(s_span, segs, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     self.args.clip)
            self.optimizer.step()
            self.scheduler.step()

    @torch.no_grad()
    def evaluate(self, loader):
        if self.args.mode == 'evaluate':
            torch.set_grad_enabled(True)
        else:
            torch.set_grad_enabled(False)
        self.model.eval()

        path = self.args.fdata if hasattr(self.args, 'fdata') else self.args.ftrain
        labels = onto4_label if '4' in path else onto5_label

        total_loss, metric_ner = 0, SegF1Metric()
        all_pred_ner_prob, le_threshold_probs = [], []

        for data in loader:
            if self.args.feat == 'bert':
                chars, feats, segs = data
                feed_dict = {"chars": chars, "feats": feats}
            elif self.args.feat == 'bigram':
                chars, bigram, segs = data
                feed_dict = {"chars": chars, "bigram": bigram}
            elif self.args.feat == 'trigram':
                chars, bigram, trigram, segs = data
                feed_dict = {"chars": chars,
                             "bigram": bigram, "trigram": trigram}
            else:
                chars, segs = data
                feed_dict = {"chars": chars}

            batch_size, seq_len = chars.shape
            lens = chars.ne(self.args.pad_index).sum(1) - 1
            mask = lens.new_tensor(range(seq_len - 1)) < lens.view(-1, 1, 1)
            mask = mask & mask.new_ones(seq_len - 1, seq_len - 1).triu_(1)

            s_span = self.model(feed_dict)
            loss, span_marginals = self.get_loss(s_span, 
                                                 segs, 
                                                 mask, 
                                                 max_len=self.args.max_ne_length)

            total_loss += loss.item()

            pred_segs, pred_ner = self.decode(s_span, mask, 
                                              max_len=self.args.max_ne_length)
            # list
            # gold_segs = [torch.nonzero(gold).tolist() for gold in segs]
            gold_segs_ner_index = [
                list(zip(*tensor2scalar(torch.nonzero(gold, as_tuple=True))))
                for gold in segs]
            gold_ner = [[(i, j, gold[i][j]) for i, j in index]
                for index, gold in zip(gold_segs_ner_index, segs.tolist())]
            
            metric_ner(pred_ner, gold_ner)
            if self.args.mode == 'train':
                continue

            threshold = float(self.args.threshold)

            pred_ner_prob = [
                [(i, j, labels[ne_ids], prob[i][j][ne_ids]) 
                 for (i, j, ne_ids) in index]
                 for index, prob in zip(pred_ner, span_marginals.tolist())
            ]            
            
            prob_le2_index = [
                list(zip(*tensor2scalar(torch.nonzero((marg >= threshold) & m.unsqueeze(-1), 
                                                      as_tuple=True))))
                for marg, m in zip(span_marginals, mask)
            ]
            # (i, j, ne_ids, prob)
            le_threshold_ner_prob = [
                [(i, j, labels[ne_ids], prob[i][j][ne_ids])
                 for (i, j, ne_ids) in index]
                 for index, prob in zip(prob_le2_index, span_marginals.tolist())
            ]

            all_pred_ner_prob.extend(pred_ner_prob)
            le_threshold_probs.extend(le_threshold_ner_prob)

        total_loss /= len(loader)

        # TODO metric
        return total_loss, metric_ner, (all_pred_ner_prob, le_threshold_probs)

    @torch.no_grad()
    def predict(self, loader):
        self.model.eval()

        all_segs_ner = []
        total_loss, metric_ner = 0, SegF1Metric()
        for data in loader:
            if self.args.feat == 'bert':
                chars, feats, segs = data
                feed_dict = {"chars": chars, "feats": feats}
            elif self.args.feat == 'bigram':
                chars, bigram, segs = data
                feed_dict = {"chars": chars, "bigram": bigram}
            elif self.args.feat == 'trigram':
                chars, bigram, trigram, segs = data
                feed_dict = {"chars": chars,
                             "bigram": bigram, "trigram": trigram}
            else:
                chars, segs = data
                feed_dict = {"chars": chars}

            batch_size, seq_len = chars.shape
            lens = chars.ne(self.args.pad_index).sum(1) - 1
            mask = lens.new_tensor(range(seq_len - 1)) < lens.view(-1, 1, 1)
            mask = mask & mask.new_ones(seq_len - 1, seq_len - 1).triu_(1)

            s_span = self.model(feed_dict)

            # pred_segs = directed_acyclic_graph(s_span, mask, s_link)
            pred_segs, pred_ner = self.decode(s_span, mask, 
                                              max_len=self.args.max_ne_length)
            gold_segs_ner_index = [
                list(zip(*tensor2scalar(torch.nonzero(gold, as_tuple=True))))
                for gold in segs
            ]
            gold_ner = [
                [(i, j, gold[i][j]) for i, j in index]
                for index, gold in zip(gold_segs_ner_index, segs.tolist())
            ]
            
            loss, _ = self.get_loss(s_span, segs, mask, 
                                    self.args.max_ne_length)
            total_loss += loss.item()
            
            metric_ner(pred_ner, gold_ner)
            all_segs_ner.extend(pred_ner)
            
        print(f'Loss: {total_loss/len(loader):.4f}')
        return all_segs_ner, metric_ner

    def get_loss(self, s_span, segs, mask, max_len=-1):
        span_loss, span_marginals = semi_Markov_loss(s_span, 
                                                     segs, 
                                                     mask, 
                                                     max_len, 
                                                     self.args.mode)
        return span_loss, span_marginals

    @staticmethod
    def decode(s_span, mask, max_len):
        pred_spans, pred_span_ner = semi_Markov_y(s_span, 
                                                  mask, 
                                                  M=max_len)

        return pred_spans, pred_span_ner
