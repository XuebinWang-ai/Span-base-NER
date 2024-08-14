# -*- coding: utf-8 -*-
import sys

from parser.modules import CHAR_LSTM, MLP, BertEmbedding, Biaffine, ElementWiseBiaffine, BiLSTM
from parser.modules.dropout import IndependentDropout, SharedDropout

import torch
import torch.nn as nn
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)


class Model(nn.Module):

    def __init__(self, args):
        super(Model, self).__init__()

        self.args = args
        self.pretrained = False
        # the embedding layer
        self.char_embed = nn.Embedding(num_embeddings=args.n_chars,
                                       embedding_dim=args.n_embed)
        n_lstm_input = args.n_embed  # 100
        if args.feat == 'bert':
            self.feat_embed = BertEmbedding(model=args.bert_model,
                                            n_layers=args.n_bert_layers,
                                            n_out=args.n_feat_embed)
            n_lstm_input += args.n_feat_embed
        if self.args.feat in {'bigram', 'trigram'}:
            self.bigram_embed = nn.Embedding(num_embeddings=args.n_bigrams,
                                             embedding_dim=args.n_embed)
            n_lstm_input += args.n_embed  # 200
        if self.args.feat == 'trigram':
            self.trigram_embed = nn.Embedding(num_embeddings=args.n_trigrams,
                                              embedding_dim=args.n_embed)
            n_lstm_input += args.n_embed

        self.embed_dropout = IndependentDropout(p=args.embed_dropout)

        # the lstm layer
        self.lstm = BiLSTM(input_size=n_lstm_input,
                           hidden_size=args.n_lstm_hidden,
                           num_layers=args.n_lstm_layers,
                           dropout=args.lstm_dropout)
        self.lstm_dropout = SharedDropout(p=args.lstm_dropout)
        # region
        # the MLP layers
        self.mlp_span_l = MLP(n_in=args.n_lstm_hidden * 2,
                              n_out=args.n_mlp_span,
                              dropout=args.mlp_dropout)
        self.mlp_span_r = MLP(n_in=args.n_lstm_hidden * 2,
                              n_out=args.n_mlp_span,  # 500
                              dropout=args.mlp_dropout)
        if args.joint:
            self.mlp_label_l = MLP(n_in=args.n_lstm_hidden * 2,
                                   n_out=args.n_mlp_label,
                                   dropout=args.mlp_dropout
                                   )
            self.mlp_label_r = MLP(n_in=args.n_lstm_hidden * 2,
                                   n_out=args.n_mlp_label,  # 100
                                   dropout=args.mlp_dropout
                                   )
        # the Biaffine layers
        self.span_attn = Biaffine(n_in=args.n_mlp_span,  # 500
                                  bias_x=True,
                                  bias_y=False)
        # endregion
        if args.joint:
            self.label_attn = Biaffine(n_in=args.n_mlp_label,  # 100
                                       n_out=args.n_pos_labels,
                                       bias_x=True,
                                       bias_y=True)

        if args.link == 'mlp':
            # a representation that a fencepost is a split point
            self.mlp_span_s = MLP(n_in=args.n_lstm_hidden * 2,
                                  n_out=args.n_mlp_span,
                                  dropout=args.mlp_dropout)

            # scores for split points
            self.score_split = nn.Linear(args.n_mlp_span, 1)

        elif args.link == 'att':
            self.split_attn = ElementWiseBiaffine(n_in=args.n_lstm_hidden,
                                                  bias_x=True,
                                                  bias_y=False)

        self.pad_index = args.pad_index
        self.unk_index = args.unk_index

    def load_pretrained(self, embed_dict=None):
        embed = embed_dict['embed'] if isinstance(
            embed_dict, dict) and 'embed' in embed_dict else None
        if embed is not None:
            self.pretrained = True
            self.char_pretrained = nn.Embedding.from_pretrained(embed)
            nn.init.zeros_(self.char_embed.weight)
            if self.args.feat == 'bigram':
                embed = embed_dict['bi_embed']
                self.bi_pretrained = nn.Embedding.from_pretrained(embed)
                nn.init.zeros_(self.bigram_embed.weight)
            elif self.args.feat == 'trigram':
                bi_embed = embed_dict['bi_embed']
                tri_embed = embed_dict['tri_embed']
                self.bi_pretrained = nn.Embedding.from_pretrained(bi_embed)
                self.tri_pretrained = nn.Embedding.from_pretrained(tri_embed)
                nn.init.zeros_(self.bigram_embed.weight)
                nn.init.zeros_(self.trigram_embed.weight)
        return self

    def forward(self, feed_dict, link=None):
        chars = feed_dict["chars"]
        batch_size, seq_len = chars.shape
        # get the mask and lengths of given batch
        mask = chars.ne(self.pad_index)
        lens = mask.sum(dim=1).cpu()
        ext_chars = chars
        # set the indices larger than num_embeddings to unk_index
        if self.pretrained:
            ext_mask = chars.ge(self.char_embed.num_embeddings)
            ext_chars = chars.masked_fill(ext_mask, self.unk_index)

        # get outputs from embedding layers
        char_embed = self.char_embed(ext_chars)
        if self.pretrained:
            char_embed += self.char_pretrained(chars)

        if self.args.feat == 'bert':
            feats = feed_dict["feats"]
            feat_embed = self.feat_embed(*feats)
            char_embed, feat_embed = self.embed_dropout(char_embed, feat_embed)
            embed = torch.cat((char_embed, feat_embed), dim=-1)
        elif self.args.feat == 'bigram':
            bigram = feed_dict["bigram"]
            ext_bigram = bigram
            if self.pretrained:
                ext_mask = bigram.ge(self.bigram_embed.num_embeddings)
                ext_bigram = bigram.masked_fill(ext_mask, self.unk_index)
            bigram_embed = self.bigram_embed(ext_bigram)
            if self.pretrained:
                bigram_embed += self.bi_pretrained(bigram)
            char_embed, bigram_embed = self.embed_dropout(
                char_embed, bigram_embed)
            embed = torch.cat((char_embed, bigram_embed), dim=-1)
        elif self.args.feat == 'trigram':
            bigram = feed_dict["bigram"]
            trigram = feed_dict["trigram"]
            ext_bigram = bigram
            ext_trigram = trigram
            if self.pretrained:
                ext_mask = bigram.ge(self.bigram_embed.num_embeddings)
                ext_bigram = bigram.masked_fill(ext_mask, self.unk_index)
                ext_mask = trigram.ge(self.trigram_embed.num_embeddings)
                ext_trigram = trigram.masked_fill(ext_mask, self.unk_index)
            bigram_embed = self.bigram_embed(ext_bigram)
            trigram_embed = self.trigram_embed(ext_trigram)
            if self.pretrained:
                bigram_embed += self.bi_pretrained(bigram)
                trigram_embed += self.tri_pretrained(trigram)
            char_embed, bigram_embed, trigram_embed = self.embed_dropout(
                char_embed, bigram_embed, trigram_embed)
            embed = torch.cat(
                (char_embed, bigram_embed, trigram_embed), dim=-1)
        else:
            embed = self.embed_dropout(char_embed)[0]

        # embed: (B, L, D)

        x = pack_padded_sequence(embed, lens, True, False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)
        x = self.lstm_dropout(x)

        # (B, L, hidden_size)
        x_f, x_b = x.chunk(2, dim=-1)  # lstm前向输出与后向输出
        s_span, s_label = None, None
        if self.args.joint:
            label_l = self.mlp_label_l(x)
            label_r = self.mlp_label_r(x)
            # [batch_size, seq_len, seq_len, n_rels]
            s_label = self.label_attn(label_l, label_r).permute(0, 2, 3, 1)
            # (B, N, N), (B, N, 1)

        if link == 'mlp':
            # minimal span
            # span_representation = torch.zeros((batch_size, seq_len - 1, seq_len - 1,
            #                                    self.args.n_lstm_hidden * 2)).to(self.args.device)
            idx = torch.triu_indices(seq_len - 1, seq_len - 1, offset=1)
            i, j = idx[0], idx[1]
            # span_representation[:, i, j] = torch.cat((x_f[:, j] - x_f[:, j],
            #                                           x_b[:, i] - x_b[:, j]), -1)
            x = torch.cat((x_f[:, j] - x_f[:, j], x_b[:, i] - x_b[:, j]), -1)
            # (B, L-1)
            x = self.mlp_span_s(x)
            x = self.score_split(x).squeeze(dim=-1)
            # span_s = self.score_split(x)
            span = torch.zeros((batch_size, seq_len - 1, seq_len - 1,)).to(self.args.device)
            span[:, i, j] = x

            return span, None

        elif link == 'att':
            # s_link = self.split_attn(x_f[:, :-1], x_b[:, 1:]).unsqueeze(dim=-1)
            pass
        else:
            # representation for fenceposts
            # (B, L-1, hidden_size*2)
            x = torch.cat((x_f[:, :-1], x_b[:, 1:]), -1)  # 错位拼接
            # apply MLPs to the BiLSTM output states
            # (B, L-1, n_mlp_span)
            span_l = self.mlp_span_l(x)
            span_r = self.mlp_span_r(x)
            # (*, i, j) is score of span (i, j)
            # (B, L-1, L-1)
            s_span = self.span_attn(span_l, span_r)
            s_label = None
            s_link = None

        return s_span, s_label

    @classmethod
    def load(cls, path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state = torch.load(path, map_location=device)
        model = cls(state['args'])
        model.load_pretrained(state['pretrained'])
        model.load_state_dict(state['state_dict'], False)
        model.to(device)

        return model

    def save(self, path):
        state_dict, pretrained = self.state_dict(), None
        if self.pretrained:
            pretrained = {'embed': state_dict.pop('char_pretrained.weight')}
            if hasattr(self, 'bi_pretrained'):
                pretrained.update(
                    {'bi_embed': state_dict.pop('bi_pretrained.weight')})
            if hasattr(self, 'tri_pretrained'):
                pretrained.update(
                    {'tri_embed': state_dict.pop('tri_pretrained.weight')})
        state = {
            'args': self.args,
            'state_dict': state_dict,
            'pretrained': pretrained
        }
        torch.save(state, path)
