# -*- coding: utf-8 -*-
# @ModuleName: span-based semi_Markov
# @Function:
# @Author: Wxb
# @Time: 2024/8/5 16:02
import sys

import torch
import torch.autograd as autograd
from parser.utils.alg import score_function_ner

def semi_Markov_loss(span_scores, gold_spans, mask, max_len, mode):
    batch_size, _, _, _ = span_scores.shape
    training = span_scores.requires_grad

    gold_scores = score_function_ner(span_scores, gold_spans, mask).sum()
    logZ = semi_Markov_z(span_scores, mask, M=max_len)

    marginals = span_scores
    if training and mode == 'evaluate':
        marginals, = autograd.grad(logZ, marginals, retain_graph=training)

    # loss = (logZ - gold_scores) / gold_spans.bool().sum()
    loss = (logZ - gold_scores) / batch_size  # TODO

    if loss < 0:
        torch.save(span_scores, '../TEMP-Scripts/span_score.pt')
        torch.save(gold_scores, '../TEMP-Scripts/gold_score.pt')
        torch.save(logZ, '../TEMP-Scripts/logz.pt')
        torch.save(gold_spans, '../TEMP-Scripts/gold_span.pt')
        torch.save(mask, '../TEMP-Scripts/mask.pt')
        print('error')
        exit()

    return loss, marginals


def semi_Markov_z(span_scores, mask, M=-1):
    """
        Args:
            span_scores (Tensor(B, L-1, L-1, T)): ...
            mask (Tensor(B, L-1, L-1)): L include <bos> and <eos>
            M (int): default 10.

        Returns:
            (Tensor(B)): logZ
        """
    batch_size, seq_len, _, label_size = span_scores.shape
    lens = mask[:, 0].sum(dim=-1)

    logZ = span_scores.new_zeros(batch_size, seq_len).double()
    for i in range(1, seq_len):
        t = 0 if M == -1 else max(0, i - M)
        logZ[:, i] = torch.logsumexp(span_scores[:, t:i, i] + logZ[:, t:i].unsqueeze(-1),
                                     dim=(-1, -2))
        

    return logZ[torch.arange(batch_size), lens].sum()


def log_sum_sum_exp(span_scores, logz_before):
    """
        Args:
            span_scores (Tensor(B, L-1, T)): ...
    """
    temp = span_scores + logz_before.unsqueeze(-1)
    # 正确的写法
    x1 = torch.logsumexp(temp, dim=-1)  
    x = torch.logsumexp(x1, dim=-1)
    
    return x

@torch.no_grad()
def semi_Markov_y(span_scores, mask, M=10):
    """
    Chinese Word Segmentation with semi-Markov algorithm.

        Args:
            span_scores (Tensor(B, N, N, T)): (*, i, j, t) is the score of label t for span(i, j)
            mask (Tensor(B, N, N))
            M (int): max length of a span, default 10.

        Returns:
            segs (list[]): segmentation sequence
    """
    batch_size, seq_len, _, label_size = span_scores.size()  # seq_len is maximum length
    lens = mask[:, 0].sum(dim=-1)
    chart = span_scores.new_zeros(batch_size, seq_len).double()
    backtrace = span_scores.new_zeros(batch_size, seq_len, dtype=int)
    labels = torch.argmax(span_scores[:, :, :, 1:], dim=-1) + 1

    for i in range(1, seq_len):
        t = 0 if M == -1 else max(0, i - M)
        max_score, max_index = torch.max(chart[:, t:i] + \
                                         torch.max(span_scores[:, t:i, i, 1:], dim=-1).values, 
                                         dim=-1)
        chart[:, i], backtrace[:, i] = max_score, max_index + t

        # max_score, max_index = torch.max(chart[:, :i] + span_scores[:, :i, i], dim=-1)
        # chart[:, i], backtrace[:, i] = max_score, max_index

    backtrace = backtrace.tolist()
    segments = [traceback(each, length) for each, length in zip(backtrace, lens.tolist())]
    pred_labels = [
        [
            (i, j, pred[i][j])
            for i, j in index
        ]
        for index, pred in zip(segments, labels.tolist())
    ]
    return segments, pred_labels


def traceback(backtrace, length):
    res = []
    left = length
    while left:
        right = backtrace[left]
        res.append((right, left))
        left = right
    res.reverse()
    return res
