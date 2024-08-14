# -*- coding: utf-8 -*-
# @ModuleName: semi_Markov
# @Function:
# @Author: Wxb
# @Time: 2023/4/21 16:37
import sys

import torch
import torch.autograd as autograd
from parser.utils.alg import score_function


def semi_Markov_loss(span_scores, gold_spans, mask, max_len, mode):
    batch_size, _, _ = span_scores.shape
    training = span_scores.requires_grad

    gold_scores = score_function(span_scores, gold_spans, mask).sum()
    # logZ = semi_Markov_z(span_scores.requires_grad_(), mask, M=max_len)
    logZ = semi_Markov_z(span_scores, mask, M=max_len)

    marginals = span_scores
    if training and mode == 'evaluate':
        marginals, = autograd.grad(logZ, marginals, retain_graph=training)

    # norm : sentence_size, word_size, char_size
    # loss = (logZ - gold_scores) / batch_size
    loss = (logZ - gold_scores) / gold_spans.sum()
    # loss = (logZ - gold_scores) / mask.sum()
    return loss, marginals


def semi_Markov_z(span_scores, mask, M=10):
    """
        Args:
            span_scores (Tensor(B, L-1, L-1)): ...
            mask (Tensor(B, L-1, L-1)): L include <bos> and <eos>
            M (int): default 10.

        Returns:
            (Tensor(B)): logZ
        """
    batch_size, seq_len, _ = span_scores.shape
    lens = mask[:, 0].sum(dim=-1)

    logZ = span_scores.new_zeros(batch_size, seq_len).double()

    for i in range(1, seq_len):
        t = max(0, i - M)
        logZ[:, i] = torch.logsumexp(logZ[:, t:i] + span_scores[:, t:i, i], dim=-1)
        # logZ[:, i] = torch.logsumexp(logZ[:, :i] + span_scores[:, :i, i], dim=-1)

    return logZ[torch.arange(batch_size), lens].sum()


@torch.no_grad()
def semi_Markov_y(span_scores, mask, M=10):
    """
    Chinese Word Segmentation with semi-Markov algorithm.

        Args:
            span_scores (Tensor(B, N, N)): (*, i, j) is score for span(i, j)
            mask (Tensor(B, N, N))
            M (int): default 10.

        Returns:
            segs (list[]): segmentation sequence
    """
    batch_size, seq_len, _ = span_scores.size()  # seq_len is maximum length
    lens = mask[:, 0].sum(dim=-1)
    chart = span_scores.new_zeros(batch_size, seq_len).double()
    backtrace = span_scores.new_zeros(batch_size, seq_len, dtype=int)

    for i in range(1, seq_len):
        t = max(0, i - M)
        max_score, max_index = torch.max(chart[:, t:i] + span_scores[:, t:i, i], dim=-1)
        chart[:, i], backtrace[:, i] = max_score, max_index + t

        # max_score, max_index = torch.max(chart[:, :i] + span_scores[:, :i, i], dim=-1)
        # chart[:, i], backtrace[:, i] = max_score, max_index

    backtrace = backtrace.tolist()

    segments = [traceback(each, length) for each, length in zip(backtrace, lens.tolist())]
    return segments


@torch.no_grad()
def semi_Markov_y_pos(span_scores, pos_scores, mask, M=10):
    """
    Chinese Word Segmentation with semi-Markov algorithm.

        Args:
            span_scores (Tensor(B, N, N)): (*, i, j) is score for span(i, j)
            mask (Tensor(B, N, N))
            M (int): default 10.

        Returns:
            segs (list[]): segmentation sequence
    """
    batch_size, seq_len, _ = span_scores.size()  # seq_len is maximum length
    lens = mask[:, 0].sum(dim=-1)
    # print(lens)
    chart = span_scores.new_zeros(batch_size, seq_len).double()
    backtrace = span_scores.new_zeros(batch_size, seq_len, dtype=int)

    for i in range(1, seq_len):
        t = max(0, i - M)
        max_score, max_index = torch.max(chart[:, t:i] + span_scores[:, t:i, i], dim=-1)
        chart[:, i], backtrace[:, i] = max_score, max_index + t

        # max_score, max_index = torch.max(chart[:, :i] + span_scores[:, :i, i], dim=-1)
        # chart[:, i], backtrace[:, i] = max_score, max_index

    backtrace = backtrace.tolist()

    segments = [traceback(each, length) for each, length in zip(backtrace, lens.tolist())]

    # pos_scores = pos_scores[mask]
    pred_pos = pos_scores.argmax(-1)
    # print(pred_pos.shape)
    pred_pos = [
        [
            (i, j, pred[i][j])
            for i, j in index
        ]
        for index, pred in zip(segments, pred_pos.tolist())
    ]

    return segments, pred_pos


def traceback(backtrace, length):
    res = []
    left = length
    while left:
        right = backtrace[left]
        res.append((right, left))
        left = right
    res.reverse()
    return res
