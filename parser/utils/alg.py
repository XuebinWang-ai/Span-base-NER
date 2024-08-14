# -*- coding: utf-8 -*-

from parser.utils.fn import stripe

import torch
import torch.autograd as autograd


def kmeans(x, k):
    x = torch.tensor(x, dtype=torch.float)
    # count the frequency of each datapoint
    d, indices, f = x.unique(return_inverse=True, return_counts=True)
    # calculate the sum of the values of the same datapoints
    total = d * f
    # initialize k centroids randomly
    c, old = d[torch.randperm(len(d))[:k]], None
    # assign labels to each datapoint based on centroids
    dists, y = torch.abs_(d.unsqueeze(-1) - c).min(dim=-1)
    # make sure number of datapoints is greater than that of clusters
    assert len(d) >= k, f"unable to assign {len(d)} datapoints to {k} clusters"

    while old is None or not c.equal(old):
        # if an empty cluster is encountered,
        # choose the farthest datapoint from the biggest cluster
        # and move that the empty one
        for i in range(k):
            if not y.eq(i).any():
                mask = y.eq(torch.arange(k).unsqueeze(-1))
                lens = mask.sum(dim=-1)
                biggest = mask[lens.argmax()].nonzero().view(-1)
                farthest = dists[biggest].argmax()
                y[biggest[farthest]] = i
        mask = y.eq(torch.arange(k).unsqueeze(-1))
        # update the centroids
        c, old = (total * mask).sum(-1) / (f * mask).sum(-1), c
        # re-assign all datapoints to clusters
        dists, y = torch.abs_(d.unsqueeze(-1) - c).min(dim=-1)
    # assign all datapoints to the new-generated clusters
    # without considering the empty ones
    y, assigned = y[indices], y.unique().tolist()
    # get the centroids of the assigned clusters
    centroids = c[assigned].tolist()
    # map all values of datapoints to buckets
    clusters = [torch.where(y.eq(i))[0].tolist() for i in assigned]

    return centroids, clusters


@torch.enable_grad()
def crf(scores, mask, target=None, marg=False):
    # get first line of mask matrix
    # (B) actual length, ignore bos, eos, and pad
    lens = mask[:, 0].sum(-1)
    total = lens.sum()
    
    batch_size, seq_len, _ = scores.shape

    training = scores.requires_grad
    # always enable the gradient computation of scores
    # in order for the computation of marginal probs
    # TODO size
    s = inside(scores.requires_grad_(), mask)
    logZ = s[0].gather(0, lens.unsqueeze(0)).sum()
    # marginal probs are used for decoding, and can be computed by
    # combining the inside algorithm and autograd mechanism
    # instead of the entire inside-outside process
    probs = scores
    if marg:
        probs, = autograd.grad(logZ, scores, retain_graph=training)
    if target is None:
        return probs

    loss = (logZ - scores[mask & target].sum()) / total
    return loss, probs


def neg_log_likelihood(scores, spans, mask, s_link=None):
    """
    Args:
        scores (Tensor(batch, seq_len, seq_len)): ...
        spans (Tensor(batch, seq_len)): include <bos> <eos> and <pad>
        mask (Tensor(batch, seq_len)): mask <bos> <eos> and <pad>
        s_link (Tensor(B, L-1)): score for split point
    """

    batch_size, _, _ = scores.shape
    gold_scores = score_function(scores, spans, mask, s_link).sum()
    logZ = partition_function(scores, mask, s_link).sum()
    # TODO batch size or total span ?
    loss = (logZ - gold_scores) / batch_size   # (batch_size)

    return loss


def score_function(scores, spans, mask, s_link=None):
    """[summary]

    Args:
        scores (Tensor(B, L-1, L-1)): ...
        spans (Tensor(B, L-1, L-1)): ground truth word segmentation, see SegmentField for details
        mask (Tensor(B, L-1, L-1)): L include <bos> and <eos>
        s_link (Tensor(B, L-1)): score for split point


    Returns:
        (Tensor(*)): scores of all spans for a batch
    """

    batch_size, seq_len, _, label_size = scores.size()
    lens = mask[:, 0].sum(dim=-1)

    if s_link is None:
        return scores[mask & spans]
    else:
        return scores[mask & spans] + s_link.expand_as(scores)[mask & spans]

def score_function_ner(scores, spans, mask, s_link=None):
    """[summary]

    Args:
        scores (Tensor(B, L-1, L-1, T)): ...
        spans (Tensor(B, L-1, L-1)): ground truth word segmentation, see NeField for details
        mask (Tensor(B, L-1, L-1)): L include <bos> and <eos>
        s_link (Tensor(B, L-1)): score for split point


    Returns:
        (Tensor(*)): scores of all spans for a batch
    """

    batch_size, seq_len, _, label_size = scores.size()
    lens = mask[:, 0].sum(dim=-1)
    # print(scores.shape)
    # # print(scores.nonzero())
    # print(spans)
    # print(spans.nonzero())
    # exit()
    scores = scores[torch.arange(batch_size)[:, None, None], 
                    torch.arange(seq_len)[None, :, None], 
                    torch.arange(seq_len)[None, None, :], 
                    spans]
    # print(scores)
    # print(scores[mask & spans.bool()])
    # exit()
    # print(scores.shape, mask.shape, spans.shape, 
    #       (mask & spans).shape, scores[mask & spans.bool()].shape)
    # print(spans.bool().sum())
    if s_link is None:
        return scores[mask & spans.bool()].double()
    else:
        return scores[mask & spans] + s_link.expand_as(scores)[mask & spans]



def partition_function(scores, mask, s_link=None):
    """
    Args:
        scores (Tensor(B, L-1, L-1)): ...
        mask (Tensor(B, L-1, L-1)): L include <bos> and <eos>
        s_link (Tensor(B, L-1)): score for split point


    Returns:
        (Tensor(B)): logZ
    """

    batch_size, seq_len, _ = scores.size()
    lens = mask[:, 0].sum(dim=-1)

    # s[*, i] is logsumexp score where a sequence segmentation path ending in i
    s = scores.new_zeros(batch_size, seq_len)
    # TODO initial ?
    # s = torch.full_like(scores[:, 0], float("-inf"))

    if s_link is not None:
        scores = scores + s_link

    for i in range(1, seq_len):
        # 0 <= k < i
        # s[:, :i] + scores[:, :i, i] is sum score of segmentation end in i and last word is (k, i)
        s[:, i] = torch.logsumexp(s[:, :i] + scores[:, :i, i], dim=-1)

    return s[torch.arange(batch_size), lens]


@torch.no_grad()
def directed_acyclic_graph(scores, mask, s_link=None):
    """Chinese Word Segmentation with Directed Acyclic Graph.

    Args:
        scores (Tensor(B, N, N)): (*, i, j) is score for span(i, j)
        mask (Tensor(B, N, N))
        s_link (Tensor(B, L-1)): score for split point

    Returns:
        segs (list[]): segmentation sequence
    """

    batch_size, seq_len, _ = scores.size()
    # actual words number: N
    # TODO no need (B, L-1, L-1), (B, L-1) is enough
    lens = mask[:, 0].sum(dim=-1)

    if s_link is not None:
        scores = scores + s_link

    # s[*, i] is max score where a sequence segmentation path ending in i
    s = scores.new_zeros(batch_size, seq_len)
    # backpoint[*, i] is split point k where sequence end in i and (k, i) is last word
    backpoints = scores.new_ones(batch_size, seq_len).long()

    for i in range(1, seq_len):
        # 0 <= k < i
        # s[:, :i] + scores[:, :i, i] is max score of segmentation end in i and last word is (k, i)
        max_values, max_indices = torch.max(s[:, :i] + scores[:, :i, i], dim=-1)
       
        s[:, i] = max_values
        backpoints[:, i] = max_indices

    def backtrack(backpoint, i):
        """

        Args:
            backpoint (list): (backpoint[i], i) is last word of segmentation
            i (int): end of word segmentation

        Returns:
            [type]: [description]
        """
        if i == 0:
            return []
        split = backpoint[i]
        sub_seg = backtrack(backpoint, split)

        return sub_seg + [(split, i)]

    backpoints = backpoints.tolist()
    segs = [backtrack(backpoints[i], length)
            for i, length in enumerate(lens.tolist())]

    return segs


def inside(scores, mask):
    batch_size, seq_len, _ = scores.shape
    # TODO difficult to understand the view of tensor
    # [seq_len, seq_len, batch_size]
    scores, mask = scores.permute(1, 2, 0), mask.permute(1, 2, 0)
    # same shape as scores, but filled with -inf
    s = torch.full_like(scores, float('-inf'))

    for w in range(1, seq_len):
        # n denotes the number of spans to iterate,
        # from span (0, w) to span (n, n+w) given width w
        n = seq_len - w

        # default: offset=0, dim1=0, dim2=1
        # diag_mask is used for ignoring the excess of each sentence
        # [batch_size, n]
        diag_mask = mask.diagonal(w)

        if w == 1:
            s.diagonal(w)[diag_mask] = scores.diagonal(w)[diag_mask]
            continue

        # [n, w, batch_size]
        s_span = stripe(s, n, w-1, (0, 1)) + stripe(s, n, w-1, (1, w), 0)
        # [batch_size, n, w]
        s_span = s_span.permute(2, 0, 1)
        s_span = s_span[diag_mask].logsumexp(-1)
        s.diagonal(w)[diag_mask] = s_span + scores.diagonal(w)[diag_mask]

    return s


def cky(scores, mask):
    lens = mask[:, 0].sum(-1)
    scores = scores.permute(1, 2, 0)
    seq_len, seq_len, batch_size = scores.shape
    s = scores.new_zeros(seq_len, seq_len, batch_size)
    p = scores.new_zeros(seq_len, seq_len, batch_size).long()

    for w in range(1, seq_len):
        n = seq_len - w
        starts = p.new_tensor(range(n)).unsqueeze(0)

        if w == 1:
            s.diagonal(w).copy_(scores.diagonal(w))
            continue
        # [n, w, batch_size]
        s_span = stripe(s, n, w-1, (0, 1)) + stripe(s, n, w-1, (1, w), 0)
        # [batch_size, n, w]
        s_span = s_span.permute(2, 0, 1)
        # [batch_size, n]
        s_span, p_span = s_span.max(-1)
        s.diagonal(w).copy_(s_span + scores.diagonal(w))
        p.diagonal(w).copy_(p_span + starts + 1)

    def backtrack(p, i, j):
        if j == i + 1:
            return [(i, j)]
        split = p[i][j]
        ltree = backtrack(p, i, split)
        rtree = backtrack(p, split, j)
        return [(i, j)] + ltree + rtree

    p = p.permute(2, 0, 1).tolist()
    trees = [backtrack(p[i], 0, length)
             for i, length in enumerate(lens.tolist())]

    return trees

