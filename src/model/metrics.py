"""Top-K evaluation metrics: HR@K (hit rate) and NDCG@K (discounted cumulative gain).

Fully aligned with EaMRec: HR uses the (cumsum>0).int() "hit or not" form,
NDCG uses dcg/idcg, where idcg is truncated based on min(pos_len, K).
"""
import numpy as np


def hit(pos_index, pos_len):
    """Hit@K (HR@K) - consistent with EaMRec.

    pos_index: [N, K] bool/int, each element indicates whether that position hits a positive sample.
    pos_len:   [N]   number of true positives per sample (user, pos_item) (unused, kept only for signature compatibility).
    Returns [N, K], where column k indicates whether top-(k+1) hits at least once.
    """
    result = np.cumsum(pos_index, axis=1)
    return (result > 0).astype(int)


def ndcg(pos_index, pos_len):
    """NDCG@K - consistent with EaMRec."""
    len_rank = np.full_like(pos_len, pos_index.shape[1])
    idcg_len = np.where(pos_len > len_rank, len_rank, pos_len)

    iranks = np.zeros_like(pos_index, dtype=np.float64)
    iranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
    idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=1)
    for row, idx in enumerate(idcg_len):
        if idx <= 0:
            continue
        idcg[row, idx:] = idcg[row, idx - 1]

    ranks = np.zeros_like(pos_index, dtype=np.float64)
    ranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
    dcg = 1.0 / np.log2(ranks + 1)
    dcg = np.cumsum(np.where(pos_index, dcg, 0), axis=1)

    idcg = np.maximum(idcg, 1e-12)
    return dcg / idcg
