"""Abstract base class for all recommendation models."""
from abc import ABC, abstractmethod
import torch.nn as nn
import torch


class BaseRecModel(nn.Module, ABC):
    """Interface inherited by all recommendation models.

    Subclasses are required to implement:
      - forward(): return an embedding dictionary (containing at least 'final')
      - loss(users, pos_items, neg_items, **kwargs): return a scalar or a dict
      - predict(users): return a [batch, n_items+1] score matrix
    """

    def __init__(self, data, args):
        super().__init__()
        self.data = data
        self.args = args
        self.n_users = data['n_users']
        self.n_items = data['n_items']

    @abstractmethod
    def forward(self, *args, **kwargs):
        ...

    @abstractmethod
    def loss(self, users, pos_items, neg_items, **kwargs):
        ...

    @abstractmethod
    def predict(self, users):
        ...

    def prepare_eval_cache(self):
        output = self.forward()
        if 'user' in output and 'item' in output:
            return output['user'], output['item']
        final = output['final']
        if isinstance(final, dict):
            return final['user'], final['item']
        return torch.split(final, [self.n_users + 1, self.n_items + 1], dim=0)

    def score_items(self, users, item_ids, cache):
        user_emb, item_emb = cache
        return user_emb[users.long()] @ item_emb[item_ids.long()].T
