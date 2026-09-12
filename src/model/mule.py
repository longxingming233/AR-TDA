"""MULE backbone + optional TDA + optional REGA plugins.

Usage (command-line switches):
    --model mule --use_tda 0 --use_rega 0   equivalent to the original MULE-master
    --model mule --use_tda 1 --use_rega 0   original + topological entropy only
    --model mule --use_tda 1 --use_rega 1   original + TDA + REGA (recommended configuration)

This implementation is aligned with EaMRec:
  1. _propagate performs layer scheduling via args.denoise_layers (progressive/last1/last2/last_half/all/none/first4);
  2. forward fills zero entropy for the corresponding behaviour when H is None, ensuring REGA's gating is always invoked;
  3. forward/loss/predict pass user_indices through to REGA;
  4. it raises an explicit error when 'buy' is missing, preventing TDA from being silently disabled.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseRecModel
from .registry import register_model
from .graph_conv import EntropyAwareGraphConv
from .rega import REGA


@register_model('mule')
class MuLeWithPlugins(BaseRecModel):
    def __init__(self, data, args):
        super().__init__(data, args)
        d = args.emb_dim

        self.bsg_types = list(data['bsg_types'])
        self.tcb_types = list(data['tcb_types'])
        self.tib_types = list(data['tib_types'])
        self.trbg_types = self.tcb_types + self.tib_types
        self.total_behaviors = ['ubg'] + self.bsg_types + self.trbg_types
        self.edge_dict = data['edge_dict']

        # ---------- Topological entropy scheduling parameters aligned with EaMRec ----------
        self.tda_mode = getattr(args, 'tda_mode', 'entropy')
        self.denoise_layers = getattr(args, 'denoise_layers', 'progressive')
        self.denoise_alpha = getattr(args, 'denoise_alpha', 'auto')

        # ---------- Ablation experiment switches ----------
        self.entropy_norm = bool(getattr(args, 'entropy_norm', True))
        self.keep_min_edges = bool(getattr(args, 'keep_min_edges', True))
        self.use_attn_agg = bool(getattr(args, 'use_attn_agg', True))
        self.rega_blend = float(getattr(args, 'rega_blend', 0.1))
        if not 0.0 <= self.rega_blend <= 1.0:
            raise ValueError('rega_blend must be in [0, 1]')

        self.user_emb = nn.Embedding(self.n_users + 1, d, padding_idx=0)
        self.item_emb = nn.Embedding(self.n_items + 1, d, padding_idx=0)

        self.convs = nn.ModuleDict()
        for b in self.total_behaviors:
            if b in self.tcb_types and args.use_tda:
                self.convs[b] = nn.ModuleList([
                    EntropyAwareGraphConv(d, d, norm_type='tda',
                                          tda_mode=args.tda_mode,
                                          tda_min=args.tda_min,
                                          tda_max=args.tda_max,
                                          entropy_norm=self.entropy_norm,
                                          keep_min_edges=self.keep_min_edges,
                                          min_keep_ratio=getattr(args, 'min_keep_ratio', 0.0),
                                          fast_entropy=getattr(args, 'fast_entropy', True),
                    )
                    for _ in range(args.tda_layers)
                ])
            else:
                self.convs[b] = nn.ModuleList([
                    EntropyAwareGraphConv(d, d, norm_type='gcn',
                                          fast_entropy=getattr(args, 'fast_entropy', True))
                    for _ in range(args.gnn_layers)
                ])

        self.aggregator = REGA(
            d,
            behavior_groups=[self.bsg_types, self.trbg_types],
            key_behavior='buy',
            enable_gate=bool(args.use_rega),
            eg_alpha=args.eg_alpha,
            attn_dim=getattr(args, 'rega_attn_dim', 16),
            rega_blend=self.rega_blend,
            use_attn_agg=self.use_attn_agg,
            diagnostics=getattr(args, 'p0_diagnostics', False),
        )

        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        self.aggregator.reset_parameters()

    # ------------------------------------------------------------------
    # Propagation: layer-wise denoising scheduling fully aligned with EaMRec.propagate
    # ------------------------------------------------------------------
    def _propagate(self, x, edges, b, target_emb=None):
        """LightGCN-style multi-layer propagation that decides per layer whether AEGD is enabled based on denoise_layers."""
        result = [x]
        last_ent = None
        num_layers = len(self.convs[b])

        # consistent with EaMRec: soft_entropy takes its own branch, all other modes are unified as 'full'
        base_mode = 'soft_entropy' if self.tda_mode == 'soft_entropy' else 'full'

        for i, conv in enumerate(self.convs[b]):
            denoise_strength = 'none'
            if self.denoise_layers == 'last1':
                denoise_strength = base_mode if (i == num_layers - 1) else 'none'
            elif self.denoise_layers == 'last2':
                denoise_strength = base_mode if (i >= num_layers - 2) else 'none'
            elif self.denoise_layers == 'last_half':
                denoise_strength = base_mode if (i >= num_layers // 2) else 'none'
            elif self.denoise_layers == 'all':
                denoise_strength = base_mode
            elif self.denoise_layers == 'none':
                denoise_strength = 'none'
            elif self.denoise_layers == 'first4':
                denoise_strength = base_mode if (i < 4) else 'none'
            elif self.denoise_layers == 'progressive':
                # consistent with EaMRec: base_mode for the first 3 layers, soft_entropy afterwards
                if i < 3:
                    denoise_strength = base_mode
                elif i < num_layers:
                    denoise_strength = 'soft_entropy'
                else:
                    denoise_strength = 'none'

            out = conv(x, edges, target_emb,
                       denoise_mode=denoise_strength,
                       denoise_alpha=self.denoise_alpha)
            if isinstance(out, tuple):
                x, ent = out
                if ent is not None:
                    last_ent = ent
            else:
                x = out
            x = F.normalize(x, dim=1)
            result.append(x / (i + 1))
        return torch.stack(result, dim=1).sum(dim=1), last_ent

    # ------------------------------------------------------------------
    # Forward: zero-fill entropy + pass user_indices through
    # ------------------------------------------------------------------
    def forward(self, user_indices=None):
        emb_dict, ent_dict = {}, {}
        init = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)

        ubg, _ = self._propagate(init, self.edge_dict['ubg'], 'ubg')
        emb_dict['ubg'] = ubg

        for b in self.bsg_types:
            emb_dict[b], _ = self._propagate(ubg, self.edge_dict[b], b)

        for b in self.tib_types:
            prev = b.split('_')[0] if '_' in b else 'ubg'
            base = emb_dict.get(prev, ubg)
            emb_dict[b], _ = self._propagate(base, self.edge_dict[b], b)

        # consistent with EaMRec: when TDA is enabled, 'buy' is required to already be in emb_dict
        if self.args.use_tda and 'buy' not in emb_dict:
            raise KeyError(
                "when use_tda=1, 'buy' must appear in bsg_types, otherwise TDA's target_emb is missing; "
                "include 'buy' in the bsg_types of statistics.json."
            )

        for b in self.tcb_types:
            prev = b.split('_')[0] if '_' in b else 'ubg'
            base = emb_dict.get(prev, ubg)
            tgt = emb_dict.get('buy') if self.args.use_tda else None
            e, H = self._propagate(base, self.edge_dict[b], b, target_emb=tgt)
            emb_dict[b] = e
            # even when H is None, fill a zero tensor so that REGA's entropy_gate is always invoked
            if H is not None:
                ent_dict[b] = H
            else:
                num_nodes = init.size(0)
                ent_dict[b] = torch.zeros(num_nodes, device=init.device)

        emb_dict['final'] = self.aggregator(emb_dict, ent_dict,
                                            user_indices=user_indices)
        return emb_dict

    # ------------------------------------------------------------------
    # Loss / prediction: pass user_indices through
    # ------------------------------------------------------------------
    def loss(self, users, pos_items, neg_items, **kwargs):
        emb = self.forward(user_indices=users)['final']
        ue, ie = torch.split(emb, [self.n_users + 1, self.n_items + 1], dim=0)
        ps = (ue[users] * ie[pos_items]).sum(-1)
        ns = (ue[users] * ie[neg_items]).sum(-1)
        loss = -F.logsigmoid(ps - ns).mean()
        return loss

    def predict(self, users):
        emb = self.forward(user_indices=users)['final']
        ue, ie = torch.split(emb, [self.n_users + 1, self.n_items + 1], dim=0)
        return ue[users.long()] @ ie.T
