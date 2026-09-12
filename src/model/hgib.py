"""HGIB (RecSys 2025) backbone + optional TDA + optional REGA plugins.

Follows the fusion pattern of MULE + AR_TDA.

Usage (command-line switches):
    --model hgib --use_tda 0 --use_rega 0   equivalent to the original HGIB
    --model hgib --use_tda 1 --use_rega 0   original + topological entropy only
    --model hgib --use_tda 1 --use_rega 1   original + TDA + REGA (recommended configuration)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseRecModel
from .registry import register_model
from .graph_conv import EntropyAwareGraphConv
from .rega import REGA


class Fusion(nn.Module):
    """Fusion module of the original HGIB (kept consistent with the original)."""
    def __init__(self, emb_dim, bsg_types, trbg_types):
        super(Fusion, self).__init__()
        self.emb_dim = emb_dim
        self.bsg_types = bsg_types
        self.trbg_types = trbg_types
        self.proj_layers = nn.ModuleList([
            nn.Linear(emb_dim, emb_dim, bias=True),
            nn.Linear(emb_dim, emb_dim, bias=True)
        ])
        
    def reset_parameters(self):
        for layer in self.proj_layers:
            nn.init.xavier_uniform_(layer.weight)
    
    def forward(self, emb_dict):
        key = emb_dict['buy']
        for i, behavior_types in enumerate([self.bsg_types, self.trbg_types]):
            proj_key = self.proj_layers[i](key)
            query_embs = torch.stack([emb_dict[bt] for bt in behavior_types], dim=1)
            scores = torch.einsum('nd,nbd->nb', proj_key, query_embs) / (self.emb_dim ** 0.5)
            attention = scores.softmax(dim=1).unsqueeze(-1)
            key = (attention * query_embs).sum(dim=1)
        return key


def gumbel_sigmoid(logits, tau=1, hard=True, threshold=0.05):
    """Gumbel-Sigmoid and straight-through weights consistent with the official HGIB."""
    gumbels = -torch.empty_like(logits).exponential_().log()
    y_soft = ((logits + gumbels) / tau).sigmoid()

    if hard:
        y_hard = torch.zeros_like(logits)
        y_hard[y_soft > threshold] = 1.0
        return y_hard * y_soft - y_soft.detach() + y_soft
    return y_soft


def kernel_matrix(x, sigma):
    return torch.exp((torch.matmul(x, x.transpose(0, 1)) - 1) / sigma)


def hsic(Kx, Ky, m):
    if m < 2:
        return Kx.new_zeros(())
    Kxy = torch.mm(Kx, Ky)
    h = torch.trace(Kxy) / m ** 2 + torch.mean(Kx) * torch.mean(Ky) - \
        2 * torch.mean(Kxy) / m
    return h * (m / (m - 1)) ** 2


class EmbLoss(nn.Module):
    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings, require_pow=False):
        if require_pow:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.pow(
                    input=torch.norm(embedding, p=self.norm), exponent=self.norm
                )
            emb_loss /= embeddings[-1].shape[0]
            emb_loss /= self.norm
            return emb_loss
        else:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.norm(embedding, p=self.norm)
            emb_loss /= embeddings[-1].shape[0]
            return emb_loss


@register_model('hgib')
class HGIB(BaseRecModel):
    def __init__(self, data, args):
        super().__init__(data, args)
        d = args.emb_dim
        self.edge_dict = data['edge_dict']
        
        self.bsg_types = list(data['bsg_types'])
        self.tcb_types = list(data['tcb_types'])
        self.tib_types = list(data['tib_types'])
        self.trbg_types = self.tcb_types + self.tib_types
        self.total_behaviors = ['ubg'] + self.bsg_types + self.trbg_types
        
        # ---------- AR_TDA parameters ----------
        self.use_tda = bool(args.use_tda)
        self.use_rega = bool(args.use_rega)
        self.tda_mode = getattr(args, 'tda_mode', 'entropy')
        self.denoise_layers = getattr(args, 'denoise_layers', 'progressive')
        self.denoise_alpha = getattr(args, 'denoise_alpha', 'auto')
        self.entropy_norm = bool(getattr(args, 'entropy_norm', True))
        self.keep_min_edges = bool(getattr(args, 'keep_min_edges', True))
        self.use_attn_agg = bool(getattr(args, 'use_attn_agg', True))
        self.rega_blend = float(getattr(args, 'rega_blend', 0.1))
        if not 0.0 <= self.rega_blend <= 1.0:
            raise ValueError('rega_blend must be in [0, 1]')
        
        # Original HGIB hyper-parameters
        self.beta = getattr(args, 'beta', 50.0)
        self.alpha = getattr(args, 'alpha', 1.0)
        self.threshold = getattr(args, 'threshold', 0.05)
        self.sigma = getattr(args, 'sigma', 1.0)
        self.temperature = 1
        
        self.dropout = nn.Dropout(0.1)
        self.ce_loss = nn.CrossEntropyLoss()
        self.reg_loss = EmbLoss()

        self.user_emb = nn.Embedding(self.n_users + 1, d, padding_idx=0)
        self.item_emb = nn.Embedding(self.n_items + 1, d, padding_idx=0)
        
        # ---------- Dynamically build graph convolution layers (with TDA support) ----------
        # Original HGIB: 3 GCN layers for TCB behaviours, 1 GCN layer for the others
        self.convs = nn.ModuleDict()
        for behavior_type in self.total_behaviors:
            if behavior_type in self.tcb_types and self.use_tda:
                # TDA mode: use EntropyAwareGraphConv + more layers
                self.convs[behavior_type] = nn.ModuleList([
                    EntropyAwareGraphConv(
                        d, d, norm_type='tda',
                        tda_mode=self.tda_mode,
                        tda_min=getattr(args, 'tda_min', 0.3),
                        tda_max=getattr(args, 'tda_max', 2.0),
                        entropy_norm=self.entropy_norm,
                        keep_min_edges=self.keep_min_edges,
                        min_keep_ratio=getattr(args, 'min_keep_ratio', 0.0),
                        fast_entropy=getattr(args, 'fast_entropy', True),
                    ) for _ in range(getattr(args, 'tda_layers', 3))
                ])
            elif behavior_type in self.tcb_types and not self.use_tda:
                # Original mode: TCB behaviours always use 3 GCN layers
                self.convs[behavior_type] = nn.ModuleList([
                    EntropyAwareGraphConv(d, d, norm_type='gcn',
                                          fast_entropy=getattr(args, 'fast_entropy', True))
                    for _ in range(3)
                ])
            else:
                # Non-TCB behaviours always use 1 GCN layer
                self.convs[behavior_type] = nn.ModuleList([
                    EntropyAwareGraphConv(d, d, norm_type='gcn',
                                          fast_entropy=getattr(args, 'fast_entropy', True))
                    for _ in range(1)
                ])

        # ---------- Aggregator selection ----------
        if self.use_rega:
            # Use the REGA aggregator
            self.aggregator = REGA(
                d,
                behavior_groups=[self.bsg_types, self.trbg_types],
                key_behavior='buy',
                enable_gate=self.use_rega,
                eg_alpha=getattr(args, 'eg_alpha', 0.3),
                use_attn_agg=self.use_attn_agg,
                attn_dim=getattr(args, 'rega_attn_dim', 16),
                rega_blend=getattr(args, 'rega_blend', 0.1),
                diagnostics=getattr(args, 'p0_diagnostics', False),
            )
        else:
            # Use the original Fusion module (kept consistent with the original HGIB)
            self.aggregator = Fusion(d, self.bsg_types, self.trbg_types)

        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        self.aggregator.reset_parameters()
            
    def graph_learner(self, behavior_type, adj_matrix, emb_table):
        row, col = adj_matrix[0], adj_matrix[1]
        row_emb = emb_table[row]
        col_emb = emb_table[col]
        logit = torch.sum(row_emb * col_emb, -1)
        logit = logit.view(-1)
        weights = gumbel_sigmoid(logit, tau=1, threshold=self.threshold)
        weights = weights + 1e-7
        return weights
    
    def _propagate(self, x, edges, b, target_emb=None, weight=None):
        """Multi-layer propagation that decides per layer whether AEGD is enabled based on denoise_layers."""
        result = [x]
        last_ent = None
        num_layers = len(self.convs[b])

        if self.use_tda:
            # TDA mode: use layer-wise denoising scheduling
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
                    if i < 3:
                        denoise_strength = base_mode
                    elif i < num_layers:
                        denoise_strength = 'soft_entropy'
                    else:
                        denoise_strength = 'none'

                out = conv(x, edges, target_emb,
                           denoise_mode=denoise_strength,
                           denoise_alpha=self.denoise_alpha,
                           edge_weight=weight)
                if isinstance(out, tuple):
                    x, ent = out
                    if ent is not None:
                        last_ent = ent
                else:
                    x = out
                x = F.normalize(x, dim=1)
                result.append(x / (i + 1))
        else:
            # Original mode: propagation identical to the original HGIB
            for i, conv in enumerate(self.convs[b]):
                # EntropyAwareGraphConv always returns an (out, node_entropy) tuple
                # pass weight to the graph convolution; these are the edge weights learned by HGIB's graph_learner
                out = conv(x, edges, target_emb, edge_weight=weight)
                if isinstance(out, tuple):
                    x, _ = out
                else:
                    x = out
                x = F.normalize(x, dim=1)
                result.append(x)  # not divided by (i+1)
        
        return torch.stack(result, dim=1).sum(dim=1), last_ent
    
    def forward(self, user_indices=None):
        emb_dict = {}
        ent_dict = {}
        
        init_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        emb_dict['init'] = init_emb
        
        # UBG propagation
        weight = self.graph_learner('ubg', self.edge_dict['ubg'], init_emb)
        ubg_emb, _ = self._propagate(init_emb, self.edge_dict['ubg'], 'ubg', weight=weight)
        emb_dict['ubg'] = ubg_emb
        
        # BSG propagation
        for behavior_type in self.bsg_types:
            weight = self.graph_learner(behavior_type, self.edge_dict[behavior_type], emb_dict["ubg"])
            bsg_emb, _ = self._propagate(self.dropout(ubg_emb), self.edge_dict[behavior_type], behavior_type, weight=weight)
            emb_dict[behavior_type] = bsg_emb
        
        # TIB propagation
        for behavior_type in self.tib_types:
            previous_behavior = behavior_type.split('_')[0]
            weight = self.graph_learner(behavior_type, self.edge_dict[behavior_type], emb_dict[previous_behavior])
            previous_emb = emb_dict[previous_behavior]
            tib_emb, _ = self._propagate(self.dropout(previous_emb), self.edge_dict[behavior_type], behavior_type, weight=weight)
            emb_dict[behavior_type] = tib_emb

        # TCB propagation (critical path for TDA)
        for behavior_type in self.tcb_types:
            previous_behavior = behavior_type.split('_')[0]
            weight = self.graph_learner(behavior_type, self.edge_dict[behavior_type], emb_dict[previous_behavior])
            previous_emb = emb_dict[previous_behavior]
            target_emb = emb_dict.get('buy')
            tcb_emb, H = self._propagate(self.dropout(previous_emb), self.edge_dict[behavior_type], behavior_type, target_emb, weight=weight)
            emb_dict[behavior_type] = tcb_emb

            if H is not None:
                ent_dict[behavior_type] = H
            else:
                num_nodes = init_emb.size(0)
                ent_dict[behavior_type] = torch.zeros(num_nodes, device=init_emb.device)

        # Aggregator
        if self.use_rega:
            # REGA mode: pass ent_dict and user_indices
            final_emb = self.aggregator(emb_dict, ent_dict, user_indices=user_indices)
        else:
            # Original mode: pass only emb_dict (consistent with the original HGIB)
            final_emb = self.aggregator(emb_dict)
        emb_dict['final'] = final_emb

        return emb_dict
    
    def hsic_graph(self, users_emb1, items_emb1, users_emb2, items_emb2):
        input_x = F.normalize(users_emb1, p=2, dim=1)
        input_y = F.normalize(users_emb2, p=2, dim=1)
        Kx = kernel_matrix(input_x, self.sigma)
        Ky = kernel_matrix(input_y, self.sigma)
        loss_user = hsic(Kx, Ky, users_emb1.shape[0])
        
        input_i = F.normalize(items_emb1, p=2, dim=1)
        input_j = F.normalize(items_emb2, p=2, dim=1)
        Ki = kernel_matrix(input_i, self.sigma)
        Kj = kernel_matrix(input_j, self.sigma)
        loss_item = hsic(Ki, Kj, users_emb1.shape[0])
        return loss_user + loss_item
    
    def cl_loss(self, x1, x2):
        pos_score = (x1 * x2).sum(dim=-1)
        pos_score = torch.exp(pos_score / self.temperature)
        ttl_score = torch.matmul(x1, x2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / self.temperature).sum(dim=1)
        return -torch.log(pos_score / ttl_score).mean()
    
    def loss(self, users, pos_items, neg_items, **kwargs):
        user_indices = kwargs.get('user_indices', users)
        emb_dict = self.forward(user_indices=user_indices)
        user_emb, item_emb = torch.split(emb_dict['final'], [self.n_users + 1, self.n_items + 1], dim=0)
        
        # Prepare the separated user/item embeddings for each behaviour
        for behavior in self.total_behaviors + ["init"]:
            u, i = torch.split(emb_dict[behavior], [self.n_users + 1, self.n_items + 1], dim=0)
            emb_dict[behavior] = {'user': u, 'item': i}

        # HSIC loss
        pt_loss = self.beta * self.hsic_graph(
            emb_dict["ubg"]["user"][users], emb_dict["ubg"]["item"][pos_items],
            emb_dict["init"]["user"][users], emb_dict["init"]["item"][pos_items]
        )
        
        for behavior in self.bsg_types:
            pt_loss += self.beta * self.hsic_graph(
                emb_dict[behavior]["user"][users], emb_dict[behavior]["item"][pos_items],
                emb_dict["ubg"]["user"][users], emb_dict["ubg"]["item"][pos_items]
            )
            
        for behavior in self.trbg_types:
            prev_behavior = behavior.split('_')[0]
            pt_loss += self.beta * self.hsic_graph(
                emb_dict[behavior]["user"][users], emb_dict[behavior]["item"][pos_items],
                emb_dict[prev_behavior]["user"][users], emb_dict[prev_behavior]["item"][pos_items]
            )

        # CL loss
        for behavior in self.bsg_types + ["ubg"]:
            pt_loss += self.alpha * self.cl_loss(user_emb[users], emb_dict[behavior]["user"][users])
            pt_loss += self.alpha * self.cl_loss(item_emb[pos_items], emb_dict[behavior]["item"][pos_items])
        
        reg_loss = self.reg_loss(user_emb, item_emb, require_pow=True)
        logits = torch.matmul(user_emb[users], item_emb.transpose(0, 1))

        return self.ce_loss(logits, pos_items) + pt_loss + 0.1 * reg_loss
    
    def predict(self, users):
        final_embeddings = self.forward(user_indices=users)['final']
        final_user_emb, final_item_emb = torch.split(final_embeddings, [self.n_users + 1, self.n_items + 1])
        user_emb = final_user_emb[users.long()]
        return user_emb @ final_item_emb.T
