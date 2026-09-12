import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.typing import OptTensor
from torch import Tensor
from torch_scatter import scatter_add, scatter_softmax, scatter_max
from torch_geometric.nn.conv import MessagePassing


class EntropyAwareGraphConv(MessagePassing):
    """
    Entropy-Aware Graph Convolution Layer (EATD/AEGD).
    Supports two normalization modes: GCN and Adaptive Entropy-based Graph Denoising (AEGD).
    The three AEGD sub-modes are:
    - soft: full weighting (soft denoising, no truncation at all, only scatter_softmax over attention scores)
    - hard: mean-based truncation (hard denoising)
    - entropy: dynamic truncation based on distribution entropy (adaptive denoising with a learnable boundary)

    Ablation switches:
    - entropy_norm=False    -> compute_entropy no longer divides by ln(max(d,2)) (w/o entropy normalisation)
    - keep_min_edges=False  -> after hard/entropy truncation, no fallback is applied to isolated nodes (w/o minimum retention ratio)

    Generalization design points:
    1) when target_emb=None and allow_self_attention=True it degrades to <x,x> self-attention,
       so that this module can still produce node topological entropy on single-behaviour / no-primary-behaviour models;
    2) softmax uses scatter_max to take the local maximum, avoiding numerical underflow caused by a global max;
    3) compute_entropy normalises by ln(max(d,2)) to remove the degree bias (can be disabled by entropy_norm).
    """

    def __init__(self, in_channels, out_channels, norm_type,
                 tda_mode='soft', tda_min=0.3, tda_max=2.0,
                 entropy_norm=True, keep_min_edges=True, min_keep_ratio=0.0,
                 fast_entropy=True):
        super(EntropyAwareGraphConv, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_type = norm_type
        self.tda_mode = tda_mode
        self.entropy_norm = entropy_norm
        self.keep_min_edges = keep_min_edges
        self.min_keep_ratio = float(min_keep_ratio)
        self.fast_entropy = bool(fast_entropy)
        self.denoise_alpha = 0.0

        if self.tda_mode == 'entropy':
            if not 0.0 < tda_min < 1.0:
                raise ValueError('tda_min must be in (0, 1) for entropy mode')
            if tda_max <= tda_min:
                raise ValueError('tda_max must be greater than tda_min')
            min_logit = torch.logit(torch.tensor(float(tda_min)).clamp(1e-4, 1 - 1e-4))
            span = torch.tensor(float(tda_max - tda_min)).clamp(min=1e-4)
            span_raw = torch.log(torch.expm1(span))
            self.tda_min_param = nn.Parameter(min_logit)
            self.tda_span = nn.Parameter(span_raw)
        else:
            self.tda_min = tda_min
            self.tda_max = tda_max

        self.last_total_edges = 0
        self.last_kept_edges = 0
        self.last_threshold_min = None
        self.last_threshold_max = None

    # LightGCN normalization (symm = D^{-1/2} A D^{-1/2})
    def gcn_norm(self, edge_index, num_nodes, edge_weight=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt = deg_inv_sqrt.masked_fill(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    # PKEF-style left normalization (D^{-1} A) -- consistent with the original PKEF normalization='left'
    def left_norm(self, edge_index, num_nodes):
        edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv = deg.pow(-1.0)
        deg_inv = deg_inv.masked_fill(deg_inv == float('inf'), 0)
        return edge_index, deg_inv[col] * edge_weight

    def compute_entropy(self, attention_score, key, num_nodes):
        """Compute the Shannon entropy H_v of each node's interaction distribution.

        High entropy -> diffuse neighbour distribution -> likely noise; low entropy -> concentrated neighbour distribution -> clear preference.
        If entropy_norm=True, divide by ln(max(d,2)) for degree normalisation.
        """
        attention_score = attention_score.squeeze(-1)
        if self.fast_entropy:
            degree = scatter_add(torch.ones_like(attention_score), key, dim=0, dim_size=num_nodes)
            mean_score = scatter_add(attention_score.detach(), key, dim=0, dim_size=num_nodes) / degree.clamp_min(1.0)
            centered = attention_score - mean_score[key]
            variance = scatter_add(centered.detach().square(), key, dim=0, dim_size=num_nodes) / degree.clamp_min(1.0)
            dispersion = variance.sqrt()
            if self.entropy_norm:
                dispersion = dispersion / torch.log(degree.clamp_min(2.0))
            entropy = torch.sigmoid(dispersion).masked_fill(degree < 2, 0.0)
            return entropy
        local_max_score, _ = scatter_max(attention_score.detach(), key, dim=0, dim_size=num_nodes)
        local_max_score[local_max_score == float('-inf')] = 0.0
        attn_exp = torch.exp(attention_score - local_max_score[key])
        attn_sum = scatter_add(attn_exp, key, dim=0, dim_size=num_nodes)
        probs = attn_exp / (attn_sum[key] + 1e-6)
        log_probs = torch.log(probs + 1e-6)
        entropy = -scatter_add(probs * log_probs, key, dim=0, dim_size=num_nodes)

        if self.entropy_norm:
            degree = scatter_add(torch.ones_like(attention_score), key, dim=0, dim_size=num_nodes)
            normalizable = degree >= 2
            normalized = torch.zeros_like(entropy)
            normalized[normalizable] = entropy[normalizable] / torch.log(degree[normalizable])
            entropy = normalized.clamp(0.0, 1.0)
        return entropy

    def _mix_alpha(self, node_entropy, key, value):
        if isinstance(value, str):
            if value.lower() != 'auto':
                raise ValueError("denoise_alpha must be 'auto' or a value in [0, 1]")
            return node_entropy[key].detach().clamp(0.0, 1.0)
        alpha = float(value)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError('denoise_alpha must be in [0, 1]')
        return alpha

    def _apply_min_keep(self, mask, key, num_nodes, attention_score=None):
        mask = mask.to(torch.bool)
        """Fallback for truncated nodes, retaining at least 1 edge or the minimum incoming-edge ratio."""
        if not self.keep_min_edges and self.min_keep_ratio <= 0:
            return mask

        degree = scatter_add(torch.ones_like(mask, dtype=torch.float), key, dim=0, dim_size=num_nodes)
        kept_count = scatter_add(mask.float(), key, dim=0, dim_size=num_nodes)
        target_count = torch.zeros_like(kept_count)

        if self.keep_min_edges:
            target_count = torch.maximum(target_count, (degree > 0).float())
        if self.min_keep_ratio > 0:
            ratio_count = torch.ceil(degree * self.min_keep_ratio).clamp(max=degree)
            target_count = torch.maximum(target_count, ratio_count)

        need_nodes = (kept_count < target_count) & (degree > 0)
        if not torch.any(need_nodes):
            return mask
        if attention_score is None:
            return mask | need_nodes[key]

        new_mask = mask.clone()
        if self.min_keep_ratio <= 0:
            _, best_edge = scatter_max(attention_score, key, dim=0, dim_size=num_nodes)
            valid = need_nodes & (best_edge >= 0) & (best_edge < mask.numel())
            new_mask[best_edge[valid]] = True
            return new_mask

        # Ratio-based edge retention is used only for experimental ablation; the default path uses the fully vectorised single-edge fallback above.
        for node in torch.nonzero(need_nodes, as_tuple=False).flatten().tolist():
            edge_pos = torch.nonzero(key == node, as_tuple=False).flatten()
            k = min(int(target_count[node].item()), edge_pos.numel())
            if k > 0:
                _, top_idx = torch.topk(attention_score[edge_pos], k=k, largest=True)
                new_mask[edge_pos[top_idx]] = True
        return new_mask

    def aegd_norm(self, edge_index, x, target_emb, num_nodes, prior_edge_weight=None):
        """Adaptive Entropy-based Graph Denoising normalization.

        Generalization compatibility: when target_emb is missing it degrades to self-attention scoring
        s_{vu} = <x_v, x_u>, so that this module can still compute node entropy and run AEGD
        on single-behaviour / no-primary-behaviour graph models (it is not short-circuited by the outer forward).

        The three sub-mode branches are:
          - 'soft'    : softmax weighting only, no edge truncation at all (w/o AEGD truncation)
          - 'hard'    : mean-based truncation + (optional fallback)
          - 'entropy' : learnable adaptive entropy-based truncation + (optional fallback)
        """
        key, query = edge_index[0], edge_index[1]
        if target_emb is None:
            attention_score = (x[key] * x[query]).sum(dim=-1)
        else:
            attention_score = (target_emb[key] * x[query]).sum(dim=-1)
        if prior_edge_weight is not None:
            if prior_edge_weight.numel() != attention_score.numel():
                raise ValueError('prior_edge_weight must align with edge_index')
            attention_score = attention_score + torch.log(prior_edge_weight.clamp(min=1e-8))

        node_entropy = None
        edge_gate = None
        self.last_total_edges = int(attention_score.numel())
        self.last_kept_edges = self.last_total_edges
        self.last_threshold_min = None
        self.last_threshold_max = None

        if self.tda_mode in ('soft', 'soft_entropy'):
            # Full weighting: no truncation at all, edge importance is distinguished only by softmax
            # Entropy is still computed for use by REGA
            node_entropy = self.compute_entropy(
                attention_score.unsqueeze(-1), key, num_nodes)

        elif self.tda_mode == 'hard':
            node_entropy = self.compute_entropy(
                attention_score.unsqueeze(-1), key, num_nodes)
            mean_score = scatter_add(attention_score, key, dim=0, dim_size=num_nodes) / \
                         (scatter_add(torch.ones_like(attention_score), key, dim=0, dim_size=num_nodes) + 1e-6)
            mask = (attention_score >= mean_score[key])
            mask = self._apply_min_keep(mask, key, num_nodes, attention_score)
            soft_gate = torch.sigmoid((attention_score - mean_score[key]) / 0.1)

            if self.training:
                alpha = self._mix_alpha(node_entropy, key, self.denoise_alpha)
                hard_gate = mask.to(attention_score.dtype)
                mixed_gate = (1.0 - alpha) * hard_gate + alpha * soft_gate
                edge_gate = mixed_gate.detach() - soft_gate.detach() + soft_gate
            else:
                edge_index = edge_index[:, mask]
                attention_score = attention_score[mask]
                key = key[mask]

        elif self.tda_mode == 'entropy':
            entropy = self.compute_entropy(attention_score.unsqueeze(-1), key, num_nodes)
            node_entropy = entropy
            degree = scatter_add(
                torch.ones_like(attention_score), key, dim=0, dim_size=num_nodes,
            )
            valid_entropy = entropy[degree >= 2]
            if valid_entropy.numel() >= 2:
                mean_entropy = valid_entropy.mean()
                std_entropy = valid_entropy.std(unbiased=False).clamp(min=1e-6)
            else:
                mean_entropy = entropy.new_tensor(0.0)
                std_entropy = entropy.new_tensor(1.0)
            entropy_normalized = (entropy[key] - mean_entropy) / std_entropy
            threshold_multiplier = 1.0 + entropy_normalized

            tda_min_dynamic = torch.sigmoid(self.tda_min_param)
            tda_max_dynamic = tda_min_dynamic + F.softplus(self.tda_span)

            tda_min_dynamic = torch.clamp(tda_min_dynamic, min=0.01, max=0.99)
            tda_max_dynamic = torch.min(
                torch.max(tda_max_dynamic, tda_min_dynamic + 1e-3),
                torch.tensor(5.0, device=tda_max_dynamic.device)
            )
            self.last_threshold_min = float(tda_min_dynamic.detach())
            self.last_threshold_max = float(tda_max_dynamic.detach())

            threshold_multiplier = torch.clamp(threshold_multiplier, min=tda_min_dynamic, max=tda_max_dynamic)
            score_std = scatter_add(
                (attention_score - scatter_add(attention_score, key, dim=0, dim_size=num_nodes)[key] /
                 (degree[key] + 1e-6)).pow(2),
                key, dim=0, dim_size=num_nodes,
            )
            score_std = torch.sqrt(score_std / (degree + 1e-6) + 1e-6)
            mean_score = scatter_add(attention_score, key, dim=0, dim_size=num_nodes) / (degree + 1e-6)
            dynamic_threshold = mean_score[key] + score_std[key] * (threshold_multiplier - 1.0)
            soft_gate = torch.sigmoid((attention_score - dynamic_threshold) / 0.1)
            hard_mask = (attention_score >= dynamic_threshold)
            mask = self._apply_min_keep(hard_mask, key, num_nodes, attention_score)
            kept_per_node = scatter_add(mask.to(attention_score.dtype), key, dim=0, dim_size=num_nodes)
            missing = kept_per_node[key] == 0
            self.last_kept_edges = int(mask.sum().item())

            if self.training:
                alpha = self._mix_alpha(node_entropy, key, self.denoise_alpha)
                hard_gate = mask.to(attention_score.dtype)
                mixed_gate = (1.0 - alpha) * hard_gate + alpha * soft_gate
                edge_gate = mixed_gate.detach() - soft_gate.detach() + soft_gate
            else:
                edge_index = edge_index[:, mask]
                attention_score = attention_score[mask]
                key = key[mask]

        if self.tda_mode != 'entropy':
            self.last_kept_edges = int(attention_score.numel())
        attention_score = attention_score.unsqueeze(-1)
        edge_weight = scatter_softmax(attention_score, key, dim=0, dim_size=num_nodes)
        if edge_gate is not None:
            edge_weight = edge_weight.squeeze(-1) * edge_gate
            normalizer = scatter_add(edge_weight, key, dim=0, dim_size=num_nodes)
            edge_weight = (edge_weight / normalizer[key].clamp(min=1e-8)).unsqueeze(-1)
        return edge_index, edge_weight, node_entropy

    def forward(self, x, edge_index, target_emb=None, compute_entropy=True,
                denoise_mode=None, denoise_alpha=0.0,
                allow_self_attention=False, edge_weight=None):
        """
        Generalization parameter notes:
        - when target_emb=None: by default it still follows "no target -> use gcn_norm" for backward compatibility;
          to enable AEGD on single-behaviour / no-primary-behaviour models, pass
          allow_self_attention=True explicitly, which replaces target_emb with the self-attention <x,x>.
        - denoise_mode/denoise_alpha: reserved for multi-layer scheduling (used by the outer propagate).
        - edge_weight: edge weights passed in from outside (used by the graph_learner of models such as HGIB).
        """
        num_nodes = x.size(0)
        node_entropy = None
        # Preserve tensor-valued schedules instead of coercing them to Python floats.
        self.denoise_alpha = denoise_alpha

        has_target = target_emb is not None or allow_self_attention

        if denoise_mode is not None:
            use_aegd = (self.norm_type == 'tda') and (denoise_mode != 'none') and has_target
        else:
            use_aegd = (self.norm_type == 'tda') and compute_entropy and has_target

        if self.norm_type == 'tda':
            if use_aegd:
                edge_index, edge_weight, node_entropy = self.aegd_norm(
                    edge_index, x, target_emb, num_nodes, prior_edge_weight=edge_weight)
            else:
                # when AEGD is not enabled, choose the fallback based on fallback_norm (default gcn for backward compatibility)
                fb = getattr(self, 'fallback_norm', 'gcn')
                if fb == 'left':
                    edge_index, edge_weight = self.left_norm(edge_index, num_nodes)
                else:
                    # both 'gcn' and 'symm' are treated as symm normalization (D^{-1/2} A D^{-1/2})
                    edge_index, edge_weight = self.gcn_norm(edge_index, num_nodes, edge_weight)
        elif self.norm_type in ('gcn', 'symm'):
            # 'gcn' and 'symm' are equivalent aliases, both being D^{-1/2} A D^{-1/2}
            edge_index, edge_weight = self.gcn_norm(edge_index, num_nodes, edge_weight)
        elif self.norm_type == 'left':
            edge_index, edge_weight = self.left_norm(edge_index, num_nodes)
        else:
            raise ValueError('Invalid normalization type')

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        return out, node_entropy

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j
