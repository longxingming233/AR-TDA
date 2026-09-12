import torch
import torch.nn as nn
import torch.nn.functional as F


class RobustEntropyGate(nn.Module):
    """Robust entropy-guided gating module: BatchNorm1d normalization + GELU activation (Robust Entropy Gate).

    Design points:
    - nn.BatchNorm1d(1) automatically maintains running_mean/running_var,
      so the gate output stays consistent for the same user at training/evaluation time;
    - during training, BN is bypassed when batch=1 to avoid PyTorch raising a ValueError;
    - output is sigmoid(MLP(BN(H))/2) in (0,1); values closer to 1 indicate higher entropy (more noise).
    """

    def __init__(self, emb_dim):
        super().__init__()
        hidden_dim = max(16, emb_dim // 2)
        self.bn = nn.BatchNorm1d(1)
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, raw_entropy):
        """
        raw_entropy: [batch_size, 1] raw topological entropy
        """
        if self.training and raw_entropy.size(0) < 2:
            norm_ent = raw_entropy - raw_entropy.detach()
        else:
            norm_ent = self.bn(raw_entropy)
        logits = self.net(norm_ent)
        gate_value = torch.sigmoid(logits / 2.0)
        return gate_value


class REGA(nn.Module):
    """Robust Entropy-Gated Aggregator (soft penalty scheme).

    Generalization design:
    - legacy call compatibility: REGA(emb_dim, bsg_types, trbg_types, args)
      automatically builds behavior_groups=[bsg_types, trbg_types] and key_behavior='buy'.
    - general call: REGA(emb_dim, behavior_groups=[g1, g2, ...], key_behavior='target',
                   enable_gate=True, eg_alpha=0.2)
      - behavior_groups can be any number of groups (1, 2, N); each group shares one attention linear head.
      - when key_behavior=None, the mean of all behaviour embeddings is used as the query.
    """

    def __init__(self, emb_dim, bsg_types=None, trbg_types=None, args=None,
                 behavior_groups=None, key_behavior='buy',
                 enable_gate=None, eg_alpha=None, use_attn_agg=True,
                 attn_dim=None, rega_blend=1.0, diagnostics=False):
        super(REGA, self).__init__()
        self.emb_dim = emb_dim
        self.attn_dim = int(attn_dim or emb_dim)
        if self.attn_dim <= 0:
            raise ValueError('attn_dim must be positive')

        # ---------- Resolve behaviour groups: prefer the new behavior_groups argument, otherwise fall back to the legacy (bsg, trbg) ----------
        if behavior_groups is None:
            assert bsg_types is not None and trbg_types is not None, \
                "behavior_groups must be provided, or both bsg_types and trbg_types must be provided"
            behavior_groups = [list(bsg_types), list(trbg_types)]
        self.behavior_groups = [list(g) for g in behavior_groups]
        # compatibility fields: legacy code may access self.bsg_types / self.trbg_types
        self.bsg_types = self.behavior_groups[0] if len(self.behavior_groups) >= 1 else []
        self.trbg_types = self.behavior_groups[1] if len(self.behavior_groups) >= 2 else []

        self.key_behavior = key_behavior
        self._set_constrained_parameter('rega_blend', rega_blend)

        # ---------- Ablation switch: whether to enable attention aggregation ----------
        # when use_attn_agg=False, REGA is removed entirely and behaviours are aggregated by directly averaging query_emb ("w/o REGA")
        self.use_attn_agg = bool(use_attn_agg)

        # ---------- Dynamically build the attention linear head for each group (not built when attention is disabled, saving parameters) ----------
        if self.use_attn_agg:
            self.query_proj = nn.ModuleList([
                nn.Linear(emb_dim, self.attn_dim) for _ in self.behavior_groups
            ])
            self.history_proj = nn.ModuleList([
                nn.Linear(emb_dim, self.attn_dim) for _ in self.behavior_groups
            ])
            self.value_proj = nn.ModuleList([
                nn.Linear(self.attn_dim, 1, bias=False) for _ in self.behavior_groups
            ])
        else:
            self.query_proj = nn.ModuleList()
            self.history_proj = nn.ModuleList()
            self.value_proj = nn.ModuleList()
        self.lin = nn.ModuleList()
        self.diagnostics = bool(diagnostics)
        self.last_attention_stats = None

        # ---------- Entropy gate switch and strength: prefer keyword arguments, otherwise fall back to args ----------
        if enable_gate is None:
            enable_gate = getattr(args, 'entropy_guided_mga', 1) == 1 if args else False
        if eg_alpha is None:
            eg_alpha = getattr(args, 'eg_alpha', 0.3) if args else 0.3
        # note: when use_attn_agg=False, entropy_gate is also meaningless and is forcibly disabled
        self.enable_entropy_gate = bool(enable_gate) and self.use_attn_agg
        self._set_constrained_parameter('eg_alpha', eg_alpha)

        if self.enable_entropy_gate:
            self.entropy_gate = RobustEntropyGate(emb_dim)

    def _set_constrained_parameter(self, name, value):
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f'{name} must be in [0, 1]')
        eps = torch.finfo(torch.get_default_dtype()).eps
        value = min(max(value, eps), 1.0 - eps)
        raw = torch.tensor(torch.logit(torch.tensor(value)))
        setattr(self, f'_{name}_raw', nn.Parameter(raw))

    @property
    def eg_alpha(self):
        return torch.sigmoid(self._eg_alpha_raw)

    @property
    def rega_blend(self):
        return torch.sigmoid(self._rega_blend_raw)

    def reset_parameters(self):
        for modules in (self.query_proj, self.history_proj, self.value_proj):
            for linear in modules:
                nn.init.xavier_uniform_(linear.weight)

    def _build_query(self, emb_dict):
        """Build the initial query: prefer the embedding corresponding to key_behavior, otherwise use the mean of all behaviours."""
        if self.key_behavior is not None and self.key_behavior in emb_dict:
            return emb_dict[self.key_behavior]
        all_embs = [emb_dict[b] for g in self.behavior_groups for b in g if b in emb_dict]
        if len(all_embs) == 0:
            raise ValueError("emb_dict is empty; cannot build the REGA query.")
        return torch.stack(all_embs, dim=0).mean(dim=0)

    def forward(self, emb_dict, entropy_dict=None, user_indices=None):
        key = self._build_query(emb_dict)
        attention_stats = []

        updated_emb = key  # fallback: ensures a return even if behavior_groups is empty
        for i, behavior_types in enumerate(self.behavior_groups):
            if len(behavior_types) == 0:
                continue

            query_emb = torch.stack([emb_dict[b] for b in behavior_types], dim=1)

            # ---------- Ablation branch: remove attention entirely and use mean aggregation ----------
            if not self.use_attn_agg:
                updated_emb = query_emb.mean(dim=1)
                key = updated_emb
                continue

            key_expanded = key.unsqueeze(1)
            interaction = torch.tanh(
                self.query_proj[i](key_expanded) + self.history_proj[i](query_emb)
            )
            attention = self.value_proj[i](interaction).softmax(dim=1)
            raw_attention = attention
            ungated_emb = (attention * query_emb).sum(dim=1)

            if self.enable_entropy_gate and entropy_dict is not None:
                gates = []
                batch_size = key.size(0)
                device = key.device

                for b_type in behavior_types:
                    if b_type in entropy_dict and entropy_dict[b_type] is not None:
                        ent = entropy_dict[b_type].detach().to(device)
                        actual_size = ent.size(0)

                        if actual_size == key.size(0):
                            raw_ent = ent.unsqueeze(-1)
                        elif user_indices is not None and key.size(0) == user_indices.size(0):
                            idx = user_indices.to(ent.device).long().clamp(max=actual_size - 1)
                            raw_ent = ent[idx].unsqueeze(-1)
                        elif actual_size == key.size(0) + 1:
                            raw_ent = ent[:key.size(0)].unsqueeze(-1)
                        else:
                            raise ValueError(
                                f"entropy for {b_type} must align with entity axis ({key.size(0)}), "
                                f"got {actual_size}; pass user_indices for a sampled batch."
                            )

                        gate_value = self.entropy_gate(raw_ent)
                        gates.append(
                            torch.clamp(1.0 - self.eg_alpha * gate_value, min=0.0, max=1.0)
                        )
                    else:
                        gates.append(torch.ones(batch_size, 1, device=device))

                gate_tensor = torch.stack(gates, dim=1)
                attention = attention * gate_tensor
                attention = attention / (attention.sum(dim=1, keepdim=True) + 1e-6)

            if self.enable_entropy_gate and entropy_dict is not None:
                gate_stats = gate_tensor
            else:
                gate_stats = torch.ones_like(attention)

            gated_emb = (attention * query_emb).sum(dim=1)
            if self.enable_entropy_gate and entropy_dict is not None:
                updated_emb = (1.0 - self.rega_blend) * ungated_emb + self.rega_blend * gated_emb
            else:
                updated_emb = ungated_emb
            if self.diagnostics:
                attention_stats.append({
                    'group': i,
                    'raw_min': float(raw_attention.detach().min()),
                    'raw_max': float(raw_attention.detach().max()),
                    'raw_sum_error': float((raw_attention.detach().sum(dim=1) - 1.0).abs().max()),
                    'gate_min': float(gate_stats.detach().min()),
                    'gate_max': float(gate_stats.detach().max()),
                    'gated_min': float(attention.detach().min()),
                    'gated_max': float(attention.detach().max()),
                    'gated_sum_error': float((attention.detach().sum(dim=1) - 1.0).abs().max()),
                    'attention_delta': float((attention.detach() - raw_attention.detach()).abs().mean()),
                })
            key = updated_emb

        self.last_attention_stats = attention_stats or None
        return updated_emb
