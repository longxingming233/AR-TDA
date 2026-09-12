"""BPR trainer, works with any BaseRecModel subclass.

evaluate / train_epoch are fully aligned with EaMRec.Trainer:
  - train_gt / test_gt are always queried by int key;
  - during evaluation, training interactions are set to -inf and model.predict(users) is called per batch;
  - hit / ndcg use the EaMRec-equivalent implementation from utils.metrics;
  - during training, NaN/Inf losses are skipped automatically and NaN gradients are zeroed, matching EaMRec behaviour.
"""
import copy
import numpy as np
import os
import time
import torch
from tqdm import tqdm
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from .metrics import hit, ndcg


class Trainer:
    def __init__(self, model, data, args):
        self.model = model
        self.data = data
        self.args = args
        self.topk = args.topk
        self.device = args.device
        self.protocol = getattr(args, 'experiment_protocol', 'unified')
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        self.best_hr = -float('inf')
        self.best_ndcg = 0.0
        self.best_epoch = 0
        self.no_improve = 0
        self.start_epoch = 1
        self.best_model_state = None
        self._p0_diagnostics_done = False
        run_name = (
            f'{args.dataset}_{args.model}_tda{int(args.use_tda)}_rega{int(args.use_rega)}_'
            f'attn{int(args.use_attn_agg)}_blend{getattr(args, "rega_blend", 0.1):g}_'
            f'{args.tda_mode}_seed{args.seed}'
        )
        self.checkpoint_path = os.path.join(args.checkpoint_dir, f'{run_name}_best.pt')
        if args.resume:
            self._load_checkpoint(args.resume)

    def _checkpoint_args(self):
        return {key: value for key, value in vars(self.args).items()}

    def _validate_checkpoint_args(self, checkpoint_args, path):
        keys = (
            'dataset', 'model', 'seed', 'experiment_protocol', 'split_seed', 'validation_ratio', 'topk',
            'emb_dim', 'gnn_layers', 'tda_layers', 'use_tda', 'use_rega',
            'use_attn_agg', 'tda_mode', 'entropy_norm', 'keep_min_edges',
            'min_keep_ratio', 'fast_entropy', 'rega_attn_dim', 'rega_blend', 'denoise_layers',
            'denoise_alpha', 'lr', 'weight_decay', 'batch_size', 'eval_batch_size',
            'eval_item_chunk_size', 'num_epochs', 'patience',
        )
        mismatches = []
        for key in keys:
            if key in checkpoint_args and checkpoint_args[key] != getattr(self.args, key, None):
                mismatches.append(f'{key}: checkpoint={checkpoint_args[key]!r}, current={getattr(self.args, key, None)!r}')
        if mismatches:
            raise ValueError(f'Checkpoint configuration mismatch: {path}; ' + '; '.join(mismatches))

    def _save_checkpoint(self, epoch):
        os.makedirs(self.args.checkpoint_dir, exist_ok=True)
        temporary_path = self.checkpoint_path + '.tmp'
        torch.save({
            'args': self._checkpoint_args(),
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_hr': self.best_hr,
            'best_ndcg': self.best_ndcg,
            'best_epoch': self.best_epoch,
            'no_improve': self.no_improve,
        }, temporary_path)
        os.replace(temporary_path, self.checkpoint_path)

    def _load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        checkpoint_args = checkpoint.get('args')
        if checkpoint_args is None:
            raise ValueError(f'Checkpoint does not contain complete args: {path}')
        self._validate_checkpoint_args(checkpoint_args, path)
        try:
            self.model.load_state_dict(checkpoint['model'], strict = False)
        except RuntimeError as exc:
            raise ValueError(f'Checkpoint is incompatible with the current model configuration: {path}') from exc
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.best_hr = float(checkpoint.get('best_hr', -float('inf')))
        self.best_ndcg = float(checkpoint.get('best_ndcg', 0.0))
        self.best_epoch = int(checkpoint.get('best_epoch', 0))
        self.no_improve = int(checkpoint.get('no_improve', 0))
        self.best_model_state = copy.deepcopy(self.model.state_dict())
        self.start_epoch = int(checkpoint.get('epoch', 0)) + 1
        logger.info(f'Resumed checkpoint: {path}')

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train_epoch(self, epoch: int):
        self.model.train()
        # notify the model of the current epoch (used by models such as LightGCN for warmup control; ignored by models without this method)
        if hasattr(self.model, 'set_epoch'):
            self.model.set_epoch(epoch)
        total_loss = 0.0
        loader = self.data['train_loader']
        pbar = tqdm(loader, desc=f'Epoch {epoch} Train', leave=False)
        for batch in pbar:
            u, p, n = [t.to(self.device, non_blocking=True) for t in batch]
            loss = self.model.loss(u, p, n)
            if isinstance(loss, dict):
                loss = loss['main_loss']

            if self.protocol != 'hgib' and (torch.isnan(loss) or torch.isinf(loss)):
                self.optimizer.zero_grad()
                continue

            before_step = None
            if getattr(self.args, 'p0_diagnostics', False) and not self._p0_diagnostics_done:
                tracked = ('tda_min_param', 'tda_span', 'entropy_gate', '_eg_alpha_raw', '_rega_blend_raw')
                before_step = {
                    name: param.detach().clone()
                    for name, param in self.model.named_parameters()
                    if param.requires_grad and any(token in name for token in tracked)
                }
            self.optimizer.zero_grad()
            loss.backward()
            raw_gradient_stats = self._p0_gradient_stats(before_step) if before_step is not None else None
            if self.protocol != 'hgib':
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

                for _, param in self.model.named_parameters():
                    if param.grad is not None and (
                            torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        param.grad.zero_()

            self.optimizer.step()
            if before_step is not None:
                self._log_p0_diagnostics(before_step, raw_gradient_stats)
                self._disable_p0_diagnostics()
                self._p0_diagnostics_done = True
            total_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4f}')
        return total_loss / max(len(loader), 1)

    def _disable_p0_diagnostics(self):
        for module in self.model.modules():
            if hasattr(module, 'diagnostics'):
                module.diagnostics = False
            if hasattr(module, 'last_attention_stats'):
                module.last_attention_stats = None

    def _p0_gradient_stats(self, before_step):
        stats = {}
        for name, param in self.model.named_parameters():
            if name not in before_step:
                continue
            if param.grad is None:
                stats[name] = ('missing', float('nan'))
            elif not torch.isfinite(param.grad).all():
                stats[name] = ('nonfinite', float(param.grad.detach().norm()))
            else:
                grad_norm = float(param.grad.detach().norm())
                stats[name] = ('nonzero' if grad_norm > 0.0 else 'zero', grad_norm)
        return stats

    def _log_p0_diagnostics(self, before_step, raw_gradient_stats):
        tracked = ('tda_min_param', 'tda_span', 'entropy_gate', '_eg_alpha_raw', '_rega_blend_raw')
        for name, param in self.model.named_parameters():
            if not any(token in name for token in tracked):
                continue
            grad_state, grad_norm = raw_gradient_stats[name]
            update_norm = float((param.detach() - before_step[name]).norm())
            logger.info(
                f'P0 PARAM name={name} grad_state={grad_state} grad_norm={grad_norm:.12g} '
                f'update_norm={update_norm:.12g}'
            )
        for name, module in self.model.named_modules():
            stats = getattr(module, 'last_attention_stats', None)
            if not stats:
                continue
            for row in stats:
                logger.info(
                    f'P0 ATTENTION module={name} group={row["group"]} '
                    f'raw_min={row["raw_min"]:.12g} raw_max={row["raw_max"]:.12g} '
                    f'raw_sum_error={row["raw_sum_error"]:.12g} '
                    f'gate_min={row["gate_min"]:.12g} gate_max={row["gate_max"]:.12g} '
                    f'gated_min={row["gated_min"]:.12g} gated_max={row["gated_max"]:.12g} '
                    f'gated_sum_error={row["gated_sum_error"]:.12g} '
                    f'attention_delta={row["attention_delta"]:.12g}'
                )

    # ------------------------------------------------------------------
    # Evaluation: fully equivalent to EaMRec.Trainer.evaluate
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, epoch: int = -1, split: str = 'test'):
        self.model.eval()
        K = self.topk
        topk_list = []
        pos_len_list = []
        loader = self.data[f'{split}_loader']
        ground_truth = self.data[f'{split}_gt']
        pbar = tqdm(loader, desc=f'Epoch {epoch} Eval', leave=False)

        shared_cache = None
        if not (self.protocol == 'hgib' and self.args.model == 'hgib'):
            shared_cache = self.model.prepare_eval_cache()
        chunk_size = self.args.eval_item_chunk_size
        item_start = 0 if self.protocol == 'hgib' else 1
        all_items = torch.arange(item_start, self.data['n_items'] + 1, device=self.device)

        for batch in pbar:
            users = batch.to(self.device, non_blocking=True)
            cache = shared_cache if shared_cache is not None else self.model.prepare_eval_cache()
            best_scores = torch.empty((users.size(0), 0), device=self.device)
            best_items = torch.empty((users.size(0), 0), dtype=torch.long, device=self.device)
            for start in range(0, len(all_items), chunk_size):
                item_ids = all_items[start:start + chunk_size]
                scores = self.model.score_items(users, item_ids, cache)
                for i, user in enumerate(users.tolist()):
                    blocked_items = list(self.data['train_gt'].get(user, []))
                    if self.protocol != 'hgib' and split == 'test':
                        blocked_items.extend(self.data.get('valid_gt', {}).get(user, []))
                    if blocked_items:
                        blocked = torch.isin(item_ids, torch.as_tensor(blocked_items, device=self.device))
                        scores[i, blocked] = -float('inf')
                candidate_scores = torch.cat([best_scores, scores], dim=1)
                candidate_items = torch.cat([
                    best_items,
                    item_ids.unsqueeze(0).expand(users.size(0), -1),
                ], dim=1)
                local_k = min(K, candidate_scores.size(1))
                best_scores, positions = torch.topk(candidate_scores, local_k, dim=1)
                best_items = torch.gather(candidate_items, 1, positions)
            idx = best_items
            idx_cpu = idx.cpu().numpy()

            # 3) hit matrix + the number of true positives per row (used for IDCG truncation in NDCG)
            for i in range(users.size(0)):
                u = users[i].item()
                gt = np.array(ground_truth[u])
                topk_list.append(np.isin(idx_cpu[i], gt))
                pos_len_list.append(len(gt))

        if not topk_list:
            raise ValueError(f'No {split} users are available for evaluation')
        topk_arr = np.vstack(topk_list)
        pos_len_arr = np.asarray(pos_len_list)

        # pos_len is unused in hit() -> fully consistent with EaMRec
        # pos_len is used for IDCG truncation in ndcg() -> fully consistent with EaMRec
        hr_arr = hit(topk_arr, pos_len_arr).mean(axis=0)
        ndcg_arr = ndcg(topk_arr, pos_len_arr).mean(axis=0)
        return float(hr_arr[K - 1]), float(ndcg_arr[K - 1])

    def _tda_statistics(self):
        rows = []
        for name, module in self.model.named_modules():
            total = getattr(module, 'last_total_edges', 0)
            if total:
                kept = getattr(module, 'last_kept_edges', total)
                rows.append({
                    'module': name,
                    'kept': kept,
                    'total': total,
                    'ratio': kept / total,
                    'threshold_min': getattr(module, 'last_threshold_min', None),
                    'threshold_max': getattr(module, 'last_threshold_max', None),
                })
        return rows

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def train(self):
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        run_started = time.perf_counter()
        epoch_times = []
        evaluation_seconds = 0.0
        for epoch in range(self.start_epoch, self.args.num_epochs + 1):
            epoch_started = time.perf_counter()
            loss = self.train_epoch(epoch)
            eval_split = 'test' if self.protocol == 'hgib' else 'valid'
            evaluation_started = time.perf_counter()
            hr, ndcg_v = self.evaluate(epoch, split=eval_split)
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            evaluation_seconds += time.perf_counter() - evaluation_started
            epoch_times.append(time.perf_counter() - epoch_started)

            improved = ndcg_v > self.best_ndcg if self.protocol == 'hgib' else hr > self.best_hr
            if improved:
                self.best_hr = hr
                self.best_ndcg = ndcg_v
                self.best_epoch = epoch
                self.no_improve = 0
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                if self.args.save_checkpoint:
                    self._save_checkpoint(epoch)
            else:
                self.no_improve += 1

            split_label = 'TEST' if self.protocol == 'hgib' else 'VAL'
            best_label = 'NDCG' if self.protocol == 'hgib' else 'HR'
            best_value = self.best_ndcg if self.protocol == 'hgib' else self.best_hr
            logger.info(
                f"Epoch {epoch:03d} | loss={loss:.4f} | "
                f"{split_label} HR@{self.topk}={hr:.4f} NDCG@{self.topk}={ndcg_v:.4f} | "
                f"best {split_label} {best_label}={best_value:.4f} @epoch{self.best_epoch}"
            )

            if self.no_improve >= self.args.patience:
                logger.info(f"Early stop at epoch {epoch}")
                break

        test_started = time.perf_counter()
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        test_hr, test_ndcg = self.evaluate(self.best_epoch, split='test')
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        test_seconds = time.perf_counter() - test_started
        total_seconds = time.perf_counter() - run_started
        peak_memory_mb = (
            torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            if self.device.type == 'cuda' else 0.0
        )
        tda_stats = self._tda_statistics()
        kept_edges = sum(row['kept'] for row in tda_stats)
        total_edges = sum(row['total'] for row in tda_stats)
        retention_ratio = kept_edges / total_edges if total_edges else 1.0
        threshold_mins = [row['threshold_min'] for row in tda_stats if row['threshold_min'] is not None]
        threshold_maxs = [row['threshold_max'] for row in tda_stats if row['threshold_max'] is not None]
        selected_by = 'TEST_NDCG' if self.protocol == 'hgib' else 'VAL_HR'
        logger.info(
            f"TEST selected_by={selected_by} epoch={self.best_epoch} | "
            f"HR@{self.topk}={test_hr:.4f} NDCG@{self.topk}={test_ndcg:.4f}"
        )
        logger.info(
            f"RESOURCE total_seconds={total_seconds:.3f} "
            f"mean_epoch_seconds={np.mean(epoch_times) if epoch_times else 0.0:.3f} "
            f"evaluation_seconds={evaluation_seconds:.3f} "
            f"test_seconds={test_seconds:.3f} peak_memory_mb={peak_memory_mb:.3f}"
        )
        logger.info(
            f"DENOISE retained_edges={kept_edges} total_edges={total_edges} "
            f"retention_ratio={retention_ratio:.6f} "
            f"threshold_min={np.mean(threshold_mins) if threshold_mins else float('nan'):.6f} "
            f"threshold_max={np.mean(threshold_maxs) if threshold_maxs else float('nan'):.6f}"
        )
        return test_hr, test_ndcg
