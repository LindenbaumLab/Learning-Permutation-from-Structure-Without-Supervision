import json
import logging
import math
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from datetime import datetime

from model import FCModel, ThresholdPolicy, TransformerEncoderModel
from number_utils import NumberGenerator
import os, sys
sys.path.append(os.pardir)
from utils import gumbel_sinkhorn_ops


def setup_logging(log_path: str) -> None:
    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = []
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_rank_mask(X, rank_temp, rank_sigma):
    """
    Build an item->position mask using differentiable soft ranks.
    Returns mask tensor and stats (mean/std of rank, effective support).
    """
    if X.dim() == 3:
        X = X.squeeze(1)
    # D[b, i, k] = (X[b, k] - X[b, i]) / rank_temp
    D = (X[:, None, :] - X[:, :, None]) / rank_temp
    soft_rank = torch.sigmoid(D).sum(dim=-1) - 0.5  # approximate count of larger elements

    positions = torch.arange(X.size(1), device=X.device, dtype=X.dtype)
    diff = positions[None, None, :] - soft_rank[:, :, None]
    mask = torch.exp(-(diff ** 2) / (2 * (rank_sigma ** 2) + 1e-9))
    mask = mask / (mask.sum(dim=-1, keepdim=True) + 1e-9)

    support = (mask > 0.01).float().sum(dim=-1).float()
    stats = {
        "rank_mean": soft_rank.mean(),
        "rank_std": soft_rank.std(),
        "support_mean": support.mean(),
    }
    return mask, stats, soft_rank


def build_true_mask(permutation, extra_k=0):
    """
    Build a mask that exposes the true item->position mapping.
    Optionally unmask 'extra_k' additional random columns per row.
    """
    inv_perm = permutation.argsort(dim=1)  # (B, n), rank of each original index
    batch, n = permutation.shape
    mask = permutation.new_zeros((batch, n, n), dtype=torch.float)
    mask.scatter_(2, inv_perm.unsqueeze(-1), 1.0)  # one-hot at true column

    if extra_k > 0:
        k = min(extra_k, max(n - 1, 0))
        if k >= n - 1:
            mask.fill_(1.0)  # full support; no RNG consumption
        elif k > 0:
            # Sample k extra columns per row without picking the true column.
            scores = torch.rand(batch, n, n, device=mask.device)
            scores.scatter_(2, inv_perm.unsqueeze(-1), -1.0)
            extra_indices = scores.topk(k=k, dim=2).indices
            mask.scatter_(2, extra_indices, 1.0)


    mask = mask / (mask.sum(dim=-1, keepdim=True) + 1e-9)
    support = (mask > 0.01).float().sum(dim=-1).float()
    stats = {
        "rank_mean": inv_perm.float().mean(),
        "rank_std": inv_perm.float().std(),
        "support_mean": support.mean(),
    }
    return mask, stats, inv_perm.float()


def build_forbid_mask(permutation, forbid_k=0):
    """
    Build a mask that only forbids some incorrect columns per row.
    Allowed columns stay uniform; true column is kept but not singled out.
    """
    inv_perm = permutation.argsort(dim=1)  # (B, n)
    batch, n = permutation.shape
    mask = permutation.new_ones((batch, n, n), dtype=torch.float)

    if forbid_k > 0:
        k = min(forbid_k, max(n - 1, 0))
        if k > 0:
            # Sample k wrong columns to forbid (never forbid the true column).
            scores = torch.rand(batch, n, n, device=mask.device)
            scores.scatter_(2, inv_perm.unsqueeze(-1), -1.0)
            forbid_indices = scores.topk(k=k, dim=2).indices
            mask.scatter_(2, forbid_indices, 0.0)

    mask = mask / (mask.sum(dim=-1, keepdim=True) + 1e-9)
    support = (mask > 0.01).float().sum(dim=-1).float()
    stats = {
        "rank_mean": inv_perm.float().mean(),
        "rank_std": inv_perm.float().std(),
        "support_mean": support.mean(),
    }
    return mask, stats, inv_perm.float()


def build_attention_bias_from_mask(mask, strength, mask_eps=1e-9):
    sim = torch.bmm(mask, mask.transpose(1, 2))
    sim = sim / (sim.sum(dim=-1, keepdim=True) + mask_eps)
    return strength * torch.log(sim.clamp(min=mask_eps))


def build_policy_features(X, eps=1e-6):
    if X.dim() == 3:
        X = X.squeeze(1)
    x_min = X.min(dim=1, keepdim=True).values
    x_max = X.max(dim=1, keepdim=True).values
    denom = (x_max - x_min).clamp(min=eps)
    x_norm = (X - x_min) / denom
    x_mean = X.mean(dim=1, keepdim=True)
    x_std = X.std(dim=1, keepdim=True, unbiased=False)
    features = torch.cat([x_mean, x_std, x_min, x_max, x_norm], dim=1)
    return features, x_norm


def build_rl_threshold_mask(x_norm, thresholds, rl_mask_frac):
    if thresholds.dim() > 1:
        thresholds = thresholds.squeeze(-1)
    batch, n = x_norm.shape
    mask = x_norm.new_ones((batch, n, n))
    half = n // 2
    k = int(math.ceil(rl_mask_frac * (n / 2)))
    k = min(k, half)

    if k > 0:
        col_idx = torch.arange(n, device=x_norm.device)
        early_cols = col_idx < half
        late_cols = col_idx >= half
        side_low = x_norm < thresholds[:, None]
        opp_cols = torch.where(side_low[:, :, None], late_cols[None, None, :], early_cols[None, None, :])
        scores = torch.rand(batch, n, n, device=x_norm.device)
        scores = scores.masked_fill(~opp_cols, -1.0)
        forbid_indices = scores.topk(k=k, dim=2).indices
        mask.scatter_(2, forbid_indices, 0.0)

    mask = mask / (mask.sum(dim=-1, keepdim=True) + 1e-9)
    support = (mask > 0.01).float().sum(dim=-1).float()
    stats = {
        "rank_mean": thresholds.mean(),
        "rank_std": thresholds.std(unbiased=False),
        "support_mean": support.mean(),
    }
    return mask, stats


def kendall_tau(pred_perm, true_perm):
    """
    Compute Kendall tau between two permutations for each sample in the batch.
    pred_perm/true_perm: Long tensors of shape (batch, n_items) containing indices.
    """
    pred_perm = pred_perm.long()
    true_perm = true_perm.long()
    inv_true = torch.argsort(true_perm, dim=1)
    inv_pred = torch.argsort(pred_perm, dim=1)
    n = inv_true.size(1)
    diff_true = inv_true.unsqueeze(2) - inv_true.unsqueeze(1)
    diff_pred = inv_pred.unsqueeze(2) - inv_pred.unsqueeze(1)
    sign_prod = diff_true * diff_pred
    mask = torch.triu(torch.ones(n, n, device=pred_perm.device, dtype=torch.bool), diagonal=1)
    concordant = (sign_prod[:, mask] > 0).sum(1).float()
    discordant = (sign_prod[:, mask] < 0).sum(1).float()
    tau = (concordant - discordant) / (0.5 * n * (n - 1))
    return tau


def build_model(cfg):
    model_type = getattr(cfg, "model_type", "conv")
    if model_type in ("conv", "fc"):
        return FCModel(cfg.hid_c, cfg.n_numbers)
    if model_type == "transformer":
        n_layers = getattr(cfg, "transformer_n_layers", 8)
        n_heads = getattr(cfg, "transformer_n_heads", 4)
        ffn_dim = getattr(cfg, "transformer_ffn_dim", cfg.hid_c * 4)
        dropout = getattr(cfg, "transformer_dropout", 0.0)
        return TransformerEncoderModel(cfg.hid_c, cfg.n_numbers, n_layers, n_heads, ffn_dim, dropout)
    raise ValueError(f"Unknown model_type: {model_type}")

def annealed_tau(cfg, epoch: int) -> float:
    tau_end = getattr(cfg, "tau_anneal_end", None)
    if tau_end is None:
        return float(cfg.tau)
    tau_start = float(getattr(cfg, "tau_anneal_start", cfg.tau))
    tau_anneal_epochs = int(getattr(cfg, "tau_anneal_epochs", 0)) or cfg.epochs
    tau_anneal_epochs = max(1, tau_anneal_epochs)
    if tau_anneal_epochs <= 1:
        return float(tau_end)
    schedule = getattr(cfg, "tau_anneal_schedule", "linear")
    t = min(epoch, tau_anneal_epochs - 1) / float(tau_anneal_epochs - 1)
    if schedule == "linear":
        return tau_start + (tau_end - tau_start) * t
    if schedule == "exp":
        if tau_start <= 0 or tau_end <= 0:
            return float(tau_end)
        return tau_start * ((tau_end / tau_start) ** t)
    raise ValueError(f"Unknown tau_anneal_schedule: {schedule}")


def normalized_entropy(probs: torch.Tensor, dim: int, norm_size: int, eps: float = 1e-9) -> torch.Tensor:
    norm_log = math.log(max(int(norm_size), 2))
    log_probs = probs.clamp(min=eps).log()
    return -(probs * log_probs).sum(dim=dim) / norm_log


def adaptive_tau_matrix(log_alpha: torch.Tensor, base_tau: float, cfg) -> torch.Tensor:
    tau_source = getattr(cfg, "tau_adapt_source", "log_alpha")
    combine = getattr(cfg, "tau_adapt_combine", "avg")
    entropy_thresh = float(getattr(cfg, "tau_adapt_entropy_thresh", 0.75))
    boost_max = max(0.0, float(getattr(cfg, "tau_adapt_boost_max", 0.1)))

    entropy_thresh = min(max(entropy_thresh, 0.0), 0.999)
    denom = max(1.0 - entropy_thresh, 1e-6)

    scaled_logits = log_alpha.detach() / base_tau
    if tau_source == "sinkhorn":
        probs = gumbel_sinkhorn_ops.log_sinkhorn_norm(scaled_logits, cfg.n_sink_iter)
        row_probs = probs
        col_probs = probs
    elif tau_source == "log_alpha":
        row_probs = torch.softmax(scaled_logits, dim=-1)
        col_probs = torch.softmax(scaled_logits, dim=-2)
    else:
        raise ValueError(f"Unknown tau_adapt_source: {tau_source}")

    n_rows = log_alpha.size(-2)
    n_cols = log_alpha.size(-1)
    entropy_row = normalized_entropy(row_probs, dim=-1, norm_size=n_cols)
    entropy_col = normalized_entropy(col_probs, dim=-2, norm_size=n_rows)

    boost_row = boost_max * torch.clamp((entropy_row - entropy_thresh) / denom, min=0.0, max=1.0)
    boost_col = boost_max * torch.clamp((entropy_col - entropy_thresh) / denom, min=0.0, max=1.0)

    tau_row = base_tau * (1.0 + boost_row)
    tau_col = base_tau * (1.0 + boost_col)
    if combine == "prod":
        tau_mat = tau_row.unsqueeze(-1) * tau_col.unsqueeze(-2) / base_tau
    elif combine == "avg":
        tau_mat = 0.5 * (tau_row.unsqueeze(-1) + tau_col.unsqueeze(-2))
    else:
        raise ValueError(f"Unknown tau_adapt_combine: {combine}")

    tau_min = base_tau
    tau_max = base_tau * (1.0 + boost_max)
    return tau_mat.clamp(min=tau_min, max=tau_max)

def train(cfg):
    logger = logging.getLogger("NumberSorting")
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
    else:
        device = "cpu"
    pin_memory = device == "cuda"
    save_model = bool(getattr(cfg, "save_model", False))
    save_log_alpha = bool(getattr(cfg, "save_log_alpha", False))
    seed = getattr(cfg, "seed", None)
    if seed is not None:
        set_seed(int(seed))
    model_type = getattr(cfg, "model_type", "conv")
    model = build_model(cfg).to(device)
    optimizer = optim.Adam(model.parameters(), cfg.lr)
    mask_mode = getattr(cfg, "mask_mode", "estimated")
    use_rl_policy = mask_mode == "rl_threshold"
    policy = None
    policy_optimizer = None
    rl_mask_frac = None
    rl_entropy_coef = None
    mask_eps = getattr(cfg, "mask_eps", 1e-9)
    spread_weight = float(getattr(cfg, "spread_weight", 0.0))
    entropy_weight = float(getattr(cfg, "entropy_weight", 0.0))
    entropy_eps = float(getattr(cfg, "entropy_eps", 1e-9))
    tau_adapt = bool(getattr(cfg, "tau_adapt", False))
    tau_adapt_start_epoch = max(1, int(getattr(cfg, "tau_adapt_start_epoch", 1)))
    early_stop_patience = int(getattr(cfg, "early_stop_patience", 0))
    early_stop_min_delta = float(getattr(cfg, "early_stop_min_delta", 0.0))
    eval_every = int(getattr(cfg, "eval_every", 0))
    best_kendall_tau = None
    best_epoch = None
    no_improve_epochs = 0
    best_model_path = getattr(cfg, "best_model_path", None)
    if best_model_path is None:
        model_path = getattr(cfg, "model_path", None)
        if model_path:
            best_model_path = os.path.join(os.path.dirname(model_path), "model_best.pth")
        else:
            best_model_path = os.path.join(cfg.run_dir, "model_best.pth")
        cfg.best_model_path = best_model_path
    if early_stop_patience > 0 and eval_every <= 0:
        logger.warning("early_stop_patience set but eval_every <= 0; early stopping disabled.")
        early_stop_patience = 0
    if use_rl_policy:
        policy_hid_c = getattr(cfg, "rl_policy_hid_c", cfg.hid_c)
        policy = ThresholdPolicy(cfg.n_numbers, policy_hid_c).to(device)
        policy_lr = getattr(cfg, "rl_policy_lr", cfg.lr)
        policy_optimizer = optim.Adam(policy.parameters(), policy_lr)
        rl_mask_frac = float(getattr(cfg, "rl_mask_frac", 0.2))
        rl_mask_frac = max(0.0, min(1.0, rl_mask_frac))
        rl_entropy_coef = float(getattr(cfg, "rl_entropy_coef", 0.01))
        policy_path = getattr(cfg, "policy_path", None)
        if policy_path is None:
            model_path = getattr(cfg, "model_path", None)
            if model_path:
                policy_path = os.path.join(os.path.dirname(model_path), "policy_weight.pth")
            else:
                policy_path = os.path.join(cfg.run_dir, "policy_weight.pth")
            cfg.policy_path = policy_path
        best_policy_path = getattr(cfg, "best_policy_path", None)
        if best_policy_path is None:
            best_policy_path = os.path.join(os.path.dirname(cfg.policy_path), "policy_best.pth")
            cfg.best_policy_path = best_policy_path

    dataset = NumberGenerator(cfg.n_numbers, cfg.n_train_lists, cfg.min_value, cfg.max_value, cfg.train_seed)
    train_loader = DataLoader(
        dataset,
        cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=pin_memory,
    )

    log_alpha_std_path = None
    if save_log_alpha and getattr(cfg, "run_dir", None):
        log_alpha_std_path = os.path.join(cfg.run_dir, "log_alpha_std.csv")
        if not os.path.exists(log_alpha_std_path):
            with open(log_alpha_std_path, "w") as f:
                f.write("epoch,mean_log_alpha_std\n")

    logger.info("training")
    eval_history = []
    early_stop = False
    for epoch in range(cfg.epochs):
        current_tau = annealed_tau(cfg, epoch)
        log_alpha_std_vals = (
            [] if cfg.eval_every > 0 and ((epoch + 1) % cfg.eval_every) == 0 else None
        )
        for i, data in enumerate(train_loader):
            X, ordered_X, permutation = data
            X = X.to(device, non_blocking=True)
            ordered_X = ordered_X.to(device, non_blocking=True)
            permutation = permutation.to(device, non_blocking=True)

            mask_stats = None
            soft_rank = None
            log_prob = None
            entropy = None
            mask = None
            attn_bias = None

            # Mask selection: estimated (default), true, true+k, or none.
            mask_mode = getattr(cfg, "mask_mode", "estimated")
            use_mask_flag = getattr(cfg, "use_rank_mask", False) or getattr(cfg, "use_masked_sinkhorn", False)
            use_mask = (mask_mode != "none") and (mask_mode != "estimated" or use_mask_flag)

            if use_mask:
                rank_mask_strength = getattr(cfg, "rank_mask_strength", 1.0)
                if mask_mode == "estimated":
                    rank_temp = getattr(cfg, "rank_temp", 0.1)
                    rank_sigma = getattr(cfg, "rank_sigma", 2.0)
                    mask, mask_stats, soft_rank = build_rank_mask(
                        X,
                        rank_temp=rank_temp,
                        rank_sigma=rank_sigma,
                    )
                elif mask_mode == "rl_threshold":
                    features, x_norm = build_policy_features(X)
                    mean, logstd = policy(features)
                    dist = torch.distributions.Normal(mean, logstd.exp())
                    thresholds = dist.sample()
                    log_prob = dist.log_prob(thresholds).squeeze(-1)
                    entropy = dist.entropy().squeeze(-1)
                    mask, mask_stats = build_rl_threshold_mask(x_norm, thresholds, rl_mask_frac)
                    soft_rank = thresholds.squeeze(-1)
                elif mask_mode in ("true", "true+k"):
                    extra_k = getattr(cfg, "true_mask_extra_k", 0) if mask_mode == "true+k" else 0
                    mask, mask_stats, soft_rank = build_true_mask(permutation, extra_k=extra_k)
                else:
                    forbid_k = getattr(cfg, "true_mask_forbid_k", 0)
                    mask, mask_stats, soft_rank = build_forbid_mask(permutation, forbid_k=forbid_k)

                if model_type == "transformer" and mask_mode != "forbid":
                    attn_bias = build_attention_bias_from_mask(mask, rank_mask_strength, mask_eps)

            log_alpha = model(X[:, None], attn_bias=attn_bias)
            if log_alpha_std_vals is not None:
                log_alpha_std_vals.append(log_alpha.detach().std().item())
            log_alpha_masked = log_alpha

            if use_mask:
                if mask_mode == "rl_threshold":
                    log_alpha_masked = log_alpha + rank_mask_strength * torch.log(mask.clamp(min=mask_eps))
                else:
                    log_alpha_masked = log_alpha + rank_mask_strength * torch.log(mask + 1e-9)

            tau_batch = current_tau
            if tau_adapt and (epoch + 1) >= tau_adapt_start_epoch:
                tau_batch = adaptive_tau_matrix(log_alpha_masked, current_tau, cfg)

            sup_losses = []
            smooth_losses = []
            smooth_per_example_losses = [] if use_rl_policy else None
            spread_losses = [] if spread_weight > 0 else None
            entropy_losses = [] if entropy_weight > 0 else None
            for _ in range(cfg.n_samples):
                gs_mat = gumbel_sinkhorn_ops.gumbel_sinkhorn(log_alpha_masked, tau_batch, cfg.n_sink_iter)
                est_ordered_X = gumbel_sinkhorn_ops.inverse_permutation(X, gs_mat)

                if cfg.mse_weight > 0:
                    sup_losses.append(torch.nn.functional.mse_loss(est_ordered_X, ordered_X))

                if cfg.smooth_loss_weight > 0 or use_rl_policy:
                    diff = est_ordered_X[:, 1:] - est_ordered_X[:, :-1]
                    if cfg.smooth_direction == "asc":
                        violation = torch.relu(-diff)
                    else:
                        violation = torch.relu(diff)
                    violation_sq = violation ** 2
                    if cfg.smooth_loss_weight > 0:
                        smooth_losses.append(violation_sq.mean())
                    if use_rl_policy:
                        smooth_per_example_losses.append(violation_sq.mean(dim=1))

                if spread_weight > 0:
                    est_std = est_ordered_X.std(dim=1, unbiased=False)
                    spread_losses.append(-est_std.mean())
                if entropy_weight > 0:
                    ent = (gs_mat * torch.log(gs_mat + entropy_eps)).sum(dim=(1, 2)).mean()
                    entropy_losses.append(ent)

            sup_loss = torch.stack(sup_losses).mean() if sup_losses else torch.tensor(0.0, device=device)
            smooth_loss = torch.stack(smooth_losses).mean() if smooth_losses else torch.tensor(0.0, device=device)
            spread_loss = (
                torch.stack(spread_losses).mean()
                if spread_losses is not None
                else torch.tensor(0.0, device=device)
            )
            entropy_loss = (
                torch.stack(entropy_losses).mean()
                if entropy_losses is not None
                else torch.tensor(0.0, device=device)
            )
            loss = (
                cfg.mse_weight * sup_loss
                + cfg.smooth_loss_weight * smooth_loss
                + spread_weight * spread_loss
                + entropy_weight * entropy_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if use_rl_policy:
                smooth_per_example = torch.stack(smooth_per_example_losses).mean(dim=0)
                reward = -smooth_per_example.detach()
                baseline = reward.mean()
                advantage = reward - baseline
                policy_loss = -(advantage * log_prob).mean() - rl_entropy_coef * entropy.mean()
                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()

            if getattr(cfg, "display", 0) > 0 and ((i + 1) % cfg.display) == 0:
                mask_log = ""
                if mask_stats is not None and soft_rank is not None:
                    mask_log = (
                        " | rank_mean %.4f | rank_std %.4f | mask_support_mean %.4f"
                        % (
                            mask_stats["rank_mean"].item(),
                            mask_stats["rank_std"].item(),
                            mask_stats["support_mean"].item(),
                        )
                    )
                extra_log = ""
                if spread_weight > 0:
                    extra_log += " | spread %f" % spread_loss.item()
                if entropy_weight > 0:
                    extra_log += " | ent %f" % entropy_loss.item()
                logger.info(
                    "%i epoch [%i/%i] training loss %f | sup %f | smooth %f%s%s",
                    epoch, i+1, len(train_loader), loss.item(), sup_loss.item(), smooth_loss.item(), mask_log, extra_log
                )

        if cfg.eval_every > 0 and ((epoch + 1) % cfg.eval_every) == 0:
            if use_rl_policy:
                metrics = evaluation(cfg, model=model, policy=policy, step=epoch + 1)
            else:
                metrics = evaluation(cfg, model=model, step=epoch + 1)
            eval_history.append(
                {
                    "epoch": epoch + 1,
                    "mean_prop_wrong": float(metrics["mean_prop_wrong"]),
                    "mean_prop_any_wrong": float(metrics["mean_prop_any_wrong"]),
                    "mean_kendall_tau": float(metrics["mean_kendall_tau"]),
                }
            )
            current_kendall_tau = metrics["mean_kendall_tau"].item()
            if best_kendall_tau is None or current_kendall_tau > best_kendall_tau + early_stop_min_delta:
                best_kendall_tau = current_kendall_tau
                best_epoch = epoch + 1
                no_improve_epochs = 0
                if save_model:
                    torch.save(model.state_dict(), best_model_path)
                    if use_rl_policy:
                        torch.save(policy.state_dict(), cfg.best_policy_path)
                    logger.info(
                        "New best Kendall Tau %.6f at epoch %d (saved to %s)",
                        best_kendall_tau,
                        epoch + 1,
                        best_model_path,
                    )
                else:
                    logger.info(
                        "New best Kendall Tau %.6f at epoch %d (save_model disabled)",
                        best_kendall_tau,
                        epoch + 1,
                    )
            elif early_stop_patience > 0:
                no_improve_epochs += 1
                if no_improve_epochs >= early_stop_patience:
                    logger.info(
                        "Early stopping at epoch %d (best tau %.6f at epoch %d)",
                        epoch + 1,
                        best_kendall_tau,
                        best_epoch,
                    )
                    early_stop = True
            model.train()
            if use_rl_policy:
                policy.train()
            if early_stop:
                break

        if log_alpha_std_vals:
            mean_log_alpha_std = sum(log_alpha_std_vals) / len(log_alpha_std_vals)
            logger.info("epoch %d mean log_alpha std %.6f", epoch, mean_log_alpha_std)
            if log_alpha_std_path is not None:
                with open(log_alpha_std_path, "a") as f:
                    f.write(f"{epoch},{mean_log_alpha_std:.6f}\n")
        if early_stop:
            break

    if getattr(cfg, "run_dir", None):
        history_path = os.path.join(cfg.run_dir, "eval_history.json")
        with open(history_path, "w") as f:
            json.dump(eval_history, f, indent=2)
        logger.info("Saved eval history to %s", history_path)

    if save_model and cfg.model_path:
        torch.save(model.state_dict(), cfg.model_path)
        if use_rl_policy:
            torch.save(policy.state_dict(), cfg.policy_path)
        logger.info("Saved model to %s", cfg.model_path)
    elif not save_model:
        logger.info("Skipping model save (save_model disabled).")
    elif cfg.model_path is None:
        logger.warning("save_model enabled but model_path is None; skipping save.")
    return model, policy

def evaluation(cfg, model=None, policy=None, step=None, use_best=False):
    logger = logging.getLogger("NumberSorting")
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
    else:
        device = "cpu"
    pin_memory = device == "cuda"
    if model is None:
        model = build_model(cfg).to(device)
        model_path = cfg.model_path
        if use_best:
            best_model_path = getattr(cfg, "best_model_path", None)
            if best_model_path and os.path.exists(best_model_path):
                model_path = best_model_path
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        model = model.to(device)

    mask_mode = getattr(cfg, "mask_mode", "estimated")
    model_type = getattr(cfg, "model_type", "conv")
    use_rl_policy = mask_mode == "rl_threshold"
    mask_eps = getattr(cfg, "mask_eps", 1e-9)
    if use_rl_policy:
        policy_hid_c = getattr(cfg, "rl_policy_hid_c", cfg.hid_c)
        if policy is None:
            policy = ThresholdPolicy(cfg.n_numbers, policy_hid_c).to(device)
            policy_path = getattr(cfg, "policy_path", None)
            if use_best:
                best_policy_path = getattr(cfg, "best_policy_path", None)
                if best_policy_path and os.path.exists(best_policy_path):
                    policy_path = best_policy_path
            if policy_path is None:
                model_path = getattr(cfg, "model_path", None)
                if model_path:
                    policy_path = os.path.join(os.path.dirname(model_path), "policy_weight.pth")
            if policy_path and os.path.exists(policy_path):
                policy.load_state_dict(torch.load(policy_path, map_location=device))
            else:
                logger.warning("Policy checkpoint not found; using random policy for evaluation.")
        else:
            policy = policy.to(device)
        policy.eval()
        rl_mask_frac = float(getattr(cfg, "rl_mask_frac", 0.2))
        rl_mask_frac = max(0.0, min(1.0, rl_mask_frac))

    dataset = NumberGenerator(cfg.n_numbers, cfg.n_test_lists, cfg.min_value, cfg.max_value, cfg.test_seed)
    test_loader = DataLoader(
        dataset,
        cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=False,
        pin_memory=pin_memory,
    )

    if step is None:
        logger.info("evaluation")
    else:
        logger.info("evaluation after epoch %d", step)
    prop_wrongs = []
    prop_any_wrongs = []
    kendall_taus = []
    mask_support_means = []
    rank_means = []
    rank_stds = []
    use_mask_flag = getattr(cfg, "use_rank_mask", False) or getattr(cfg, "use_masked_sinkhorn", False)
    use_mask = (mask_mode != "none") and (mask_mode != "estimated" or use_mask_flag)

    model.eval()
    with torch.inference_mode():
        for data in test_loader:
            X, ordered_X, permutation = data
            X = X.to(device, non_blocking=True)
            ordered_X = ordered_X.to(device, non_blocking=True)
            permutation = permutation.to(device, non_blocking=True)

            mask = None
            mask_stats = None
            attn_bias = None

            if use_mask:
                rank_mask_strength = getattr(cfg, "rank_mask_strength", 1.0)
                if mask_mode == "estimated":
                    rank_temp = getattr(cfg, "rank_temp", 0.1)
                    rank_sigma = getattr(cfg, "rank_sigma", 2.0)
                    mask, mask_stats, _ = build_rank_mask(
                        X,
                        rank_temp=rank_temp,
                        rank_sigma=rank_sigma,
                    )
                elif mask_mode == "rl_threshold":
                    features, x_norm = build_policy_features(X)
                    mean, logstd = policy(features)
                    dist = torch.distributions.Normal(mean, logstd.exp())
                    thresholds = dist.sample()
                    mask, mask_stats = build_rl_threshold_mask(x_norm, thresholds, rl_mask_frac)
                elif mask_mode in ("true", "true+k"):
                    extra_k = getattr(cfg, "true_mask_extra_k", 0) if mask_mode == "true+k" else 0
                    mask, mask_stats, _ = build_true_mask(permutation, extra_k=extra_k)
                else:
                    forbid_k = getattr(cfg, "true_mask_forbid_k", 0)
                    mask, mask_stats, _ = build_forbid_mask(permutation, forbid_k=forbid_k)

                if model_type == "transformer" and mask_mode != "forbid":
                    attn_bias = build_attention_bias_from_mask(mask, rank_mask_strength, mask_eps)

            log_alpha = model(X[:, None], attn_bias=attn_bias)

            if use_mask:
                if mask_mode == "rl_threshold":
                    log_alpha = log_alpha + rank_mask_strength * torch.log(mask.clamp(min=mask_eps))
                else:
                    log_alpha = log_alpha + rank_mask_strength * torch.log(mask + 1e-9)
                rank_means.append(mask_stats["rank_mean"].detach())
                rank_stds.append(mask_stats["rank_std"].detach())
                mask_support_means.append(mask_stats["support_mean"].detach())

            assingment_matrix = gumbel_sinkhorn_ops.gumbel_matching(log_alpha, noise=False)

            est_permutation = assingment_matrix.max(1)[1].float()

            prop_wrong = (permutation - est_permutation).sign().abs().mean(1)
            prop_any_wrong = (permutation - est_permutation).sign().abs().sum(1).sign()
            kendall_tau_batch = kendall_tau(est_permutation, permutation)

            prop_wrongs.append(prop_wrong)
            prop_any_wrongs.append(prop_any_wrong)
            kendall_taus.append(kendall_tau_batch)

    mean_prop_wrong = torch.cat(prop_wrongs).mean()
    mean_prop_any_wrong = torch.cat(prop_any_wrongs).mean()
    mean_kendall_tau = torch.cat(kendall_taus).mean()

    logger.info("Mean Prop Wrongs %f", mean_prop_wrong)
    logger.info("Mean Prop Any Wrongs %f", mean_prop_any_wrong)
    logger.info("Mean Kendall Tau %f", mean_kendall_tau)
    use_rank_mask = getattr(cfg, "use_rank_mask", False) or getattr(cfg, "use_masked_sinkhorn", False)
    if use_mask and mask_support_means:
        logger.info(
            "Rank mask stats | rank_mean %f | rank_std %f | mask_support_mean %f",
            torch.stack(rank_means).mean(),
            torch.stack(rank_stds).mean(),
            torch.stack(mask_support_means).mean(),
        )
    return {
        "mean_prop_wrong": mean_prop_wrong,
        "mean_prop_any_wrong": mean_prop_any_wrong,
        "mean_kendall_tau": mean_kendall_tau,
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # gumbel sinkhorn option
    parser.add_argument("--tau", default=1.0, type=float, help="temperture parameter")
    parser.add_argument(
        "--tau_anneal_end",
        default=None,
        type=float,
        help="final tau for annealing (disabled when unset)",
    )
    parser.add_argument(
        "--tau_anneal_schedule",
        default="linear",
        choices=["linear", "exp"],
        help="annealing schedule for tau",
    )
    parser.add_argument(
        "--tau_anneal_epochs",
        default=0,
        type=int,
        help="epochs to anneal tau over (0 uses total epochs)",
    )
    parser.add_argument("--tau_adapt", action="store_true", help="enable entropy-based adaptive tau")
    parser.add_argument(
        "--tau_adapt_source",
        default="log_alpha",
        choices=["log_alpha", "sinkhorn"],
        help="entropy source for adaptive tau",
    )
    parser.add_argument(
        "--tau_adapt_combine",
        default="avg",
        choices=["avg", "prod"],
        help="combine row/col tau via average or product",
    )
    parser.add_argument(
        "--tau_adapt_entropy_thresh",
        default=0.75,
        type=float,
        help="normalized entropy threshold to start increasing tau",
    )
    parser.add_argument(
        "--tau_adapt_boost_max",
        default=0.1,
        type=float,
        help="maximum relative tau increase over the annealed tau",
    )
    parser.add_argument(
        "--tau_adapt_start_epoch",
        default=1,
        type=int,
        help="1-based epoch to start adaptive tau",
    )
    parser.add_argument("--n_sink_iter", default=20, type=int, help="number of iterations for sinkhorn normalization")
    parser.add_argument("--n_samples", default=5, type=int, help="number of samples from gumbel-sinkhorn distribution")
    parser.add_argument("--use_masked_sinkhorn", action="store_true", help="enable masked Sinkhorn (deprecated, use --use_rank_mask)")
    parser.add_argument("--use_rank_mask", action="store_true", help="enable rank-based mask for Sinkhorn/Hungarian")
    parser.add_argument("--rank_temp", default=0.1, type=float, help="temperature for soft rank comparisons")
    parser.add_argument("--rank_sigma", default=2.0, type=float, help="Gaussian width over positions for rank mask")
    parser.add_argument("--rank_mask_strength", default=1.0, type=float, help="log-space multiplier applied to the rank mask")
    parser.add_argument("--mask_mode", default="estimated", choices=["none", "estimated", "true", "true+k", "forbid", "rl_threshold"], help="mask source: none/estimated/ground-truth/ground-truth with extra columns/ground-truth forbidding random wrong columns/rl_threshold")
    parser.add_argument("--true_mask_extra_k", default=0, type=int, help="additional random columns to keep per row when mask_mode=true+k")
    parser.add_argument("--true_mask_forbid_k", default=0, type=int, help="number of wrong columns to forbid per row when mask_mode=forbid")
    parser.add_argument("--rl_mask_frac", default=0.2, type=float, help="fraction of opposite-region columns to forbid per row for mask_mode=rl_threshold")
    parser.add_argument("--mask_topk", default=8, type=int, help="(deprecated) number of candidates to keep per row in the old mask")
    parser.add_argument("--mask_strength", default=20.0, type=float, help="(deprecated) scale on smoothness costs for the old mask")
    # datase option
    parser.add_argument("--n_numbers", default=50, type=int, help="number of sorted numbers")
    parser.add_argument("--n_train_lists", default=10000, type=int, help="number of sorted number lists for training")
    parser.add_argument("--n_test_lists", default=100, type=int, help="number of sorted number lists for evaluation")
    parser.add_argument("--min_value", default=0, type=float, help="minimum value of uniform distribution")
    parser.add_argument("--max_value", default=1, type=float, help="maximum value of uniform distribution")
    parser.add_argument("--train_seed", default=1, type=int, help="random seed for training data generation")
    parser.add_argument("--test_seed", default=2, type=int, help="random seed for evaluation data generation")
    parser.add_argument("--seed", default=None, type=int, help="global RNG seed for model/training (disabled when unset)")
    parser.add_argument("--num_workers", default=8, type=int, help="number of threads for CPU parallel")
    # optimizer option
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument("--batch_size", default=100, type=int, help="mini-batch size")
    parser.add_argument("--epochs", default=100, type=int, help="number of epochs")
    parser.add_argument("--eval_every", default=5, type=int, help="run evaluation every N epochs (0 disables)")
    parser.add_argument(
        "--early_stop_patience",
        default=0,
        type=int,
        help="stop after N evals without Kendall tau improvement (0 disables)",
    )
    parser.add_argument(
        "--early_stop_min_delta",
        default=0.001,
        type=float,
        help="minimum Kendall tau improvement to reset early stopping",
    )
    # misc
    parser.add_argument("--hid_c", default=64, type=int, help="number of hidden channels / transformer d_model")
    parser.add_argument("--model_type", default="conv", choices=["conv", "transformer"], help="model family to train")
    parser.add_argument("--transformer_n_layers", default=8, type=int, help="number of transformer encoder blocks")
    parser.add_argument("--transformer_n_heads", default=4, type=int, help="number of attention heads per block")
    parser.add_argument("--transformer_ffn_dim", default=128, type=int, help="feedforward width inside transformer blocks")
    parser.add_argument("--transformer_dropout", default=0.0, type=float, help="dropout in transformer blocks")
    parser.add_argument("--out_dir", default="log", type=str, help="/path/to/output directory")
    parser.add_argument("--display", default=50, type=int, help="display loss every 'display' iteration. if set to 0, won't display")
    parser.add_argument("--eval_only", action="store_true", help="evaluation without training")
    parser.add_argument("--eval_best", action="store_true", help="evaluate best checkpoint if available")
    parser.add_argument("--save_model", action="store_true", help="save final/best checkpoints")
    parser.add_argument("--save_log_alpha", action="store_true", help="save log_alpha_std.csv during training")
    parser.add_argument("--model_path", default=None, type=str, help="path to load/save model checkpoint")
    parser.add_argument("--smooth_loss_weight", default=1, type=float, help="weight for smoothness loss (set >0 to enable)")
    parser.add_argument("--smooth_direction", default="asc", choices=["asc", "desc"], help="expected ordering direction for smoothness loss")
    parser.add_argument("--mse_weight", default=0, type=float, help="weight for supervised MSE loss (can be 0 for unsupervised)")
    parser.add_argument("--spread_weight", default=0.0, type=float, help="weight for spread regularizer on est_ordered_X")
    parser.add_argument("--entropy_weight", default=0.0, type=float, help="weight for permutation entropy penalty")
    parser.add_argument("--entropy_eps", default=1e-9, type=float, help="epsilon for permutation entropy log")

    cfg = parser.parse_args()

    if not os.path.exists(cfg.out_dir):
        os.mkdir(cfg.out_dir)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(cfg.out_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    if cfg.model_path is None:
        cfg.model_path = os.path.join(run_dir, "model_weight.pth")
    cfg.run_dir = run_dir

    setup_logging(os.path.join(run_dir, "console.log"))
    logger = logging.getLogger("NumberSorting")

    logger.info(cfg)

    trained_model = None
    trained_policy = None
    if not cfg.eval_only:
        trained_model, trained_policy = train(cfg)
    use_best = getattr(cfg, "eval_best", False) and getattr(cfg, "save_model", False)
    if trained_model is not None and not use_best:
        evaluation(cfg, model=trained_model, policy=trained_policy)
    else:
        evaluation(cfg, use_best=use_best)
