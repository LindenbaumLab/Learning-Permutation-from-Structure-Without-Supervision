import argparse
import logging
import math
import os
import random
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import FCModel
from number_utils import NumberGenerator
from train import evaluation as sinkhorn_evaluation
from train import train as sinkhorn_train


# --- Helpers ---------------------------------------------------------------


def setup_logger(log_path: Optional[str] = None):
    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = []
    if log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)
    logging.getLogger("NumberSorting").handlers = []


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def linear_n_numbers(start: int, end: int, steps: int) -> List[int]:
    return [int(x) for x in np.linspace(start, end, steps, dtype=int)]


def mean_and_ci(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = arr.mean(axis=-1)
    std = arr.std(axis=-1, ddof=1) if arr.shape[-1] > 1 else np.zeros_like(mean)
    sem = std / math.sqrt(arr.shape[-1]) if arr.shape[-1] > 0 else std
    ci = 1.96 * sem
    return mean, ci


def parse_value_ranges(values: str) -> List[Tuple[float, float]]:
    ranges: List[Tuple[float, float]] = []
    if not values:
        return ranges
    for raw in values.split(";"):
        raw = raw.strip()
        if not raw:
            continue
        raw = raw.strip("[]()")
        if ":" in raw:
            parts = raw.split(":", 1)
        else:
            parts = raw.split(",", 1)
        if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
            raise ValueError(f"Bad range: {raw!r}. Use min,max or min:max.")
        min_val = float(parts[0].strip())
        max_val = float(parts[1].strip())
        if max_val < min_val:
            raise ValueError(f"Bad range: {raw!r}. max_value < min_value.")
        ranges.append((min_val, max_val))
    return ranges


def parse_n_values(values: str) -> List[int]:
    n_values: List[int] = []
    if not values:
        return n_values
    raw = values.strip().strip("[]()")
    if ":" in raw:
        parts = [p.strip() for p in raw.split(":") if p.strip()]
        if len(parts) != 3:
            raise ValueError(f"Bad n_values: {raw!r}. Use comma-separated ints or start:end:steps.")
        try:
            start = int(parts[0])
            end = int(parts[1])
            steps = int(parts[2])
        except ValueError as exc:
            raise ValueError(f"Bad n_values: {raw!r}. Use integers for start:end:steps.") from exc
        if steps <= 0:
            raise ValueError(f"Bad n_values: {raw!r}. steps must be > 0.")
        return linear_n_numbers(start, end, steps)
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return n_values
    try:
        n_values = [int(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"Bad n_values: {raw!r}. Use comma-separated integers.") from exc
    return n_values


def parse_methods(values: str) -> List[str]:
    if not values:
        return []
    return [value.strip() for value in values.split(",") if value.strip()]


def resolve_method_filter(
    tokens: List[str],
    sinkhorn_methods: List[str],
    baseline_methods: Set[str],
) -> Optional[Set[str]]:
    token_set = {token.strip().lower().replace("-", "_") for token in tokens if token.strip()}
    if not token_set or "all" in token_set:
        return None
    sinkhorn_methods_set = {method.lower() for method in sinkhorn_methods}
    baseline_methods_set = {method.lower() for method in baseline_methods}
    base_names: Set[str] = set()
    model_tags: Set[str] = set()
    for method in sinkhorn_methods_set:
        if "_" not in method:
            continue
        base_name, model_tag = method.rsplit("_", 1)
        base_names.add(base_name)
        model_tags.add(model_tag)

    allowed: Set[str] = set()
    recognized: Set[str] = set()
    for token in token_set:
        if token in ("baseline", "baselines"):
            allowed |= baseline_methods_set
            recognized.add(token)
            continue
        if token == "sinkhorn":
            allowed |= sinkhorn_methods_set
            recognized.add(token)
            continue
        if token in baseline_methods_set:
            allowed.add(token)
            recognized.add(token)
        if token in sinkhorn_methods_set:
            allowed.add(token)
            recognized.add(token)
        if token in base_names:
            allowed |= {method for method in sinkhorn_methods_set if method.startswith(f"{token}_")}
            recognized.add(token)
        if token in model_tags:
            allowed |= {method for method in sinkhorn_methods_set if method.endswith(f"_{token}")}
            recognized.add(token)
    unknown = token_set - recognized
    if unknown:
        all_methods = sorted(sinkhorn_methods_set | baseline_methods_set)
        base_names_list = sorted(base_names)
        model_tags_list = sorted(model_tags)
        raise ValueError(
            f"Unknown --methods entries: {sorted(unknown)}. Known methods: {all_methods}. "
            f"Groups: sinkhorn, baselines. Base names: {base_names_list}. Model tags: {model_tags_list}."
        )
    if not allowed:
        raise ValueError("No methods selected by --methods.")
    return allowed


def format_range_label(min_val: float, max_val: float) -> str:
    return f"[{min_val:g},{max_val:g}]"


def format_range_tag(min_val: float, max_val: float) -> str:
    return f"{min_val:g}_{max_val:g}"


# --- NeuralSort components -------------------------------------------------


class ScoreModel(nn.Module):
    """
    Simple 1D conv model that outputs a scalar score per item for NeuralSort.
    """

    def __init__(self, hid_c: int, out_c: int):
        super().__init__()
        self.g1 = nn.Sequential(nn.Conv1d(1, hid_c, 1), nn.ReLU(True))
        self.g2 = nn.Conv1d(hid_c, 1, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (B, n) or (B,1,n)
        if x.dim() == 2:
            x = x[:, None]
        scores = self.g2(self.g1(x)).squeeze(1)  # (B, n)
        return scores


def neural_sort(scores: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """
    NeuralSort operator from "Differentiable Sorting and Ranking via Optimization".
    Returns a soft permutation matrix of shape (B, n, n).
    """
    b, n = scores.shape
    scores = scores.unsqueeze(-1)  # (B, n, 1)
    one = torch.ones((n, 1), device=scores.device, dtype=scores.dtype)
    A_s = torch.abs(scores - scores.transpose(1, 2))  # (B, n, n)
    B = torch.matmul(A_s, torch.matmul(one, one.transpose(0, 1)))  # (B, n, n)
    scaling = (n + 1 - 2 * (torch.arange(n, device=scores.device, dtype=scores.dtype) + 1)).view(1, 1, n)
    C = torch.matmul(scores, scaling)  # (B, n, n)
    P_hat = torch.softmax((C - B) / tau, dim=-1)
    return P_hat




def kendall_tau_from_perm(pred_perm: torch.Tensor, true_perm: torch.Tensor) -> torch.Tensor:
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
    return (concordant - discordant) / (0.5 * n * (n - 1))


def greedy_decode_perm(P: torch.Tensor) -> torch.Tensor:
    """
    Greedy one-to-one decoding from soft permutation matrix.
    P: (B, n, n) where rows=positions, cols=items.
    Returns: (B, n) pos->item permutation with unique items.
    """
    B, n, _ = P.shape
    perm = torch.empty((B, n), device=P.device, dtype=torch.long)
    for b in range(B):
        used = torch.zeros(n, device=P.device, dtype=torch.bool)
        for k in range(n):
            row = P[b, k].clone()
            row[used] = -1e9
            j = row.argmax().item()
            perm[b, k] = j
            used[j] = True
    return perm

# --- Config dataclass ------------------------------------------------------


@dataclass
class RunConfig:
    n_numbers: int
    n_train_lists: int
    n_test_lists: int
    smooth_loss_weight: float
    mse_weight: float
    mask_mode: str
    use_rank_mask: bool
    true_mask_forbid_k: int
    rank_mask_strength: float
    rank_temp: float
    rank_sigma: float
    tau: float
    tau_anneal_end: Optional[float]
    tau_anneal_schedule: str
    tau_anneal_epochs: int
    tau_adapt: bool
    tau_adapt_source: str
    tau_adapt_combine: str
    tau_adapt_entropy_thresh: float
    tau_adapt_boost_max: float
    tau_adapt_start_epoch: int
    n_sink_iter: int
    n_samples: int
    lr: float
    batch_size: int
    num_workers: int
    epochs: int
    eval_every: int
    early_stop_patience: int
    early_stop_min_delta: float
    eval_best: bool
    hid_c: int
    model_type: str
    transformer_n_layers: int
    transformer_n_heads: int
    transformer_ffn_dim: int
    transformer_dropout: float
    spread_weight: float
    entropy_weight: float
    entropy_eps: float
    out_dir: str
    run_dir: str
    save_model: bool = False
    save_log_alpha: bool = False
    min_value: float = 0.0
    max_value: float = 1.0


# --- Training helpers ------------------------------------------------------


def run_sinkhorn(cfg: RunConfig, seed: int) -> float:
    from types import SimpleNamespace

    cfg_ns = SimpleNamespace(
        seed=seed,
        tau=cfg.tau,
        tau_anneal_end=cfg.tau_anneal_end,
        tau_anneal_schedule=cfg.tau_anneal_schedule,
        tau_anneal_epochs=cfg.tau_anneal_epochs,
        tau_adapt=cfg.tau_adapt,
        tau_adapt_source=cfg.tau_adapt_source,
        tau_adapt_combine=cfg.tau_adapt_combine,
        tau_adapt_entropy_thresh=cfg.tau_adapt_entropy_thresh,
        tau_adapt_boost_max=cfg.tau_adapt_boost_max,
        tau_adapt_start_epoch=cfg.tau_adapt_start_epoch,
        n_sink_iter=cfg.n_sink_iter,
        n_samples=cfg.n_samples,
        n_numbers=cfg.n_numbers,
        n_train_lists=cfg.n_train_lists,
        n_test_lists=cfg.n_test_lists,
        min_value=cfg.min_value,
        max_value=cfg.max_value,
        train_seed=seed,
        test_seed=seed + 1,
        num_workers=cfg.num_workers,
        lr=cfg.lr,
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        eval_every=cfg.eval_every,
        early_stop_patience=cfg.early_stop_patience,
        early_stop_min_delta=cfg.early_stop_min_delta,
        eval_best=cfg.eval_best,
        hid_c=cfg.hid_c,
        model_type=cfg.model_type,
        transformer_n_layers=cfg.transformer_n_layers,
        transformer_n_heads=cfg.transformer_n_heads,
        transformer_ffn_dim=cfg.transformer_ffn_dim,
        transformer_dropout=cfg.transformer_dropout,
        spread_weight=cfg.spread_weight,
        entropy_weight=cfg.entropy_weight,
        entropy_eps=cfg.entropy_eps,
        out_dir=cfg.out_dir,
        display=0,
        eval_only=False,
        smooth_loss_weight=cfg.smooth_loss_weight,
        smooth_direction="asc",
        mse_weight=cfg.mse_weight,
        use_rank_mask=cfg.use_rank_mask,
        mask_mode=cfg.mask_mode,
        true_mask_forbid_k=cfg.true_mask_forbid_k,
        rank_temp=cfg.rank_temp,
        rank_sigma=cfg.rank_sigma,
        rank_mask_strength=cfg.rank_mask_strength,
        model_path=os.path.join(cfg.run_dir, f"model_seed{seed}.pth"),
        best_model_path=os.path.join(cfg.run_dir, "model_best.pth"),
        run_dir=cfg.run_dir,
        save_model=cfg.save_model,
        save_log_alpha=cfg.save_log_alpha,
    )
    os.makedirs(cfg_ns.run_dir, exist_ok=True)
    set_seed(seed)
    model, policy = sinkhorn_train(cfg_ns)
    if cfg_ns.eval_best and cfg.save_model:
        metrics = sinkhorn_evaluation(cfg_ns, use_best=cfg_ns.eval_best)
    else:
        if cfg_ns.eval_best and not cfg.save_model:
            logging.getLogger("benchmark").warning(
                "eval_best requested but save_model disabled; using last model."
            )
        metrics = sinkhorn_evaluation(cfg_ns, model=model, policy=policy)
    return metrics["mean_kendall_tau"].item()


def run_neuralsort(
    n_numbers: int,
    n_train_lists: int,
    n_test_lists: int,
    smooth_loss_weight: float,
    mse_weight: float,
    tau: float,
    lr: float,
    batch_size: int,
    epochs: int,
    hid_c: int,
    seed: int,
    device: Optional[str] = None,
    train_free: bool = True,
    min_value: float = 0.0,
    max_value: float = 1.0,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(seed)
    model = None
    if not train_free:
        model = ScoreModel(hid_c, 1).to(device)
        optimizer = optim.Adam(model.parameters(), lr)
        dataset = NumberGenerator(n_numbers, n_train_lists, min_value, max_value, seed)
        train_loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=0, drop_last=True)
        for _ in range(epochs):
            for X, ordered_X, _ in train_loader:
                X = X.to(device)
                ordered_X = ordered_X.to(device)
                scores = model(X)
                soft_perm = neural_sort(-scores, tau=tau)
                est_ordered = torch.bmm(soft_perm.transpose(1, 2), X.unsqueeze(-1)).squeeze(-1)
                sup_loss = torch.nn.functional.mse_loss(est_ordered, ordered_X) if mse_weight > 0 else 0.0
                if smooth_loss_weight > 0:
                    diff = est_ordered[:, 1:] - est_ordered[:, :-1]
                    violation = torch.relu(-diff)
                    smooth_loss = (violation ** 2).mean()
                else:
                    smooth_loss = 0.0
                total_loss = mse_weight * sup_loss + smooth_loss_weight * smooth_loss
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
    # Evaluation
    test_dataset = NumberGenerator(n_numbers, n_test_lists, min_value, max_value, seed + 1)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True, num_workers=0, drop_last=False)
    if model is not None:
        model.eval()
    taus = []
    with torch.no_grad():
        for X, _, permutation in test_loader:
            X = X.to(device)
            permutation = permutation.to(device)
            scores = X if train_free else model(X)
            soft_perm = neural_sort(-scores, tau=tau)  # NEGATE to sort ascending
            pred_pos_to_item = greedy_decode_perm(soft_perm.transpose(1, 2))
            taus.append(kendall_tau_from_perm(pred_pos_to_item, permutation))
    return torch.cat(taus).mean().item()


def run_softsort_baseline(
    n_numbers: int,
    n_test_lists: int,
    tau: float,
    batch_size: int,
    seed: int,
    device: Optional[str] = None,
    min_value: float = 0.0,
    max_value: float = 1.0,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(seed)
    sorter = SoftSort(tau=tau, hard=False).to(device)
    test_dataset = NumberGenerator(n_numbers, n_test_lists, min_value, max_value, seed + 1)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True, num_workers=0, drop_last=False)
    taus = []
    with torch.no_grad():
        for X, _, permutation in test_loader:
            X = X.to(device)
            permutation = permutation.to(device)
            soft_perm = sorter(-X)
            pred_pos_to_item = greedy_decode_perm(soft_perm)
            taus.append(kendall_tau_from_perm(pred_pos_to_item, permutation))
    return torch.cat(taus).mean().item()


def run_fast_soft_sort_baseline(
    n_numbers: int,
    n_test_lists: int,
    tau: float,
    batch_size: int,
    seed: int,
    min_value: float = 0.0,
    max_value: float = 1.0,
):
    device = "cpu"
    set_seed(seed)
    test_dataset = NumberGenerator(n_numbers, n_test_lists, min_value, max_value, seed + 1)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True, num_workers=0, drop_last=False)
    taus = []
    with torch.no_grad():
        for X, _, permutation in test_loader:
            X = X.to(device)
            permutation = permutation.to(device)
            soft_ranks = fast_soft_rank(
                X,
                direction="ASCENDING",
                regularization_strength=tau,
                regularization="l2",
            )
            pred_pos_to_item = soft_ranks.argsort(dim=1)
            taus.append(kendall_tau_from_perm(pred_pos_to_item, permutation))
    return torch.cat(taus).mean().item()


def run_diffsort_baseline(
    n_numbers: int,
    n_test_lists: int,
    tau: float,
    batch_size: int,
    seed: int,
    steepness: Optional[float] = None,
    device: Optional[str] = None,
    min_value: float = 0.0,
    max_value: float = 1.0,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if steepness is None:
        steepness = max(1.0, 1.0 / tau)
    set_seed(seed)
    sorter = DiffSortNet("odd_even", n_numbers, device=device, steepness=steepness)
    test_dataset = NumberGenerator(n_numbers, n_test_lists, min_value, max_value, seed + 1)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True, num_workers=0, drop_last=False)
    taus = []
    with torch.no_grad():
        for X, _, permutation in test_loader:
            X = X.to(device)
            permutation = permutation.to(device)
            _, soft_perm = sorter(X)
            pred_pos_to_item = greedy_decode_perm(soft_perm.transpose(1, 2))
            taus.append(kendall_tau_from_perm(pred_pos_to_item, permutation))
    return torch.cat(taus).mean().item()


# --- Benchmark driver ------------------------------------------------------


def benchmark(args):
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.out_dir = os.path.join(args.out_dir, run_id)
    os.makedirs(args.out_dir, exist_ok=True)
    setup_logger(os.path.join(args.out_dir, "benchmark.log"))
    logger = logging.getLogger("benchmark")
    method_tokens = parse_methods(args.methods)
    allowed_methods: Optional[Set[str]] = None
    baseline_methods = {"neural_sort", "softsort", "fast_soft_sort", "diffsort"}
    if args.seed is not None:
        seeds = [int(args.seed)]
    else:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    value_ranges = parse_value_ranges(args.value_ranges)
    if not value_ranges:
        raise ValueError("value_ranges is empty.")
    n_values = parse_n_values(args.n_values)
    if not n_values:
        raise ValueError("n_values is empty.")

    results = []  # list of (min_value, max_value, method, n, seed, kendall_tau)

    for min_val, max_val in value_ranges:
        range_label = format_range_label(min_val, max_val)
        range_tag = format_range_tag(min_val, max_val)
        for n in n_values:
            logger.info("Running range=%s n_numbers=%d", range_label, n)
            forbid_10 = max(1, int(0.10 * n))
            forbid_30 = max(1, int(0.30 * n))

            method_defs = [
                (
                    "supervised",
                    {
                        "smooth_loss_weight": 0.0,
                        "mse_weight": 1.0,
                        "mask_mode": "none",
                        "use_rank_mask": False,
                        "true_mask_forbid_k": 0,
                        "tau_adapt": args.tau_adapt,
                    },
                ),
                (
                    "unsupervised",
                    {
                        "smooth_loss_weight": 1.0,
                        "mse_weight": 0.0,
                        "mask_mode": "none",
                        "use_rank_mask": False,
                        "true_mask_forbid_k": 0,
                        "tau_adapt": args.tau_adapt,
                    },
                ),
                (
                    "unsup_tau_adapt",
                    {
                        "smooth_loss_weight": 1.0,
                        "mse_weight": 0.0,
                        "mask_mode": "none",
                        "use_rank_mask": False,
                        "true_mask_forbid_k": 0,
                        "tau_adapt": True,
                    },
                ),
                # (
                #     "unsup_forbid_10pct",
                #     {
                #         "smooth_loss_weight": 1.0,
                #         "mse_weight": 0.0,
                #         "mask_mode": "forbid",
                #         "use_rank_mask": True,
                #         "true_mask_forbid_k": forbid_10,
                #     },
                # ),
                # (
                #     "unsup_forbid_30pct",
                #     {
                #         "smooth_loss_weight": 1.0,
                #         "mse_weight": 0.0,
                #         "mask_mode": "forbid",
                #         "use_rank_mask": True,
                #         "true_mask_forbid_k": forbid_30,
                #     },
                # ),
            ]
            model_defs = [
                ("conv", "conv"),
                # ("transformer", "transformer"),
            ]
            sinkhorn_method_names = [
                f"{base_name}_{model_tag}" for base_name, _ in method_defs for model_tag, _ in model_defs
            ]
            if method_tokens and allowed_methods is None:
                allowed_methods = resolve_method_filter(method_tokens, sinkhorn_method_names, baseline_methods)
                if allowed_methods is not None:
                    logger.info("Filtering to methods: %s", ", ".join(sorted(allowed_methods)))
            method_cfgs: Dict[str, RunConfig] = {}
            for base_name, base_cfg in method_defs:
                for model_tag, model_type in model_defs:
                    lr = args.transformer_lr if model_type == "transformer" else args.lr
                    epochs = args.transformer_epochs if model_type == "transformer" else args.epochs
                    spread_weight = args.transformer_spread_weight if model_type == "transformer" else 0.0
                    entropy_weight = args.transformer_entropy_weight if model_type == "transformer" else 0.0
                    entropy_eps = args.transformer_entropy_eps
                    method_key = f"{base_name}_{model_tag}"
                    use_tau_anneal = base_name.startswith("unsup")
                    run_dir = os.path.join(
                        args.out_dir,
                        f"range_{range_tag}",
                        f"bench_{base_name}_{model_tag}_n{n}",
                    )
                    method_cfgs[method_key] = RunConfig(
                        n_numbers=n,
                        n_train_lists=args.n_train_lists,
                        n_test_lists=args.n_test_lists,
                        smooth_loss_weight=base_cfg["smooth_loss_weight"],
                        mse_weight=base_cfg["mse_weight"],
                        mask_mode=base_cfg["mask_mode"],
                        use_rank_mask=base_cfg["use_rank_mask"],
                        true_mask_forbid_k=base_cfg["true_mask_forbid_k"],
                        rank_mask_strength=args.rank_mask_strength,
                        rank_temp=args.rank_temp,
                        rank_sigma=args.rank_sigma,
                        tau=args.tau,
                        tau_anneal_end=args.tau_anneal_end if use_tau_anneal else None,
                        tau_anneal_schedule=args.tau_anneal_schedule,
                        tau_anneal_epochs=args.tau_anneal_epochs,
                        tau_adapt=base_cfg["tau_adapt"],
                        tau_adapt_source=args.tau_adapt_source,
                        tau_adapt_combine=args.tau_adapt_combine,
                        tau_adapt_entropy_thresh=args.tau_adapt_entropy_thresh,
                        tau_adapt_boost_max=args.tau_adapt_boost_max,
                        tau_adapt_start_epoch=args.tau_adapt_start_epoch,
                        n_sink_iter=args.n_sink_iter,
                        n_samples=args.n_samples,
                        lr=lr,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        epochs=epochs,
                        eval_every=args.eval_every,
                        early_stop_patience=args.early_stop_patience,
                        early_stop_min_delta=args.early_stop_min_delta,
                        eval_best=args.eval_best,
                        hid_c=args.hid_c,
                        model_type=model_type,
                        transformer_n_layers=args.transformer_n_layers,
                        transformer_n_heads=args.transformer_n_heads,
                        transformer_ffn_dim=args.transformer_ffn_dim,
                        transformer_dropout=args.transformer_dropout,
                        spread_weight=spread_weight,
                        entropy_weight=entropy_weight,
                        entropy_eps=entropy_eps,
                        out_dir=args.out_dir,
                        run_dir=run_dir,
                        save_model=args.save_model,
                        save_log_alpha=args.save_log_alpha,
                        min_value=min_val,
                        max_value=max_val,
                    )

            if allowed_methods is not None:
                method_cfgs = {
                    method_name: cfg
                    for method_name, cfg in method_cfgs.items()
                    if method_name.lower() in allowed_methods
                }

            for method_name, cfg in method_cfgs.items():
                for seed in seeds:
                    ktau = run_sinkhorn(cfg, seed)
                    results.append((min_val, max_val, method_name, n, seed, ktau))
                    logger.info(
                        "range=%s method=%s n=%d seed=%d kendall_tau=%.4f",
                        range_label,
                        method_name,
                        n,
                        seed,
                        ktau,
                    )

            # NeuralSort runs separately.
            if allowed_methods is None or "neural_sort" in allowed_methods:
                for seed in seeds:
                    ktau = run_neuralsort(
                        n_numbers=n,
                        n_train_lists=args.n_train_lists,
                        n_test_lists=args.n_test_lists,
                        smooth_loss_weight=1.0,
                        mse_weight=0.01,
                        tau=args.tau,
                        lr=args.lr,
                        batch_size=args.batch_size,
                        epochs=args.epochs,
                        hid_c=args.hid_c,
                        seed=seed,
                        train_free=True,
                        min_value=min_val,
                        max_value=max_val,
                    )
                    results.append((min_val, max_val, "neural_sort", n, seed, ktau))
                    logger.info(
                        "range=%s method=neural_sort n=%d seed=%d kendall_tau=%.4f",
                        range_label,
                        n,
                        seed,
                        ktau,
                    )

            baseline_defs = [
                ("softsort", run_softsort_baseline),
                ("fast_soft_sort", run_fast_soft_sort_baseline),
                ("diffsort", run_diffsort_baseline),
            ]
            for method_name, runner in baseline_defs:
                if allowed_methods is not None and method_name not in allowed_methods:
                    continue
                for seed in seeds:
                    extra_kwargs = {}
                    if method_name == "diffsort":
                        extra_kwargs["steepness"] = args.diffsort_steepness
                    ktau = runner(
                        n_numbers=n,
                        n_test_lists=args.n_test_lists,
                        tau=args.tau,
                        batch_size=args.batch_size,
                        seed=seed,
                        min_value=min_val,
                        max_value=max_val,
                        **extra_kwargs,
                    )
                    results.append((min_val, max_val, method_name, n, seed, ktau))
                    logger.info(
                        "range=%s method=%s n=%d seed=%d kendall_tau=%.4f",
                        range_label,
                        method_name,
                        n,
                        seed,
                        ktau,
                    )

    # Save CSV.
    csv_path = os.path.join(args.out_dir, "benchmark_kendall_tau.csv")
    with open(csv_path, "w") as f:
        f.write("min_value,max_value,method,n_numbers,seed,kendall_tau\n")
        for min_val, max_val, method, n, seed, ktau in results:
            f.write(f"{min_val},{max_val},{method},{n},{seed},{ktau:.6f}\n")
    logger.info("Saved CSV to %s", csv_path)

    # Aggregate for plotting.
    methods = sorted({r[2] for r in results})
    n_to_idx = {n: i for i, n in enumerate(n_values)}
    seed_to_idx = {seed: i for i, seed in enumerate(seeds)}
    range_meta = [
        (min_val, max_val, format_range_label(min_val, max_val), format_range_tag(min_val, max_val))
        for min_val, max_val in value_ranges
    ]
    data: Dict[Tuple[str, str], np.ndarray] = {}
    for _, _, label, _ in range_meta:
        for method in methods:
            data[(method, label)] = np.zeros((len(n_values), len(seeds)))
    for min_val, max_val, method, n, seed, ktau in results:
        label = format_range_label(min_val, max_val)
        data[(method, label)][n_to_idx[n], seed_to_idx[seed]] = ktau

    if len(range_meta) == 1:
        _, _, range_label, _ = range_meta[0]
        plt.figure(figsize=(8, 5))
        base_styles = {
            "supervised": "-",
            "unsupervised": "--",
            # "unsup_forbid_10pct": "-.",
            # "unsup_forbid_30pct": ":",
        }
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        model_types = []
        for method in methods:
            if method in baseline_methods:
                continue
            if "_" not in method:
                continue
            _, model_type = method.rsplit("_", 1)
            if model_type not in model_types:
                model_types.append(model_type)
        base_colors = {
            model_type: color_cycle[i % len(color_cycle)] if color_cycle else None
            for i, model_type in enumerate(model_types)
        }
        baseline_colors = {}
        if "neural_sort" in methods:
            baseline_colors["neural_sort"] = "black"
        extra_baselines = [m for m in methods if m in baseline_methods and m != "neural_sort"]
        for i, method in enumerate(extra_baselines):
            baseline_colors[method] = color_cycle[i % len(color_cycle)] if color_cycle else None

        def lighten_color(color, amount=0.35):
            rgb = np.array(mcolors.to_rgb(color))
            return tuple(rgb + (1.0 - rgb) * amount)

        def darken_color(color, amount=0.2):
            rgb = np.array(mcolors.to_rgb(color))
            return tuple(rgb * (1.0 - amount))

        for method in methods:
            mean, ci = mean_and_ci(data[(method, range_label)])
            if method in baseline_methods:
                color = baseline_colors.get(method, "C0")
                linestyle = "-"
            else:
                base_name, model_type = method.rsplit("_", 1)
                base_color = base_colors.get(model_type, None) or "C0"
                if base_name == "supervised":
                    color = lighten_color(base_color, 0.4)
                else:
                    color = darken_color(base_color, 0.2)
                linestyle = base_styles.get(base_name, "-")
            plt.errorbar(
                n_values,
                mean,
                yerr=ci,
                capsize=4,
                label=method,
                color=color,
                linestyle=linestyle,
                marker="o",
            )
        plt.xlabel("n_numbers")
        plt.ylabel("Kendall tau (higher is better)")
        plt.title(f"Benchmark across methods and n_numbers ({len(seeds)} seeds, range {range_label})")
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(args.out_dir, "benchmark_kendall_tau.png")
        plt.savefig(plot_path, dpi=200)
        plt.close()
        logger.info("Saved plot to %s", plot_path)
    else:
        for method in methods:
            plt.figure(figsize=(7, 4))
            for _, _, label, _ in range_meta:
                mean, ci = mean_and_ci(data[(method, label)])
                plt.errorbar(
                    n_values,
                    mean,
                    yerr=ci,
                    fmt="-o",
                    capsize=4,
                    label=label,
                )
            plt.xlabel("n_numbers")
            plt.ylabel("Kendall tau (higher is better)")
            plt.title(f"{method} across value ranges ({len(seeds)} seeds)")
            plt.legend()
            plt.tight_layout()
            safe_method = method.replace("/", "_")
            plot_path = os.path.join(args.out_dir, f"benchmark_kendall_tau_{safe_method}_ranges.png")
            plt.savefig(plot_path, dpi=200)
            plt.close()
            logger.info("Saved plot to %s", plot_path)


# --- Main ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Benchmark methods vs Kendall tau.")
    parser.add_argument("--seed", default=None, type=int, help="single seed to run (overrides --seeds)")
    parser.add_argument("--seeds", default="2", type=str, help="Comma-separated seeds.")
    parser.add_argument(
        "--n_values",
        default="5,10,50,100",
        type=str,
        help="Comma-separated n_numbers or start:end:steps for linear spacing.",
    )
    parser.add_argument("--n_train_lists", default=10000, type=int)
    parser.add_argument("--n_test_lists", default=100, type=int)
    parser.add_argument(
        "--value_ranges",
        default="0,1;10,11",
        type=str,
        help="Semicolon-separated ranges, each as min,max or min:max.",
    )
    parser.add_argument("--tau", default=1.5, type=float)
    parser.add_argument(
        "--diffsort_steepness",
        default=10.0,
        type=float,
        help="DiffSort steepness (higher -> sharper swaps).",
    )
    parser.add_argument(
        "--tau_anneal_end",
        default=0.5,
        type=float,
        help="final tau for annealing in unsupervised runs",
    )
    parser.add_argument(
        "--tau_anneal_schedule",
        default="linear",
        choices=["linear", "exp"],
        help="annealing schedule for tau in unsupervised runs",
    )
    parser.add_argument(
        "--tau_anneal_epochs",
        default=150,
        type=int,
        help="epochs to anneal tau over (0 uses total epochs)",
    )
    parser.add_argument("--tau_adapt", action="store_true", help="enable entropy-based adaptive tau")
    parser.add_argument(
        "--tau_adapt_source",
        default="sinkhorn",
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
        default=0.7,
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
    parser.add_argument("--n_sink_iter", default=10, type=int)
    parser.add_argument("--n_samples", default=5, type=int)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--num_workers", default=4, type=int, help="DataLoader worker processes")
    parser.add_argument("--epochs", default=150, type=int)
    parser.add_argument("--eval_every", default=5, type=int, help="run evaluation every N epochs (0 disables)")
    parser.add_argument(
        "--early_stop_patience",
        default=5,
        type=int,
        help="stop after N evals without Kendall tau improvement (0 disables)",
    )
    parser.add_argument(
        "--early_stop_min_delta",
        default=0.001,
        type=float,
        help="minimum Kendall tau improvement to reset early stopping",
    )
    parser.add_argument(
        "--eval_best",
        action="store_true",
        default=True,
        help="report best checkpoint in benchmark",
    )
    parser.add_argument("--hid_c", default=64, type=int, help="hidden channels / transformer d_model")
    parser.add_argument("--transformer_lr", default=5e-4, type=float, help="learning rate for transformer runs")
    parser.add_argument("--transformer_epochs", default=400, type=int, help="epochs for transformer runs")
    parser.add_argument("--transformer_n_layers", default=4, type=int, help="number of transformer encoder blocks")
    parser.add_argument("--transformer_n_heads", default=4, type=int, help="number of attention heads per block")
    parser.add_argument("--transformer_ffn_dim", default=128, type=int, help="feedforward width inside transformer blocks")
    parser.add_argument("--transformer_dropout", default=0.0, type=float, help="dropout in transformer blocks")
    parser.add_argument(
        "--transformer_spread_weight",
        default=0.0,
        type=float,
        help="weight for spread regularizer on est_ordered_X (transformer runs)",
    )
    parser.add_argument(
        "--transformer_entropy_weight",
        default=1e-4,
        type=float,
        help="weight for permutation entropy penalty (transformer runs)",
    )
    parser.add_argument(
        "--transformer_entropy_eps",
        default=1e-9,
        type=float,
        help="epsilon for permutation entropy log (transformer runs)",
    )
    parser.add_argument("--out_dir", default="log/", type=str)
    parser.add_argument("--rank_temp", default=0.1, type=float)
    parser.add_argument("--rank_sigma", default=2.0, type=float)
    parser.add_argument("--rank_mask_strength", default=1.0, type=float)
    parser.add_argument(
        "--methods",
        default="all",
        type=str,
        help=(
            "Comma-separated methods/groups to run. Names: supervised_conv, unsupervised_conv, neural_sort, "
            "softsort, fast_soft_sort, diffsort. Groups: sinkhorn, baselines. Model tags: conv, transformer. "
            "Use 'all' for everything."
        ),
    )
    parser.add_argument("--save_model", action="store_true", help="save final/best checkpoints")
    parser.add_argument("--save_log_alpha", action="store_true", help="save log_alpha_std.csv during training")
    args = parser.parse_args()
    benchmark(args)


if __name__ == "__main__":
    main()
