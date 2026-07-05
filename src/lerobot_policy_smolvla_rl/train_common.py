"""Shared training utilities used by all three trainers.

Currently exposes the learning-rate schedule so the RECAP, SnapFlow, and critic
trainers cannot drift apart: linear warmup followed by cosine decay down to a
configurable ``min_lr`` floor.
"""

import math

import torch


def build_warmup_cosine_scheduler(
    optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
    base_lr: float,
    min_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear-warmup + cosine-decay schedule with a ``min_lr`` floor.

    For ``step < warmup_steps`` the LR ramps linearly from 0 to ``base_lr``.
    Afterwards it follows a half-cosine from ``base_lr`` down to ``min_lr`` over
    the remaining ``total_steps - warmup_steps`` steps.

    This is returned as a :class:`~torch.optim.lr_scheduler.LambdaLR`, so the
    multiplicative factor is expressed relative to ``base_lr`` (the optimizer's
    initial LR). ``diffusers.get_scheduler("cosine")`` decays all the way to 0
    and ignores ``min_lr`` — this helper is the reason the critic now honors its
    ``--min_lr`` flag.
    """

    min_factor = min_lr / base_lr if base_lr > 0 else 0.0

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        # Interpolate between 1.0 and min_lr/base_lr
        return min_factor + (1.0 - min_factor) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
