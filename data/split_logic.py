import random
from dataclasses import dataclass
from typing import List


@dataclass
class SplitIdx:
    """Index splits for retain/forget/held-out/val."""
    retain: List[int]
    forget: List[int]
    heldout: List[int]
    val: List[int]


def split_random(indices: List[int], seed: int) -> List[int]:
    """Return a shuffled copy of indices."""
    rng = random.Random(seed)
    idx = indices[:]
    rng.shuffle(idx)
    return idx


def stratified_split(
    labels: List[int],
    seed: int,
    val_frac: float,
    forget_frac: float,
    heldout_frac: float,
) -> SplitIdx:
    """Reserve validation per class, then split the remainder into forget/heldout/retain."""
    per_class = {}
    for i, y in enumerate(labels):
        per_class.setdefault(y, []).append(i)
    rng = random.Random(seed)
    for c in per_class:
        rng.shuffle(per_class[c])

    val, retain, forget, held = [], [], [], []
    for _, idxs in per_class.items():
        n = len(idxs)
        n_val = int(round(val_frac * n))
        val += idxs[:n_val]
        rem = idxs[n_val:]
        n_forget = int(round(forget_frac * len(rem)))
        n_held = int(round(heldout_frac * len(rem)))
        forget += rem[:n_forget]
        held += rem[n_forget:n_forget + n_held]
        retain += rem[n_forget + n_held:]
    return SplitIdx(retain=retain, forget=forget, heldout=held, val=val)


def stratified_split_random(
    labels: List[int],
    seed: int,
    val_frac: float,
    forget_frac: float,
    heldout_frac: float,
) -> SplitIdx:
    """Reserve validation, then randomly split the remainder into forget/heldout/retain."""
    all_idx = list(range(len(labels)))
    all_idx = split_random(all_idx, seed)
    n = len(all_idx)
    n_val = int(round(val_frac * n))
    val_idx = all_idx[:n_val]
    rem = all_idx[n_val:]
    n_forget = int(round(forget_frac * len(rem)))
    n_held = int(round(heldout_frac * len(rem)))
    forget = rem[:n_forget]
    held = rem[n_forget:n_forget + n_held]
    retain = rem[n_forget + n_held:]
    return SplitIdx(retain=retain, forget=forget, heldout=held, val=val_idx)


def class_split(
    labels: List[int],
    seed: int,
    val_frac: float,
    heldout_frac: float,
    forget_classes: List[int],
    forget_frac: float,
) -> SplitIdx:
    """Reserve validation/heldout per class, then split training samples into forget/retain."""
    if not 0.0 <= val_frac < 1.0 or not 0.0 <= heldout_frac < 1.0:
        raise ValueError("Class split requires validation and heldout fractions in [0, 1).")
    if not 0.0 < forget_frac <= 1.0:
        raise ValueError("Class split requires forget_frac in (0, 1].")

    per_class = {}
    for i, y in enumerate(labels):
        per_class.setdefault(y, []).append(i)
    target_classes = set(forget_classes)
    if not target_classes or not target_classes.issubset(per_class):
        raise ValueError("Class split requires nonempty forget_classes present in the dataset.")

    rng = random.Random(seed)
    retain, forget, held, val = [], [], [], []
    for cls, idxs in per_class.items():
        idxs = idxs[:]
        rng.shuffle(idxs)

        n_val = int(round(val_frac * len(idxs)))
        val += idxs[:n_val]
        rem = idxs[n_val:]
        n_held = int(round(heldout_frac * len(rem)))
        held += rem[:n_held]
        train_pool = rem[n_held:]

        if cls in target_classes:
            n_forget = int(round(forget_frac * len(train_pool)))
            if n_forget == 0:
                raise ValueError(f"Class split selects no forget samples from class {cls}.")
            forget += train_pool[:n_forget]
            retain += train_pool[n_forget:]
        else:
            retain += train_pool

    return SplitIdx(retain=retain, forget=forget, heldout=held, val=val)
