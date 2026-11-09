import random
from copy import deepcopy
from typing import Any, Dict, Tuple
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchmetrics.classification import MulticlassAccuracy

from models import build_model
from utils.utils import build_optimizer
from .base import UnlearningStrategy


class _UnlearningData(Dataset):
    """Union of forget and retain samples, labeled 1 and 0 for teacher routing."""

    def __init__(self, forget_data: Dataset, retain_data: Dataset) -> None:
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.forget_len = len(forget_data)
        self.retain_len = len(retain_data)

    def __len__(self) -> int:
        return self.forget_len + self.retain_len

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if index < self.forget_len:
            return self.forget_data[index][0], 1
        return self.retain_data[index - self.forget_len][0], 0


class _BadTeacherModule(pl.LightningModule):
    """Distill forget samples from a random teacher and retain samples from the base teacher."""

    def __init__(
        self,
        model: pl.LightningModule,
        competent_teacher: pl.LightningModule,
        incompetent_teacher: pl.LightningModule,
        cfg: Dict[str, Any],
        retain_count: int,
        forget_count: int,
    ) -> None:
        """Initialize the student and two frozen teachers."""
        super().__init__()
        self.model = model
        self.competent_teacher = competent_teacher
        self.incompetent_teacher = incompetent_teacher
        self.cfg = cfg
        self.temperature = float(cfg.temperature)
        self.use_weighted_loss = bool(cfg.use_weighted_loss)
        if self.temperature <= 0.0:
            raise ValueError("Bad Teacher requires temperature > 0.")

        if self.use_weighted_loss:
            retain_weight = float(cfg.retain_weight)
            forget_weight = float(cfg.forget_weight)
            if not 0.0 <= retain_weight <= 1.0 or not 0.0 <= forget_weight <= 1.0:
                raise ValueError("Bad Teacher weights must be in [0, 1].")
            if abs(retain_weight + forget_weight - 1.0) > 1e-6:
                raise ValueError("Bad Teacher weights must sum to 1.")
            if retain_count <= 0 or forget_count <= 0:
                raise ValueError("Weighted Bad Teacher requires both data groups.")
            # Correct for each group's sampling probability in the shuffled union.
            total_count = retain_count + forget_count
            self.retain_scale = retain_weight * total_count / retain_count
            self.forget_scale = forget_weight * total_count / forget_count

        for teacher in (self.competent_teacher, self.incompetent_teacher):
            teacher.requires_grad_(False)
            teacher.eval()

        self.ce_loss = nn.CrossEntropyLoss()
        self.val_acc = MulticlassAccuracy(num_classes=self.model.num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=self.model.num_classes)

    def train(self, mode: bool = True):
        """Keep both teachers in eval mode when Lightning toggles the student."""
        super().train(mode)
        self.competent_teacher.eval()
        self.incompetent_teacher.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the student."""
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Distill the routed teacher predictions on a mixed forget/retain batch."""
        del batch_idx
        x, labels = batch
        with torch.no_grad():
            competent_probs = F.softmax(self.competent_teacher(x) / self.temperature, dim=1)
            incompetent_probs = F.softmax(self.incompetent_teacher(x) / self.temperature, dim=1)
            targets = torch.where(labels.bool().unsqueeze(1), incompetent_probs, competent_probs)

        logits = self.model(x)
        # Preserve the original element-wise KL mean, without temperature-squared scaling.
        per_sample_kl = F.kl_div(
            F.log_softmax(logits / self.temperature, dim=1), targets, reduction="none"
        ).mean(dim=1)
        if self.use_weighted_loss:
            weights = torch.where(
                labels.bool(), per_sample_kl.new_tensor(self.forget_scale), per_sample_kl.new_tensor(self.retain_scale)
            )
            loss = (weights * per_sample_kl).mean()
        else:
            loss = per_sample_kl.mean()

        self.log("train/loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def _shared_eval_step(self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        """Run a shared validation/test step."""
        x, y = batch
        logits = self.model(x)
        loss = self.ce_loss(logits, y)
        metric = {"val": self.val_acc, "test": self.test_acc}[stage]
        metric(logits.argmax(dim=1), y)
        self.log(f"{stage}/acc", metric, on_epoch=True, on_step=False, prog_bar=(stage == "val"))
        self.log(f"{stage}/loss", loss, on_epoch=True, on_step=False, prog_bar=(stage == "val"))
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Run one validation step."""
        del batch_idx
        return self._shared_eval_step(batch, "val")

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Run one test step."""
        del batch_idx
        return self._shared_eval_step(batch, "test")

    def configure_optimizers(self) -> Any:
        """Configure the optimizer and scheduler."""
        return build_optimizer(self.model.parameters(), self.cfg.optim, self.cfg.scheduler)


class BadTeacher(UnlearningStrategy):
    """Bad Teacher / blindspot unlearning from Chundawat et al. (AAAI 2023)."""

    def _build_incompetent_teacher(self) -> pl.LightningModule:
        """Build a randomly initialized teacher, even for pretrained experiments."""
        cfg_copy = deepcopy(self.cfg)
        cfg_copy.model.model.pretrained = False
        return build_model(cfg_copy)

    def _build_unlearning_loader(self) -> DataLoader:
        """Mix forget data with a sampled retain subset using deterministic transforms."""
        fraction = float(self.cfg.unlearn.retain_fraction)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("Bad Teacher retain_fraction must be in [0, 1].")
        retain_data = self.dm.eval_retain
        subset_size = int(fraction * len(retain_data))
        indices = random.sample(range(len(retain_data)), subset_size)
        dataset = _UnlearningData(self.dm.eval_forget, Subset(retain_data, indices))
        if dataset.forget_len == 0 or dataset.retain_len == 0:
            raise ValueError("Bad Teacher requires non-empty forget and selected retain sets.")

        return DataLoader(
            dataset,
            batch_size=self.dm.bs,
            shuffle=True,
            num_workers=self.dm.num_workers,
            pin_memory=self.dm.pin_memory,
            persistent_workers=self.dm.persistent_workers if self.dm.num_workers > 0 else False,
            prefetch_factor=self.dm.prefetch_factor if self.dm.num_workers > 0 else None,
        )

    def run(self, base_model: pl.LightningModule) -> pl.LightningModule:
        """Execute Bad Teacher and return the updated student."""
        competent_teacher = deepcopy(base_model)
        incompetent_teacher = self._build_incompetent_teacher()
        train_loader = self._build_unlearning_loader()
        module = _BadTeacherModule(
            model=base_model,
            competent_teacher=competent_teacher,
            incompetent_teacher=incompetent_teacher,
            cfg=self.cfg.unlearn,
            retain_count=train_loader.dataset.retain_len,
            forget_count=train_loader.dataset.forget_len,
        )
        trainer = self.new_trainer()
        trainer.fit(
            module,
            train_dataloaders=train_loader,
            val_dataloaders=self.dm.val_dataloader(),
        )
        return module.model
