"""
Training / Validation Engine dùng chung cho HTR.
Có thể dùng cho:
- CRNN + CTC
- DenseNet + LSTM Attention
- DenseNet + Transformer về sau
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None
from src.metrics import MetricAccumulator

# =========================================================
# Type Definitions
# =========================================================
TrainStepFn = Callable[
    [torch.nn.Module, Dict[str, Any], torch.device],
    torch.Tensor
]
EvalStepFn = Callable[
    [torch.nn.Module, Dict[str, Any], torch.device],
    Tuple[torch.Tensor, list[str]]
]

# =========================================================
# Early Stopping
# =========================================================
@dataclass
class EarlyStopping:
    patience: int = 10
    mode: str = "min"
    min_delta: float = 0.0
    best: Optional[float] = None
    num_bad_epochs: int = 0
    def step(self, value: float) -> bool:
        if self.best is None:
            improved = True
        elif self.mode == "min":
            improved = value < self.best - self.min_delta
        elif self.mode == "max":
            improved = value > self.best + self.min_delta
        else:
            raise ValueError("mode phải là 'min' hoặc 'max'.")
        if improved:
            self.best = value
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        return self.num_bad_epochs >= self.patience

# =========================================================
# Scheduler
# =========================================================
def build_scheduler(
    optimizer: Optimizer,
    scheduler_name: Optional[str],
    max_epochs: int
):
    if scheduler_name is None:
        return None
    name = scheduler_name.lower()
    if name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=max_epochs
        )
    if name in {"plateau", "reducelronplateau"}:
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2
        )
    raise ValueError(
        "scheduler_name phải là None, 'cosine' hoặc 'plateau'."
    )

# =========================================================
# Move Batch
# =========================================================
def _move_batch(
    batch: Dict[str, Any],
    device: torch.device
) -> Dict[str, Any]:
    moved = dict(batch)
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(
                device,
                non_blocking=True
            )
    return moved

# =========================================================
# HTR Trainer
# =========================================================
class HTRTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        device: Optional[torch.device] = None,
        scheduler=None,
        checkpoint_dir: str = "checkpoints",
        monitor: str = "val_CER",
        monitor_mode: str = "min",
        early_stopping_patience: int = 10,
        early_stopping_delta: float = 0.0,
        grad_clip_norm: Optional[float] = 5.0,
        log_dir: Optional[str] = "runs/htr"
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.scheduler = scheduler
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.grad_clip_norm = grad_clip_norm
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(
            self.checkpoint_dir,
            exist_ok=True
        )
        self.best_checkpoint_path = os.path.join(
            self.checkpoint_dir,
            "best_model.pth"
        )
        # TensorBoard
        if log_dir and SummaryWriter is not None:
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            mode=monitor_mode,
            min_delta=early_stopping_delta
        )
        self.history = []
        self.start_epoch = 1
        self.model.to(self.device)

    # =====================================================
    # Train One Epoch
    # =====================================================
    def train_one_epoch(
        self,
        train_loader,
        train_step_fn: TrainStepFn
    ) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        for batch in train_loader:
            batch = _move_batch(
                batch,
                self.device
            )
            self.optimizer.zero_grad(
                set_to_none=True
            )
            loss = train_step_fn(
                self.model,
                batch,
                self.device
            )
            if not torch.is_tensor(loss) or loss.ndim != 0:
                raise ValueError(
                    "train_step_fn phải trả về scalar torch.Tensor loss."
                )
            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    f"Loss không hữu hạn: {loss.item()}"
                )
            # Backpropagation
            loss.backward()
            # Gradient Clipping
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip_norm
                )
            self.optimizer.step()
            total_loss += float(
                loss.detach().item()
            )
            num_batches += 1
        if num_batches == 0:
            raise ValueError(
                "train_loader không có batch nào."
            )
        return total_loss / num_batches

    # =====================================================
    # Validation
    # =====================================================
    @torch.no_grad()
    def validate(
        self,
        val_loader,
        eval_step_fn: EvalStepFn
    ) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        metrics = MetricAccumulator()
        for batch in val_loader:
            raw_targets = list(
                batch.get("raw_texts", [])
            )
            batch = _move_batch(
                batch,
                self.device
            )
            loss, predictions = eval_step_fn(
                self.model,
                batch,
                self.device
            )
            if not torch.is_tensor(loss) or loss.ndim != 0:
                raise ValueError(
                    "eval_step_fn phải trả về scalar loss."
                )
            predictions = list(predictions)
            if len(predictions) != len(raw_targets):
                raise ValueError(
                    "Số prediction phải bằng số raw_texts."
                )
            metrics.update(
                raw_targets,
                predictions
            )
            total_loss += float(
                loss.detach().item()
            )
            num_batches += 1
        if num_batches == 0:
            raise ValueError(
                "val_loader không có batch nào."
            )
        val_cer, val_wer = metrics.compute(
            as_percent=True
        )
        return {
            "val_loss": total_loss / num_batches,
            "val_CER": val_cer,
            "val_WER": val_wer
        }

    # =====================================================
    # Save Checkpoint
    # =====================================================
    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        path: Optional[str] = None
    ) -> str:
        path = path or self.best_checkpoint_path
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict()
                if self.scheduler is not None
                else None
            ),
            "metrics": metrics,
            "best_monitor": self.early_stopping.best,
            "history": self.history
        }
        torch.save(state, path)
        return path

    # =====================================================
    # Load Checkpoint
    # =====================================================
    def load_checkpoint(
        self,
        path: str,
        map_location=None
    ):
        checkpoint = torch.load(
            path,
            map_location=map_location or self.device
        )
        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        if checkpoint.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )
        if (
            self.scheduler is not None
            and checkpoint.get("scheduler_state_dict")
        ):
            self.scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )
        self.history = checkpoint.get(
            "history",
            []
        )
        self.start_epoch = (
            int(checkpoint.get("epoch", 0)) + 1
        )
        self.early_stopping.best = checkpoint.get(
            "best_monitor"
        )
        return checkpoint

    # =====================================================
    # Fit
    # =====================================================
    def fit(
        self,
        train_loader,
        val_loader,
        train_step_fn: TrainStepFn,
        eval_step_fn: EvalStepFn,
        epochs: int,
        resume_path: Optional[str] = None
    ):
        if resume_path:
            self.load_checkpoint(resume_path)
        for epoch in range(
            self.start_epoch,
            epochs + 1
        ):
            # Train
            train_loss = self.train_one_epoch(
                train_loader,
                train_step_fn
            )
            # Validation
            val_metrics = self.validate(
                val_loader,
                eval_step_fn
            )
            metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                **val_metrics
            }
            current = float(
                metrics[self.monitor]
            )
            self.history.append(metrics)
            # TensorBoard
            if self.writer:
                for key, value in metrics.items():
                    if key != "epoch":
                        self.writer.add_scalar(
                            key,
                            value,
                            epoch
                        )
                self.writer.add_scalar(
                    "learning_rate",
                    self.optimizer.param_groups[0]["lr"],
                    epoch
                )
            # Scheduler
            if isinstance(
                self.scheduler,
                ReduceLROnPlateau
            ):
                self.scheduler.step(
                    val_metrics["val_CER"]
                )
            elif self.scheduler is not None:
                self.scheduler.step()
            # Kiểm tra cải thiện
            previous_best = self.early_stopping.best
            should_stop = self.early_stopping.step(
                current
            )
            improved = (
                previous_best is None
                or (
                    self.monitor_mode == "min"
                    and current < previous_best - self.early_stopping.min_delta
                )
                or (
                    self.monitor_mode == "max"
                    and current > previous_best + self.early_stopping.min_delta
                )
            )
            # Lưu best model
            if improved:
                self.save_checkpoint(
                    epoch,
                    metrics
                )
            # Hiển thị kết quả
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_metrics['val_loss']:.4f} | "
                f"CER={val_metrics['val_CER']:.2f}% | "
                f"WER={val_metrics['val_WER']:.2f}%"
            )
            # Early Stopping
            if should_stop:
                print(
                    f"Early stopping tại epoch {epoch}: "
                    f"{self.monitor} không cải thiện trong "
                    f"{self.early_stopping.patience} epoch."
                )
                break
        # Lưu history
        history_path = os.path.join(
            self.checkpoint_dir,
            "training_history.json"
        )
        with open(
            history_path,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                self.history,
                f,
                ensure_ascii=False,
                indent=2
            )
        if self.writer:
            self.writer.close()
        return self.history

__all__ = [
    "EarlyStopping",
    "build_scheduler",
    "HTRTrainer"
]