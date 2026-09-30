import argparse
import os
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn

from src.dataset import create_dataloader
from src.engine import HTRTrainer, build_scheduler
from src.metrics import MetricAccumulator
from src.vocab import VietnameseVocab
from models.crnn import CRNN


def encode_ctc_targets(
    raw_texts,
    vocab: VietnameseVocab,
    device: torch.device,
):
    """
    CTC KHÔNG dùng <sos>/<eos>.
    Encode trực tiếp từng ký tự rồi nối các target lại.
    """
    encoded = [
        torch.tensor(
            vocab.encode(text, add_special_tokens=False),
            dtype=torch.long,
            device=device,
        )
        for text in raw_texts
    ]

    target_lengths = torch.tensor(
        [len(x) for x in encoded],
        dtype=torch.long,
        device=device,
    )

    targets = torch.cat(encoded, dim=0)

    return targets, target_lengths


def make_ctc_train_step(vocab, criterion):
    def train_step(model, batch, device):
        images = batch["images"]
        raw_texts = batch["raw_texts"]

        logits = model(images)  # [B, T, C]
        log_probs = logits.log_softmax(dim=-1).permute(1, 0, 2)

        targets, target_lengths = encode_ctc_targets(
            raw_texts,
            vocab,
            device,
        )

        input_lengths = torch.full(
            (images.size(0),),
            logits.size(1),
            dtype=torch.long,
            device=device,
        )

        loss = criterion(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
        )

        return loss

    return train_step


def ctc_greedy_decode(logits, vocab):
    """
    CTC Greedy:
        argmax -> collapse repeated tokens -> remove blank.
    """
    pred_ids = logits.argmax(dim=-1)
    results = []

    special_to_remove = {
        vocab.pad_id,
        vocab.sos_id,
        vocab.eos_id,
    }

    for seq in pred_ids:
        decoded = []
        previous = vocab.pad_id

        for token_id in seq.tolist():
            if token_id == vocab.pad_id:
                previous = token_id
                continue

            if token_id == previous:
                continue

            if token_id not in special_to_remove:
                decoded.append(token_id)

            previous = token_id

        results.append(
            vocab.decode(
                decoded,
                remove_special_tokens=True,
            )
        )

    return results


def make_ctc_eval_step(vocab, criterion):
    def eval_step(model, batch, device):
        images = batch["images"]
        raw_texts = list(batch["raw_texts"])

        logits = model(images)
        log_probs = logits.log_softmax(dim=-1).permute(1, 0, 2)

        targets, target_lengths = encode_ctc_targets(
            raw_texts,
            vocab,
            device,
        )

        input_lengths = torch.full(
            (images.size(0),),
            logits.size(1),
            dtype=torch.long,
            device=device,
        )

        loss = criterion(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
        )

        predictions = ctc_greedy_decode(
            logits,
            vocab,
        )

        return loss, predictions

    return eval_step


@torch.no_grad()
def evaluate_test(model, test_loader, eval_step_fn, device):
    model.eval()

    metrics = MetricAccumulator()
    total_loss = 0.0
    batches = 0

    for batch in test_loader:
        raw_texts = list(batch["raw_texts"])

        moved = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                moved[key] = value.to(
                    device,
                    non_blocking=True,
                )
            else:
                moved[key] = value

        loss, predictions = eval_step_fn(
            model,
            moved,
            device,
        )

        metrics.update(
            raw_texts,
            predictions,
        )

        total_loss += float(loss.item())
        batches += 1

    cer_value, wer_value = metrics.compute(
        as_percent=True,
    )

    return {
        "test_loss": total_loss / max(batches, 1),
        "test_CER": cer_value,
        "test_WER": wer_value,
    }


def build_loaders(args, vocab):
    from src.augmentation import HTRAugmenter

    augmenter = HTRAugmenter(
        prob=args.augmentation_prob,
    )

    train_loader = create_dataloader(
        annotation_file=args.train_annotations,
        vocab=vocab,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        augmentation=augmenter,
    )

    val_loader = create_dataloader(
        annotation_file=args.val_annotations,
        vocab=vocab,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        augmentation=None,
    )

    test_loader = create_dataloader(
        annotation_file=args.test_annotations,
        vocab=vocab,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        augmentation=None,
    )

    return train_loader, val_loader, test_loader


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_annotations", type=str, required=True)
    parser.add_argument("--val_annotations", type=str, required=True)
    parser.add_argument("--test_annotations", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--augmentation_prob", type=float, default=0.4)

    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/crnn")
    parser.add_argument("--log_dir", type=str, default="runs/crnn")

    parser.add_argument(
        "--scheduler",
        type=str,
        default="plateau",
        choices=["plateau", "cosine", "none"],
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    vocab = VietnameseVocab()

    train_loader, val_loader, test_loader = build_loaders(
        args,
        vocab,
    )

    model = CRNN(
        num_classes=len(vocab),
        hidden_size=256,
        lstm_layers=2,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = None
    if args.scheduler != "none":
        scheduler = build_scheduler(
            optimizer,
            args.scheduler,
            args.epochs,
        )

    criterion = nn.CTCLoss(
        blank=vocab.pad_id,
        zero_infinity=True,
    )

    train_step_fn = make_ctc_train_step(
        vocab,
        criterion,
    )

    eval_step_fn = make_ctc_eval_step(
        vocab,
        criterion,
    )

    trainer = HTRTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        checkpoint_dir=args.checkpoint_dir,
        monitor="val_CER",
        monitor_mode="min",
        early_stopping_patience=10,
        grad_clip_norm=5.0,
        log_dir=args.log_dir,
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        train_step_fn=train_step_fn,
        eval_step_fn=eval_step_fn,
        epochs=args.epochs,
    )

    best_path = trainer.best_checkpoint_path
    print(f"\nBest checkpoint: {best_path}")

    if os.path.exists(best_path):
        trainer.load_checkpoint(best_path)

    test_metrics = evaluate_test(
        model,
        test_loader,
        eval_step_fn,
        device,
    )

    print("\n===== TEST RESULT =====")
    print(f"Test Loss : {test_metrics['test_loss']:.4f}")
    print(f"Test CER  : {test_metrics['test_CER']:.2f}%")
    print(f"Test WER  : {test_metrics['test_WER']:.2f}%")

    return history, test_metrics


if __name__ == "__main__":
    main()
