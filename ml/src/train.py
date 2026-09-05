import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from dataloader import create_dataloader
from model import CNNLSTM


MODEL_DIR = Path("ml/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def calculate_metrics(y_true, y_pred):
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)

    tp = ((y_true == 1) & (y_pred == 1)).sum().item()
    tn = ((y_true == 0) & (y_pred == 0)).sum().item()
    fp = ((y_true == 0) & (y_pred == 1)).sum().item()
    fn = ((y_true == 1) & (y_pred == 0)).sum().item()

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    max_batches=None,
):
    model.train()

    total_loss = 0.0
    all_true = []
    all_pred = []

    start = time.time()

    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at batch {batch_idx}")

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        predictions = torch.argmax(logits, dim=1)

        total_loss += loss.item() * x.size(0)

        all_true.extend(y.detach().cpu().tolist())
        all_pred.extend(predictions.detach().cpu().tolist())

    samples = len(all_true)

    if samples == 0:
        raise RuntimeError("No training samples were processed.")

    metrics = calculate_metrics(all_true, all_pred)
    metrics["loss"] = total_loss / samples
    metrics["time"] = time.time() - start
    metrics["samples"] = samples

    return metrics


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    max_batches=None,
):
    model.eval()

    total_loss = 0.0
    all_true = []
    all_pred = []

    start = time.time()

    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        predictions = torch.argmax(logits, dim=1)

        total_loss += loss.item() * x.size(0)

        all_true.extend(y.cpu().tolist())
        all_pred.extend(predictions.cpu().tolist())

    samples = len(all_true)

    if samples == 0:
        raise RuntimeError("No validation samples were processed.")

    metrics = calculate_metrics(all_true, all_pred)
    metrics["loss"] = total_loss / samples
    metrics["time"] = time.time() - start
    metrics["samples"] = samples

    return metrics


def print_metrics(name, metrics):
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Loss:      {metrics['loss']:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(
        f"TP: {metrics['tp']} | "
        f"TN: {metrics['tn']} | "
        f"FP: {metrics['fp']} | "
        f"FN: {metrics['fn']}"
    )
    print(f"Samples:   {metrics['samples']}")
    print(f"Time:      {metrics['time']:.2f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Train CNN-LSTM on SisFall."
    )

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=0.001)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("SisFall CNN-LSTM Training")
    print("=========================")
    print(f"Device:       {device}")
    print(f"Batch size:   {args.batch_size}")
    print(f"Epochs:       {args.epochs}")
    print(f"Learning rate:{args.learning_rate}")

    train_loader = create_dataloader(
        "train",
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        cache_size=8,
    )

    val_loader = create_dataloader(
        "val",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        cache_size=8,
    )

    print()
    print(f"Training samples:   {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    model = CNNLSTM().to(device)

    # Class weights based on the training-window counts.
    normal_count = 30801
    fall_count = 17734
    total = normal_count + fall_count

    normal_weight = total / (2 * normal_count)
    fall_weight = total / (2 * fall_count)

    class_weights = torch.tensor(
        [normal_weight, fall_weight],
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    print(f"Model parameters:  {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Class weights:     normal={normal_weight:.4f}, fall={fall_weight:.4f}")

    best_val_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        print()
        print(f"========== Epoch {epoch}/{args.epochs} ==========")

        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            max_batches=args.max_train_batches,
        )

        print_metrics("Training", train_metrics)

        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            max_batches=args.max_val_batches,
        )

        print_metrics("Validation", val_metrics)

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_name": "CNNLSTM",
                "input_channels": 6,
                "num_classes": 2,
                "window_size": 400,
                "best_val_f1": best_val_f1,
                "epoch": epoch,
            }

            checkpoint_path = MODEL_DIR / "cnn_lstm_best.pt"
            torch.save(checkpoint, checkpoint_path)

            print()
            print(f"Saved best checkpoint: {checkpoint_path}")
            print(f"Best validation F1:   {best_val_f1:.4f}")

    print()
    print("=========================")
    print("Training complete")
    print("=========================")


if __name__ == "__main__":
    main()
