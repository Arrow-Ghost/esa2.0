import time

import torch
import torch.nn as nn
import torch.optim as optim

from dataloader import create_dataloader
from model import CNNLSTM


def main():
    print("CNN-LSTM Training Smoke Test")
    print("============================")

    device = torch.device("cpu")
    print("Device:", device)

    # Small batch for the smoke test.
    batch_size = 32
    max_batches = 20

    print("Batch size:", batch_size)
    print("Test batches:", max_batches)
    print()

    # ---------------------------------------------------------
    # Data
    # ---------------------------------------------------------
    train_loader = create_dataloader(
        "train",
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        cache_size=8,
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------
    model = CNNLSTM().to(device)

    print("Model parameters:",
          sum(p.numel() for p in model.parameters()
              if p.requires_grad))

    # ---------------------------------------------------------
    # Loss + optimizer
    # ---------------------------------------------------------
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    start_time = time.time()

    for batch_idx, (x, y) in enumerate(train_loader):

        if batch_idx >= max_batches:
            break

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        output = model(x)

        loss = criterion(output, y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * x.size(0)

        predictions = output.argmax(dim=1)

        total_correct += (
            predictions == y
        ).sum().item()

        total_samples += x.size(0)

        if (batch_idx + 1) % 5 == 0:
            print(
                f"Batch {batch_idx + 1:02d}/{max_batches} | "
                f"Loss: {loss.item():.4f}"
            )

    elapsed = time.time() - start_time

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    print()
    print("============================")
    print("Smoke test complete")
    print("============================")
    print(f"Samples processed: {total_samples}")
    print(f"Average loss:      {average_loss:.4f}")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Time:              {elapsed:.2f} seconds")
    print(f"Time/sample:       {elapsed / total_samples:.4f} seconds")

    print()

    if not torch.isfinite(torch.tensor(average_loss)):
        raise RuntimeError("Loss became NaN or infinite.")

    if total_samples == 0:
        raise RuntimeError("No training samples were processed.")

    print("Forward pass:       PASS")
    print("Loss calculation:   PASS")
    print("Backpropagation:    PASS")
    print("Optimizer update:   PASS")
    print("Training loop:      PASS")


if __name__ == "__main__":
    main()
