import os

import torch
from torch.optim import AdamW
from tqdm.auto import tqdm


def train_one_epoch(model, dataloader, optimizer, device="cpu"):
    """Run one supervised training epoch for a seq2seq model.

    Args:
        model: Torch seq2seq model returning a ``loss`` attribute.
        dataloader: Iterable of batches containing ``input_ids``, ``attention_mask``, and
            ``labels`` tensors.
        optimizer: Optimizer used to update model parameters.
        device (str | torch.device, optional): Device to move each batch onto. Defaults to
            ``"cpu"``.

    Returns:
        float: Mean training loss across all batches in the dataloader.
    """
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def evaluate_one_epoch(model, dataloader, device="cpu"):
    """Run one validation epoch without gradient updates.

    Args:
        model: Torch seq2seq model returning a ``loss`` attribute.
        dataloader: Iterable of validation batches with the same schema as training batches.
        device (str | torch.device, optional): Device to move each batch onto. Defaults to
            ``"cpu"``.

    Returns:
        float: Mean validation loss across all batches.
    """
    model.eval()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Validation", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += outputs.loss.item()
    return total_loss / max(len(dataloader), 1)


def fit_seq2seq_model(
    model,
    train_loader,
    val_loader,
    device="cpu",
    lr=5e-5,
    epochs=2,
    save_dir=None,
):
    """Train a seq2seq model for multiple epochs and optionally save checkpoints.

    Args:
        model: Torch seq2seq model exposing ``save_pretrained``.
        train_loader: Training dataloader yielding tokenized batches.
        val_loader: Validation dataloader yielding tokenized batches.
        device (str | torch.device, optional): Device used for model and batch tensors.
            Defaults to ``"cpu"``.
        lr (float, optional): AdamW learning rate. Defaults to ``5e-5``.
        epochs (int, optional): Number of training epochs to run. Defaults to ``2``.
        save_dir (str | None, optional): Directory where the model is saved after each epoch.
            ``None`` disables checkpoint writing. Defaults to ``None``.

    Returns:
        list[dict[str, int | float]]: Training-history rows containing epoch number, train loss,
            and validation loss for each completed epoch.
    """
    optimizer = AdamW(model.parameters(), lr=lr)
    model.to(device)
    history = []
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device=device)
        val_loss = evaluate_one_epoch(model, val_loader, device=device)
        row = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss}
        history.append(row)
        print(f"Epoch {epoch + 1}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
    return history
