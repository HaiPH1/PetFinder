import torch
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error


def train_fn(
    dataloader, model, criterion, optimizer, scheduler=None, device="cuda", accu_step=1
):
    model.train()
    loss_score = 0
    pbar = tqdm(dataloader, total=len(dataloader))
    y_pre, y_gold = [], []
    for i, (batch) in enumerate(pbar):
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        loss_score += loss.item()

        y_pre.extend(list(logits.detach().cpu().numpy()))
        y_gold.extend(list(labels.detach().cpu().numpy()))

        loss.backward()
        if (i + 1) % accu_step == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            model.zero_grad()
            optimizer.zero_grad()

    return loss_score / len(dataloader), mean_squared_error(y_gold, y_pre)


def validation_fn(dataloader, model, criterion, device="cuda"):
    model.eval()
    with torch.no_grad():
        loss_score = 0
        pbar = tqdm(dataloader, total=len(dataloader))
        y_pre, y_gold = [], []
        for batch in pbar:
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)

            logits = model(images)
            y_pre.extend(list(logits.detach().cpu().numpy()))
            y_gold.extend(list(labels.detach().cpu().numpy()))

            loss = criterion(logits, labels)
            loss_score += loss.item()

        return loss_score / len(dataloader), mean_squared_error(y_gold, y_pre)
