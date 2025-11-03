#!/usr/bin/env python3

# train.py - minimal DeepSpeed-wrapped training on simulated binary data

import argparse
import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import deepspeed
from deepspeed.accelerator import get_accelerator
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import TensorDataset
import sys


def make_synthetic_data(n_samples=200, n_features=20, seed=42):
    torch.manual_seed(seed)
    x = torch.randn(n_samples, n_features)
    linear_sep = x[:, : n_features // 2].sum(dim=1)
    y = (linear_sep > 0).float().unsqueeze(1)
    return x, y

class SimpleNet(nn.Module):
    def __init__(self, n_features=20, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x)

def get_dataloader(x, y, batch_size, shuffle=True):
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="ds_checkpoints")
    return parser.parse_known_args()[0]

def main():

    args = parse_args()
    torch.manual_seed(args.seed)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    deepspeed.init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    get_accelerator().set_device(local_rank)

    x, y = make_synthetic_data(n_samples=2000, n_features=20, seed=args.seed)

    ds = TensorDataset(x, y)
    
    sampler = None
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        sampler = DistributedSampler(ds, shuffle=True)
        shuffle = False
    else:
        shuffle = True

    train_loader = DataLoader(ds, batch_size=4, sampler=sampler, shuffle=shuffle)

    model = SimpleNet(n_features=x.shape[1], hidden=64)

    ds_config = {
        "train_batch_size": 64,
        "gradient_accumulation_steps": 1,
        "fp16": {"enabled": False},
        "zero_optimization": {"stage": 0}
    }

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=ds_config,
        model=model,
        optimizer=optimizer,
        model_parameters=model.parameters(),
        config=ds_config
    )
    device = model_engine.device

    criterion = nn.BCEWithLogitsLoss()

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model_engine.train()
        sampler.set_epoch(epoch)
        epoch_loss = 0.0
        steps = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model_engine(batch_x)
            if logits.dim() == 2 and logits.size(1) == 1:
                pass
            elif logits.dim() == 1:
                logits = logits.unsqueeze(1)
            loss = criterion(logits, batch_y)
            model_engine.backward(loss)
            model_engine.step()
            epoch_loss += loss.item()
            steps += 1
            global_step += 1

        avg_loss = epoch_loss / max(1, steps)
        if local_rank == 0:
            print(f"Epoch {epoch:2d} finished. Avg loss: {avg_loss:.4f}")

        save_dir = os.path.join(args.save_dir, f"epoch{epoch}")
        os.makedirs(save_dir, exist_ok=True)
        model_engine.save_checkpoint(save_dir)

    if local_rank == 0:
        print(f"Training completed.")
        # sys.exit(1)

if __name__ == "__main__":
    main()    
    try:
        dist.barrier()
    except Exception:
        pass
    try:
        dist.destroy_process_group()
    except Exception:
        pass
