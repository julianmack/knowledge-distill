import os
from pathlib import Path 

import torch

def save_checkpoint(model, epoch, log_dir):
    name = model.__class__.__name__
    log_dir = Path(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    checkpoint = {"model": model.state_dict()}
    torch.save(checkpoint, log_dir / f"{name}_{epoch}.pt")

def get_device(model):
    device = "cpu"
    if len(model.state_dict().values()) > 0:
        for value in model.state_dict().values():
            if isinstance(value, torch.Tensor):
                return value.device
    return device
