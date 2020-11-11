"""Teacher is not trained in this project but unpack batch function is 
given below."""
import torch as torch

def unpack_batch_send_to_device(batch, device, **kwargs):
    """Unpacks batch."""
    (texts, labels) = batch

    labels = [torch.LongTensor([label]) for label in labels]
    labels = torch.cat(labels, dim=0)
    lengths = None

    x = texts
    y = labels
    x_len = lengths

    x, y, x_len = x, y.to(device), x_len
    return x, y, x_len
