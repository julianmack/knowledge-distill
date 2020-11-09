import argparse
from pathlib import Path
import os
import time

import torch
from torch.nn.utils.rnn import pad_sequence

from distill.student import ConvClassifier
from distill.glove import GloveTokenizer
from distill.data import get_train_valid_test_loaders
from distill.utils import save_checkpoint, get_device
from distill.labels import probs_to_labels, all_labels
from distill.evaluate import print_eval_res, evaluate


def train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_csv',
        type=str,
        default='data/fin_news_all-data.csv',
        help='Data CSV'
    )
    parser.add_argument(
        '--log_dir_prefix',
        type=str,
        default='logs/',
        help='Data CSV'
    )
    parser.add_argument(
        '--expt_name',
        type=str,
        default=None,
        help='name of experiment'
    )
    parser.add_argument(
        '--glove_fp',
        type=str,
        default='model/glove.6B/glove.6B.200d.txt',
        help='GloVe fp'
    )
    parser.add_argument(
        '--glove_dim',
        type=int,
        default=200,
        help='GloVe hidden dimension size'
    )
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)

    args = parser.parse_args()
    expt_name = args.expt_name or time.strftime("%Y_%m_%d_%H_%M_%S")
    if os.path.isdir(expt_name) and os.listdir(expt_name):
        raise ValueError(f'directory={expt_name} already exists')

    args.log_dir = Path(args.log_dir_prefix) / expt_name

    return args

def train_init(args):
    train_loader, valid_loader, test_loader = get_train_valid_test_loaders(
        csv_file=args.input_csv,
        headers=['label', 'text'],
        batch_size=args.batch_size,
    )
    tokenizer = GloveTokenizer(glove_fp=args.glove_fp)
    assert tokenizer.model.dim == args.glove_dim
    model = ConvClassifier(glove_dim=args.glove_dim)

    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=0.001,
    )

    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    return {
        'train_loader': train_loader,
        'valid_loader': valid_loader,
        'test_loader': test_loader,
        'glove_tokenizer': tokenizer,
        'model': model,
        'optimizer': optimizer,
        'criterion': criterion,
        'epochs': args.epochs,
        'log_dir': args.log_dir,
        'unpack_kwargs': {'glove_tokenizer': tokenizer},
        'eval_every': 10,
    }


def unpack_batch_send_to_device(batch, device, glove_tokenizer):
    """Unpacks batch + performs tokenization + label/text batch collation."""
    (texts, labels) = batch

    texts = [glove_tokenizer(text)[0] for text in texts]
    labels = [torch.LongTensor([label]) for label in labels]
    lengths = torch.IntTensor([len(x) for x in texts])

    texts = pad_sequence(texts, batch_first=True)
    labels = torch.cat(labels, dim=0)

    B, T, H = texts.shape
    B2, = labels.shape

    assert B == B2 == len(texts)
    assert H == glove_tokenizer.model.dim

    x = texts
    y = labels
    x_len = lengths

    x, y, x_len = x.to(device), y.to(device), x_len.to(device)
    return x, y, x_len

def train(
    model,
    train_loader,
    criterion,
    optimizer,
    epochs,
    log_dir,
    valid_loader=None,
    eval_every=1,
    unpack_kwargs={},
    verbose=True,
    **kwargs,
):
    log_dir = Path(log_dir)

    iteration = 0
    train_losses = []
    train_eval_res = []
    valid_eval_res = []
    for epoch in range(1, epochs + 1):
        print(f"Starting {epoch=}", end=", ")
        if epoch == 0:
            save_checkpoint(model, epoch, log_dir)
        train_loss, iteration = train_epoch(
            model, train_loader, criterion, optimizer, iteration, unpack_kwargs
        )
        train_losses.append(train_loss)

        eval_this_epoch = (epoch % eval_every == 0) or epoch == epochs

        if not eval_this_epoch:
            continue
        # save every epoch when evaluation is performed
        save_checkpoint(model, epoch, log_dir)
        print()
        res = evaluate(
            model=model,
            loader=train_loader,
            subset="train",
            iteration=iteration,
            unpack_kwargs=unpack_kwargs,
            unpack_batch_fn=unpack_batch_send_to_device,
            all_labels=all_labels,
            probs_to_labels=probs_to_labels,
        )
        train_eval_res.append(res)
        if verbose:
            print_eval_res(train_eval_res[-1])
        if not valid_loader:
            continue
        res = evaluate(
            model=model,
            loader=valid_loader,
            subset="valid",
            iteration=iteration,
            unpack_kwargs=unpack_kwargs,
            unpack_batch_fn=unpack_batch_send_to_device,
            all_labels=all_labels,
            probs_to_labels=probs_to_labels,
        )
        valid_eval_res.append(res)
        if verbose:
            print_eval_res(valid_eval_res[-1])

    return train_losses, train_eval_res, valid_eval_res


def train_epoch(model, loader, criterion, optimizer, iteration, unpack_kwargs):
    """Train one epoch of model."""
    device = get_device(model)
    model.train()

    count = 0
    train_loss = 0.
    for batch in loader:
        iteration += 1
        x, y, x_len = unpack_batch_send_to_device(
            batch,
            device,
            **unpack_kwargs,
        )
        y_hat = model(x, x_len)
        optimizer.zero_grad()
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        count += y.size(0)
    return train_loss / count, iteration

if __name__ == '__main__':
    args = train_args()
    train_args = train_init(args)
    train(**train_args)
