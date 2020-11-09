import argparse
import time
import os
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

from distill.student import ConvClassifier
from distill.glove import GloveTokenizer
from distill.data import get_train_valid_test_loaders
from distill.train.utils import train, train_epoch

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
        'unpack_batch_function': unpack_batch_send_to_device,
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

if __name__ == '__main__':
    args = train_args()
    train_args = train_init(args)
    train(**train_args)
