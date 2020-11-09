import argparse
import time
import os
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


from distill.student import ConvClassifier
from distill.teacher import TeacherNLPClassifier
from distill.glove import GloveTokenizer
from distill.data import get_train_valid_test_loaders
from distill.train.utils import train, train_epoch
from distill.train.train_student import unpack_batch_send_to_device
from distill.data import CSVTextDataset

def train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_csv',
        type=str,
        default='data/headlines.csv',
        help='Data CSV'
    )
    parser.add_argument(
        '--eval_csv',
        type=str,
        default='data/fin_news_all-data.csv',
        help='Data CSV'
    )
    parser.add_argument(
        '--log_dir_prefix',
        type=str,
        default='logs/distill',
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
    parser.add_argument(
        '--teacher_model_dir',
        type=str,
        default='./model',
        help='Teacher model directory'
    )
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)

    args = parser.parse_args()
    expt_name = args.expt_name or time.strftime("%Y_%m_%d_%H_%M_%S")
    if os.path.isdir(expt_name) and os.listdir(expt_name):
        raise ValueError(f'directory={expt_name} already exists')

    args.log_dir = Path(args.log_dir_prefix) / expt_name

    return args

def train_init(args):
    _, valid_loader, test_loader = get_train_valid_test_loaders(
        csv_file=args.eval_csv,
        headers=['label', 'text'],
        batch_size=args.batch_size,
    )
    train_dataset = CSVTextDataset(args.input_csv, headers=['text'])
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=train_dataset.collate_batch
    )
    tokenizer = GloveTokenizer(glove_fp=args.glove_fp)
    assert tokenizer.model.dim == args.glove_dim
    model = ConvClassifier(glove_dim=args.glove_dim)

    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=0.001,
    )

    # instead of cross entropy which requires hard labels, use BCE across
    # all logits
    criterion = torch.nn.BCELoss(reduction="sum")
    teacher = TeacherNLPClassifier(args.teacher_model_dir)
    if torch.cuda.is_available():
        teacher = teacher.cuda()
    return {
        'train_loader': train_loader,
        'valid_loader': valid_loader,
        'test_loader': test_loader,
        'glove_tokenizer': tokenizer,
        'model': model,
        'optimizer': optimizer,
        'criterion': criterion,
        'log_dir': args.log_dir,
        'unpack_batch_function': unpack_batch_and_gen_teacher_samples,
        'unpack_kwargs': {
            'glove_tokenizer': tokenizer,
            'teacher': teacher,
        },
        'eval_every': 1,
        'eval_train': False,
        'epochs': args.epochs,
        'eval_unpack_batch_fn': unpack_batch_send_to_device,
        'eval_unpack_kwargs': {
            'glove_tokenizer': tokenizer,
        },
        'train_eval_iterations': 200,
    }


def unpack_batch_and_gen_teacher_samples(
    batch,
    device,
    glove_tokenizer,
    teacher,
):
    """Unpacks batch + produces teacher labels."""
    texts = batch

    labels = teacher(texts)
    texts = [glove_tokenizer(text)[0] for text in texts]
    lengths = torch.IntTensor([len(x) for x in texts])

    texts = pad_sequence(texts, batch_first=True)

    B, T, H = texts.shape
    B2, C = labels.shape

    assert B == B2 == len(texts)
    assert H == glove_tokenizer.model.dim
    assert C == 3

    x = texts
    y = labels
    x_len = lengths

    x, y, x_len = x.to(device), y.to(device), x_len.to(device)
    return x, y, x_len

if __name__ == '__main__':
    args = train_args()
    train_args = train_init(args)
    train(**train_args)
