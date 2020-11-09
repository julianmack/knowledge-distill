import argparse

import torch
from torch.nn.utils.rnn import pad_sequence

from distill.student import ConvClassifier
from distill.glove import GloveTokenizer
from distill.data import get_train_valid_test_loaders
from distill.utils import save_checkpoint, get_device


def train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_csv',
        type=str,
        default='data/fin_news_all-data.csv',
        help='Data CSV'
    )
    parser.add_argument(
        '--glove_fp',
        type=str,
        default='model/glove.6B/glove.6B.50d.txt',
        help='GloVe fp'
    )
    parser.add_argument(
        '--glove_dim',
        type=int,
        default=50,
        help='GloVe hidden dimension size'
    )
    parser.add_argument('--batch_size', type=int, default=128)
    return parser.parse_args()

def train_init(args):

    train_loader, valid_loader, test_loader = get_train_valid_test_loaders(
        csv_file=args.input_csv,
        headers=['label', 'text'],
        batch_size=args.batch_size,
    )
    tokenizer = GloveTokenizer(glove_fp=args.glove_fp)
    assert tokenizer.model.dim == args.glove_dim
    model = ConvClassifier(glove_dim=args.glove_dim)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=0.001,
    )

    criterion = torch.nn.BCELoss(reduction="sum")
    return {
        'train_loader': train_loader,
        'valid_loader': valid_loader,
        'test_loader': test_loader,
        'glove_tokenizer': tokenizer,
        'model': model,
        'optimizer': optimizer,
        'criterion': criterion,
    }


def unpack_batch_send_to_device(batch, device, glove_tokenizer):
    """Unpacks batch + performs tokenization + label/text batch collation."""
    batch = (texts, labels)

    texts = [glove_tokenizer(text) for text in texts]
    labels = [torch.FloatTensor(label) for label in labels]
    lengths = torch.IntTensor([len(x) for x in texts])

    texts = pad_sequence(texts, batch_first=True)
    labels = pad_sequence(labels, batch_first=True)

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
    valid_loader=None,
    train_loader_eval=None,
    eval_every=1,
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
            model, train_loader, criterion, optimizer, iteration,
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
            loader=train_loader_eval or train_loader,
            subset="train",
            iteration=iteration,
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
        )
        valid_eval_res.append(res)
        if verbose:
            print_eval_res(valid_eval_res[-1])

    return train_losses, train_eval_res, valid_eval_res


def train_epoch(model, loader, criterion, optimizer, iteration):
    """Train one epoch of model."""
    device = utils.get_device(model)
    model.train()

    count = 0
    correct = 0
    train_loss = 0.
    for batch in loader:
        iteration += 1
        x, y, x_len = unpack_batch_send_to_device(batch, device)
        y_hat = model(x, x_len)
        optimizer.zero_grad()
        loss = criterion(y_hat, y.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        correct += calc_num_correct(y_hat, y)
        count += y.size(0)
        exit_
    return train_loss / count, iteration


@torch.no_grad()
def calc_num_correct(y_hat, y):
    y_hat_cls = (y_hat >= 0.5)
    return (y_hat_cls == y).sum().item()

@torch.no_grad()
def eval_model(model, loader, criterion):
    device = list(model.state_dict().values())[0].device
    model.eval()
    count = 0
    correct = 0
    val_loss = 0.
    for x, x_len, y, _ in loader:
        B,_ = y.shape
        x, y = x.to(device), y.to(device)
        y_hat = model(x, x_len)
        loss = criterion(y_hat, y.float())
        val_loss += loss.item()
        correct += calc_num_correct(y_hat, y)
        count += y.size(0)
    model.train()
    return val_loss, correct / count


if __name__ == '__main__':
    args = train_args()
    train_args = train_init(args)
    train(**train_args)
