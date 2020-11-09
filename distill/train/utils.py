import os
from pathlib import Path

from distill.evaluate import print_eval_res, evaluate
from distill.labels import probs_to_labels, all_labels
from distill.utils import save_checkpoint, get_device

def train(
    model,
    train_loader,
    criterion,
    optimizer,
    epochs,
    log_dir,
    unpack_batch_function,
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
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            iteration=iteration,
            unpack_kwargs=unpack_kwargs,
            unpack_batch_function=unpack_batch_function
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
            unpack_batch_fn=unpack_batch_function,
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
            unpack_batch_fn=unpack_batch_function,
            all_labels=all_labels,
            probs_to_labels=probs_to_labels,
        )
        valid_eval_res.append(res)
        if verbose:
            print_eval_res(valid_eval_res[-1])

    return train_losses, train_eval_res, valid_eval_res



def train_epoch(model, loader, criterion, optimizer, iteration, unpack_batch_function, unpack_kwargs):
    """Train one epoch of model."""
    device = get_device(model)
    model.train()

    count = 0
    train_loss = 0.
    for batch in loader:
        iteration += 1
        x, y, x_len = unpack_batch_function(
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
