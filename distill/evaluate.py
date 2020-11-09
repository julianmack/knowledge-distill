"""This file is (mostly) copied from a personal project from a private repo.
"""
from collections import Counter
from collections import OrderedDict
from enum import auto
from enum import Enum

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from distill.utils import get_device

class Metrics(Enum):
    ACCURACY = auto()
    F1 = auto()
    CONFUSION_MATRIX = auto()


METRICS_REQUIRE_LABELS = [Metrics.CONFUSION_MATRIX, Metrics.F1]


@torch.no_grad()
def evaluate(
    model,
    loader,
    subset,
    unpack_batch_fn,
    all_labels,
    probs_to_labels,
    iteration=0,
    unpack_kwargs = {},
    max_iterations=None,
    metrics=list(Metrics),
):
    device = get_device(model)
    model.eval()
    require_labels = any([x in METRICS_REQUIRE_LABELS for x in metrics])
    # init accumulators
    samples_seen = 0
    if Metrics.ACCURACY in metrics:
        num_correct = 0
        num_pos_correct = 0
        num_pos_seen = 0
    if require_labels:
        y_hat_label_all = []
        y_labels_all = []

    # evaluate
    count = 0
    for batch in loader:
        count += 1
        x, y, x_len = unpack_batch_fn(
            batch,
            device,
            **unpack_kwargs,
        )
        y_hat = model(x, x_len)

        # convert to labels
        y_hat_labels = probs_to_labels(y_hat)
        if len(y.shape) == 2:
            y_labels = probs_to_labels(y)
        elif len(y.shape) == 1:
            y_labels = idx_to_labels(y, all_labels)
        # update accumulators
        samples_seen += len(y)
        if Metrics.ACCURACY in metrics:
            num_correct += calc_correct(y_hat_labels, y_labels)
        if require_labels:
            y_hat_label_all.extend(y_hat_labels)
            y_labels_all.extend(y_labels)
        if max_iterations and count > max_iterations:
            break
    # compile results
    results = {}
    if Metrics.ACCURACY in metrics:
        res = {'av': num_correct / samples_seen}
        results[Metrics.ACCURACY] = res
    if Metrics.F1 in metrics:
        f1_all = f1_score(
            y_labels_all,
            y_hat_label_all,
            labels=all_labels,
            average=None,
        )
        res = zip(all_labels, f1_all)
        res = OrderedDict(res)
        res["av"] = sum(res.values()) / len(res)
        res["av_weight"] = f1_score(
            y_labels_all,
            y_hat_label_all,
            average="weighted",
        )
        res["micro"] = f1_score(
            y_labels_all,
            y_hat_label_all,
            average="micro",
        )
        results[Metrics.F1] = res
    if Metrics.CONFUSION_MATRIX in metrics:
        results[Metrics.CONFUSION_MATRIX] = confusion_matrix(
            y_labels_all,
            y_hat_label_all,
            labels=all_labels,
        )
    assert set(metrics) == set(
        results.keys()
    ), f"{set(metrics)=}!={set(results.keys())=}"
    model.train()

    results['all_labels'] = all_labels
    return results

def idx_to_labels(idxs, all_labels):
    return [all_labels[idx] for idx in idxs]

# METRIC calculation
def calc_correct(y_hats, ys, label=None):
    """Takes predicted and true labels and returns number_correct.

    Args:
        y_hats: predicted labels.

        ys: True labels.

        label: Label to calculate correct number of (if None, calculate for
            all labels.)
    """
    assert len(y_hats) == len(ys)
    num_correct = 0
    for i, y in enumerate(ys):
        y_hat = y_hats[i]
        if label is None:
            num_correct += int(y_hat == y)
        else:
            num_correct += int(label == y_hat == y)
    return num_correct


def print_eval_res(results):
    all_labels = results.pop('all_labels')
    metrics = results.keys()
    if len(metrics) == 0:
        print("No results to print")
        return
    print("~" * 81)
    if Metrics.ACCURACY in metrics:
        _print_dict(results, "Accuracy:       \t", Metrics.ACCURACY, True)
    if Metrics.F1 in metrics:
        _print_dict(results, "F1 Scores:      \t", Metrics.F1, False)
    if Metrics.CONFUSION_MATRIX in metrics:
        conf_mat = results[Metrics.CONFUSION_MATRIX]
        labels_str = ",".join(all_labels)
        print(f"Confusion [{labels_str}]\n{conf_mat}")


def _print_dict(results, title, metric, percentage):
    print(title, end="")
    for k, v in results[metric].items():
        if percentage:
            v *= 100
            print(f"{k}={v:.1f}%  ", end="")
        else:
            print(f"{k}={v:.3f}  ", end="")
    print()
