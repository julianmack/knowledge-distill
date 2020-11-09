import math
from typing import List
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader

SEED = 2 # DO NOT CHANGE

class CSVTextDataset(torch.utils.data.Dataset):
    """Dataset class to load textual data.

    This class **does not have responsibility for performing
    tokenizing/embedding** so `.get` returns raw text instead of tensors. It
    is more efficient (with `n_workers > 1`) to produce tensors in the
    dataset but it also introduces complexities which I've decided to
    sidestep here.

    This class has responsibility for splitting into test/train/valid
    data subsets.

    This dataset is flexible enough to load both provided datasets
    (fin_news_all-data.csv & headlines.csv). It is assumed that the
    provided csvs are small enough to fit in RAM.

    Args:
        csv_file: CSV input file.csv

        headers: Array of csv header names. Must include 'text' and can
            optionally include 'label'.

        subset_start: Optional float in [0, 1) that determines amount of data
            in subset. For example, if `subset_start = 0` and
            `subset_end = 0.7` the the subset will use the first 70% of the
            data in `csv_file`.

        subset_end: Optional float in (0, 1]. See subset_start.
    """

    def __init__(
        self,
        csv_file: str,
        headers: List[str],
        subset_start: Optional[float] = None,
        subset_end: Optional[float] = None,
    ):
        assert 'text' in headers
        if subset_start is None:
            assert subset_end is None
        else:
            assert subset_end is not None
            assert 0 <= subset_start < 1
            assert 0 < subset_end <= 1
            assert subset_start < subset_end
        super().__init__()
        self.headers = headers
        self.subset_start = subset_start
        self.subset_end = subset_end
        try:
            data = pd.read_csv(
                csv_file, names=self.headers, quotechar='"'
            )
            trim_quote = False
        except UnicodeEncodeError:
            # for reasons I haven't investigated much, pandas doesn't
            # excape the commas within quotes "" for the fin_news_all-data
            # file. As there are just two columns can hackily exscape this
            # by using ',"' as the separator (and trimming the final ").
            pd.read_csv(csv_file, names=self.headers, quotechar='"', sep=',"')
            trim_quote = True

        # shuffle data before splitting. Use fixed random seed so that
        # this is deterministic.
        data = data.sample(frac=1, random_state=SEED).reset_index(drop=True)

        self.text = data['text'].tolist()
        labels = None
        if 'label' in self.headers:
            labels = data['label']
            # map labels to positive sentiment value in [0,1]
            labels = labels.map(
                {'negative': 0, 'neutral': 0.5, 'positive': 1.}
            )
            labels = labels.tolist()
        self.labels = labels

        if self.subset_start is not None:
            start_idx = math.floor(len(self.text) * self.subset_start)
            end_idx = math.floor(len(self.text) * self.subset_end)
            self.text = self.text[start_idx: end_idx]
            if self.labels:
                self.labels = self.labels[start_idx: end_idx]

        if trim_quote:
            self.text = [x.strip('"') for x in self.text]
        if self.labels:
            assert len(self.labels) == len(self.text)

    def __getitem__(self, index):
        item = self.text[index]
        if self.labels:
            item = (item, self.labels[index])
        return item

    def __len__(self):
        return len(self.text)

    def collate_batch(self, batch: List):
        texts, lengths, labels = [], []

        for data in batch:
            if isinstance(data, tuple):
                text, label = data
                texts.append(text)
                labels.append(label)
            else:
                texts.append(data)
            lengths.append(len(texts[-1]))


        if labels:
            data = (data, labels)
        lengths = torch.tensor(lengths)
        return data, lengths

def get_train_valid_test_loaders(csv_file, headers, batch_size):
    """Returns train/valid/test in 60:20:20 ratio."""
    train = CSVTextDataset(csv_file, headers, 0, 0.6)
    valid = CSVTextDataset(csv_file, headers, 0.6, 0.8)
    test = CSVTextDataset(csv_file, headers, 0.8, 1)
    num_workers = 2
    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train.collate_batch,
    )
    val_loader = DataLoader(
        valid,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=valid.collate_batch,
    )
    test_loader = DataLoader(
        test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=test.collate_batch,
    )

    return train, valid, test

def test_dataset_cls(fp, headers):
    """Monolithinc test of dataset functionality."""
    dataset = CSVTextDataset(fp, headers=headers)

    # check dataset methods work w/o throwing errors
    dataset[2]
    len(dataset)

    dataset2 = CSVTextDataset(fp, headers=headers, subset_start=0, subset_end=0.75)

    # check that dataset suffle is consistent across different instances
    assert dataset[2] == dataset2[2]
    assert dataset[5] == dataset2[5]

    # check that subset length is correct
    assert len(dataset2) == math.floor(len(dataset) * 0.75), (
        f"{len(dataset2)} != {math.floor(len(dataset) * 0.75)}"
        )
    print('All tests run and passed')


if __name__ == '__main__':
    fp = 'data/headlines.csv'
    headers = ['text']
    test_dataset_cls(fp, headers=headers)
