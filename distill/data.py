import math
from typing import List
from typing import Optional

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

SEED = 2 # DO NOT CHANGE


class CSVTextDataset(torch.utils.data.Dataset):
    """Dataset class to load textual data.

    This dataset is flexible enough to load both provided datasets
    (fin_news_all-data.csv & headlines.csv).

    It is assumed that the provided csv is small enough to fit in RAM.

    This class also has responsibility for splitting into test/train/valid
    data subsets.

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
        data = pd.read_csv(csv_file, names=self.headers)

        # shuffle data before splitting. Use fixed random seed so that
        # this is deterministic.
        data = data.sample(frac=1, random_state=SEED).reset_index(drop=True)

        self.text = data['text'].tolist()
        labels = None
        if 'label' in self.headers:
            labels = data['label'].tolist()
        self.labels = labels

        if self.subset_start is not None:
            start_idx = math.floor(len(self.text) * self.subset_start)
            end_idx = math.floor(len(self.text) * self.subset_end)
            self.text = self.text[start_idx: end_idx]
            if self.labels:
                self.labels = self.labels[start_idx: end_idx]

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

        data = pad_sequence(texts, batch_first=True)
        if labels:
            labels = pad_sequence(labels, batch_first=True)
            data = (data, labels)
        lengths = torch.tensor(lengths)
        return data, lengths


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
