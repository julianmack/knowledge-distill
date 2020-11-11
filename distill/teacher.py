from typing import List
import json
from pathlib import Path

import torch

# to prevent a pandas FutureWarning do...
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from fast_bert.prediction import BertClassificationPredictor

from distill.labels import label_to_idx

class TeacherNLPClassifier(torch.nn.Module):
    def __init__(self, model_dir, label_to_idx=label_to_idx):
        super().__init__()
        model_dir = Path(model_dir)
        model_config = model_dir / 'model_config.json'
        with open(model_config) as f:
            config = json.load(f)

        self.model = BertClassificationPredictor(
            model_path=str(model_dir / 'model_out'),
            label_path=str(model_dir), # location for labels.csv file
            model_type=config['model_type'],
            multi_label=config['multi_label'],
            do_lower_case=config['do_lower_case'],
        )
        self.label_to_idx = label_to_idx

    def forward(self, texts: List[str], lengths=None):
        results = self.model.predict_batch(texts)
        # results is a List[List[Tuple]] of `label, probability`.
        # convert this to a onehot tensor
        final = torch.zeros((len(results), len(self.label_to_idx)))
        for i, result in enumerate(results):
            for (label, prob) in result:
                idx = self.label_to_idx[label]
                final[i, idx] = prob
        return final

def test_teacher(model_dir):
    model = TeacherNLPClassifier(model_dir)
    texts = ["This is good news!", "This is terrible news!"]

    res = model(texts)
    print(res)

if __name__ == '__main__':
    model_dir = 'model'
    test_teacher(model_dir)
