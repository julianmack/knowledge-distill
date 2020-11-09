import torch
all_labels = ['negative', 'neutral', 'positive']
label_to_idx = {label: i for i, label in enumerate(all_labels)}

def prob_to_label(prob):
    """Converts prob to label."""
    idx = torch.argmax(prob)
    return all_labels[idx]


def probs_to_labels(probs):
    """Batched version of `prob_to_label`."""
    labels = []
    for prob in probs:
        labels.append(prob_to_label(prob))
    return labels
