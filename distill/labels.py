all_labels = [':)', ':|', ':(']


def prob_to_label(prob):
    """Converts prob to label."""
    if prob < 0.3:
        label = ':('
    elif prob < 0.7:
        label = ':|'
    elif prob <= 1.0:
        label = ':)'
    else:
        probability = prob.item()
        raise ValueError(f"{probability=} is not a valid probability")
    return label


def probs_to_labels(probs):
    """Batched version of `prob_to_label`."""
    labels = []
    for prob in probs:
        labels.append(prob_to_label(prob))
    return labels
