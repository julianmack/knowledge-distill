import torch
import numpy as np


class GloveEmbedding(torch.nn.Module):
    """Loads and excecutes Glove embedding."""
    def __init__(self, location):
        super().__init__()
        embeddings_dict = {}
        with open(location, 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embeddings_dict[word] = vector
        # embeddings_dict is word -> vector
        self.word_to_id_ = {k: i for i, k in enumerate(embeddings_dict.keys())}

        embed_tensor = torch.FloatTensor(list(embeddings_dict.values()))
        self.embed = torch.nn.Embedding.from_pretrained(embed_tensor)

        # null value is 'unk' vector
        self.null = self.word_to_id_['unk']

    def word_to_id(self, word):
        return self.word_to_id_.get(word, self.null)

    def forward(self, x):
        return self.embed(x)

def test_glove_embedding(location, dim=50):
    """Check glove embedding functions work."""
    embedding = GloveEmbedding(location)
    king_id = embedding.word_to_id('king')
    king_id = torch.LongTensor([king_id])
    king_embedding = embedding(king_id)
    assert king_embedding.shape == (1, dim), (
        f'{king_embedding.shape} != (1, {dim})'
    )

if __name__ == '__main__':
    location = 'model/glove.6B/glove.6B.50d.txt'
    test_glove_embedding(location, dim=50)
