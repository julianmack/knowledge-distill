import torch
import numpy as np
import tokenizers
from tokenizers import normalizers
from tokenizers.pre_tokenizers import Whitespace


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
        self.id_to_word_ = {i: k for k, i in self.word_to_id_.items()}

        embed_tensor = torch.FloatTensor(list(embeddings_dict.values()))
        self.embed = torch.nn.Embedding.from_pretrained(embed_tensor)

        # null value is 'unk' vector
        self.null = self.word_to_id_['unk']

    def word_to_id(self, word):
        return self.word_to_id_.get(word, self.null)
    def words_to_ids(self, words):
        return [self.word_to_id(y) for y in words]
    def id_to_word(self, id):
        if id >= len(self.word_to_id_):
            raise ValueError(f'{id=} is not valid')
        return self.id_to_word_[id]
    def ids_to_words(self, ids):
        return [self.id_to_word(id) for id in ids]
    def forward(self, x):
        return self.embed(x)

class GloveTokenizer():
    """Glove Tokenizer that maps from text -> vectors.

    GloVe embeddings are not contextual (are just a lookup) so in the
    huggingface paradim, they are actually part of the tokenizer. Doing this
    properly would involve creating a pre-trained tokenizer
    that is formed using something to the effect of:
    ```python
    import tokenizer
    from tokenizers.models import WordPiece
    tokenizer = tokenizers.Tokenizer(WordPiece())
    # add normalizer/pre_tokenizer
    # ...
    ```
    ... but this feels like overkill here. I'll make my own version here
    but it will quite a bit slower.
    """
    def __init__(self, glove_fp=None, embedding=None):
        assert glove_fp or embedding
        super().__init__()

        self.pre_token = Whitespace() # i.e. tokenize on whitespace + punct

        if glove_fp:
            self.model = GloveEmbedding(location=glove_fp)
        else:
            self.model = embedding

    def normalize(self, text):
        """Lowercases text."""
        return text.lower()

    def to_words(self, text):
        """Sentence to tokens (on whitespace)"""
        tokens = self.pre_token.pre_tokenize(text)
        # tokens is List[Tuple] where tuple: (token, position).
        return [x for x, _ in tokens]

    def __call__(self, text: str):
        text = self.normalize(text)
        words = self.to_words(text)
        ids = self.model.words_to_ids(words)
        ids = torch.LongTensor(ids)
        return self.model(ids), words

def test_glove_embedding(embedding, dim=50):
    """Check glove embedding functions work."""
    king_id = embedding.word_to_id('king')
    king_id = torch.LongTensor([king_id])
    king_embedding = embedding(king_id)
    assert king_embedding.shape == (1, dim), (
        f'{king_embedding.shape} != (1, {dim})'
    )

def test_glove_tokenizer(embedding):
    tokenizer = GloveTokenizer(embedding=embedding)

    text = 'test, sentence Here'
    expected_tokens = ['test', ',', 'sentence', 'here']

    embedded, words = tokenizer(text)
    assert words == expected_tokens

    text = 'Company ltd. number 129'
    expected_tokens = ['company', 'ltd', '.', 'number', '129']

    embedded, words = tokenizer(text)
    assert words == expected_tokens

    ids = embedding.words_to_ids(expected_tokens)
    words_reconstruct = embedding.ids_to_words(ids)
    # none unknown so all should be the same
    assert expected_tokens == words_reconstruct

    ids = embedding.words_to_ids(['adsaas', 'hello', 'asddasda'])
    words_reconstruct = embedding.ids_to_words(ids)
    # none unknown so all should be the same
    assert ['unk', 'hello', 'unk'] == words_reconstruct

    print('tokenizer tests run and complete')

if __name__ == '__main__':
    location = 'model/glove.6B/glove.6B.50d.txt'
    embedding = GloveEmbedding(location)
    test_glove_embedding(embedding, dim=50)

    test_glove_tokenizer(embedding)
