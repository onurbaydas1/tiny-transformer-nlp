from collections import Counter

class BPETokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.token_to_id = {}
        self.id_to_token = {}

    def get_vocab(self):
        return self.token_to_id

    def train(self, texts):
        tokens = [list(word) + ['</w>'] for line in texts for word in line.split()]
        vocab = Counter([tuple(token) for token in tokens])

        for _ in range(self.vocab_size):
            pairs = Counter()
            for word, freq in vocab.items():
                for i in range(len(word) - 1):
                    pairs[(word[i], word[i+1])] += freq
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_vocab = Counter()
            for word, freq in vocab.items():
                w = list(word)
                i = 0
                while i < len(w) - 1:
                    if (w[i], w[i+1]) == best:
                        w[i:i+2] = [''.join(best)]
                    i += 1
                new_vocab[tuple(w)] += freq
            vocab = new_vocab

        self.vocab = vocab
        tokens = set([t for word in vocab for t in word])
        self.token_to_id = {tok: i for i, tok in enumerate(sorted(tokens))}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

    def encode(self, text):
        words = text.strip().split()
        tokens = []
        for word in words:
            chars = list(word) + ['</w>']
            while len(chars) > 1:
                for i in range(len(chars) - 1):
                    pair = chars[i] + chars[i+1]
                    if pair in self.token_to_id:
                        chars[i:i+2] = [pair]
                        break
                else:
                    break
            tokens += chars
        return [self.token_to_id[t] for t in tokens if t in self.token_to_id]

    def decode(self, ids):
        return ' '.join(self.id_to_token[i].replace('</w>', '') for i in ids if i in self.id_to_token)
