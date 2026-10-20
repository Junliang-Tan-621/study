class SimpleTokenizer:
    def __init__(self):
        self.vocab = {}
        self.id_to_token = {}
    def train(self, text):

        tokens = sorted(list(set(text)))

        for i, token in enumerate(tokens):

            self.vocab[token] = i
            self.id_to_token[i] = token
    def encode(self, text):
        ids = []
        for token in text:
            ids.append(self.vocab[token])
        return ids
    def decode(self, ids):
        text =  ""
        for id in ids:
            text += self.id_to_token[id]
        return text


tokenizer = SimpleTokenizer()
tokenizer.train("abc")
result = tokenizer.decode([0, 1, 2])

print(result)

