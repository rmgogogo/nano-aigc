'''
Toy dataset and tokenizer for quick prototype.

ds = ToyDataset()
tn = ToyTokenizer()
print(ds[30])
print(tn.tokenize(ds[30]))
print(tn.detokenize(tn.tokenize(ds[30])))
'''

import torch

class ToyTokenizer:
    def __init__(self):
        self.eos = 0
        self.token_add = 1
        self.token_equal = 2
        self.zero = 3

    def tokenize(self, text):
        chars = text.split(' ')
        result = []
        for char in chars:
            if len(char)==0:
                continue
            if char == '+':
                token = self.token_add
            elif char == '=':
                token = self.token_equal
            else:
                num = int(char)
                token = self.zero + num
            result.append(token)
        return result
    
    def token2char(self, token):
        if token == self.token_add:
            char = '+'
        elif token == self.token_equal:
            char = '='
        elif token == self.eos:
            char = ''
        else:
            char = str(token.item() - self.zero)
        return char

    def detokenize(self, tokens):
        result = []
        for token in tokens:
            char = self.token2char(token)
            result.append(char)
        return ' '.join(result)

class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None, n_epochs=1):
        self.transform = transform
        self.repeat = n_epochs # for better training performance to avoid setup new pipeline
        self.num_samples = 100
        pass

    def __len__(self):
        return self.num_samples * self.repeat

    def __getitem__(self, idx):
        idx = idx % self.num_samples
        x = idx // 10
        y = idx % 10
        z = x + y
        result = f'{x} + {y} = {z}'

        if self.transform:
            result = self.transform(result)
        return result
    
class TokenizerTransform:
    def __init__(self, max_seq=10):
        self.tokenizer = ToyTokenizer()
        self.max_seq = max_seq

    def __call__(self, text):
        tokens = self.tokenizer.tokenize(text)
        padding_size = self.max_seq - len(tokens)
        tokens = tokens + [self.tokenizer.eos for _ in range(padding_size)]
        return torch.tensor(tokens, dtype=torch.int)