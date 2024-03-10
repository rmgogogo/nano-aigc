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
        result.append(self.eos)
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
    def __init__(self, max_factor=9, transform=None, n_epochs=1):
        self.max_factor = max_factor
        self.transform = transform
        self.repeat = n_epochs # for better training performance, PyTorch didn't well handle epoch data pipeline
        pass

    def __len__(self):
        return 100*self.repeat

    def __getitem__(self, idx):
        idx == idx % self.repeat
        x = idx // 10
        y = idx % 10
        z = x + y
        result = f'{x} + {y} = {z}'

        if self.transform:
            result = self.transform(result)
        return result
    
class TokenizerTransform:
    def __init__(self):
        self.tokenizer = ToyTokenizer()

    def __call__(self, text):
        tokens = self.tokenizer.tokenize(text)
        return torch.tensor(tokens, dtype=torch.int)