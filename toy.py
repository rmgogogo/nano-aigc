'''
Toy dataset and tokenizer for quick prototype.

ds = ToyDataset()
tn = ToyTokenizer()
print(ds[30])
print(tn.tokenize(ds[30]))
print(tn.detokenize(tn.tokenize(ds[30])))
'''

import torch

class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, max_factor=9):
        self.max_factor = max_factor
        pass

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        x = idx // 10
        y = idx % 10
        z = x + y
        text = f'{x} + {y} = {z}'
        return text
        
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

    def detokenize(self, tokens):
        result = []
        for token in tokens:
            if token == self.token_add:
                char = '+'
            elif token == self.token_equal:
                char = '='
            else:
                char = str(token - self.zero)
            result.append(char)
        return ' '.join(result)