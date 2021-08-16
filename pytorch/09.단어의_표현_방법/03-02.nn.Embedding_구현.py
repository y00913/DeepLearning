import torch

train_data = 'you need to know how to code'
word_set = set(train_data.split()) # 중복을 제거한 단어들의 집합인 단어 집합 생성.
vocab = {tkn: i+2 for i, tkn in enumerate(word_set)}  # 단어 집합의 각 단어에 고유한 정수 맵핑.
vocab['<unk>'] = 0
vocab['<pad>'] = 1

import torch.nn as nn

embedding_layer = nn.Embedding(num_embeddings = len(vocab), 
                               embedding_dim = 3,
                               padding_idx = 1)

print(embedding_layer.weight)