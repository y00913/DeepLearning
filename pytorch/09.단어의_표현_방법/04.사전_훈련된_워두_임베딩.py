from torchtext.legacy import data, datasets

TEXT = data.Field(sequential=True, batch_first=True, lower=True)
LABEL = data.Field(sequential=False, batch_first=True)

trainset, testset = datasets.IMDB.splits(TEXT, LABEL)

print('훈련 데이터의 크기 : {}' .format(len(trainset)))
# 훈련 데이터의 크기 : 25000
print(vars(trainset[0]))

from gensim.models import KeyedVectors

word2vec_model = KeyedVectors.load_word2vec_format('./pytorch/09.단어의_표현_방법/eng_w2v')

print(word2vec_model['this']) # 영어 단어 'this'의 임베딩 벡터값 출력
# print(word2vec_model['self-indulgent']) # 영어 단어 'self-indulgent'의 임베딩 벡터값 출력



import torch
import torch.nn as nn
from torchtext.legacy.vocab import Vectors

vectors = Vectors(name="./pytorch/09.단어의_표현_방법/eng_w2v") # 사전 훈련된 Word2Vec 모델을 vectors에 저장
TEXT.build_vocab(trainset, vectors=vectors, max_size=10000, min_freq=10) # Word2Vec 모델을 임베딩 벡터값으로 초기화

print(TEXT.vocab.stoi)
print('임베딩 벡터의 개수와 차원 : {} '.format(TEXT.vocab.vectors.shape))
print(TEXT.vocab.vectors[0]) # <unk>의 임베딩 벡터값
print(TEXT.vocab.vectors[1]) # <pad>의 임베딩 벡터값
print(TEXT.vocab.vectors[10]) # this의 임베딩 벡터값
print(TEXT.vocab.vectors[10000]) # 단어 'self-indulgent'의 임베딩 벡터값

embedding_layer = nn.Embedding.from_pretrained(TEXT.vocab.vectors, freeze=False)
print(embedding_layer(torch.LongTensor([10]))) # 단어 this의 임베딩 벡터값



from torchtext.legacy.vocab import Glove

TEXT.build_vocab(trainset, vectors=GloVe(name='6B', dim=300), max_size=10000, min_freq=10)
LABEL.build_vocab(trainset)

print(TEXT.vocab.stoi)
print('임베딩 벡터의 개수와 차원 : {} '.format(TEXT.vocab.vectors.shape))
print(TEXT.vocab.vectors[0]) # <unk>의 임베딩 벡터값
print(TEXT.vocab.vectors[1]) # <pad>의 임베딩 벡터값
print(TEXT.vocab.vectors[10]) # this의 임베딩 벡터값
print(TEXT.vocab.vectors[9999]) # seeing의 임베딩 벡터값

embedding_layer = nn.Embedding.from_pretrained(TEXT.vocab.vectors, freeze=False)
embedding_layer(torch.LongTensor([10])) # 단어 this의 임베딩 벡터값