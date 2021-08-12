import urllib.request
import pandas as pd

urllib.request.urlretrieve("https://raw.githubusercontent.com/LawrenceDuan/IMDb-Review-Analysis/master/IMDb_Reviews.csv", filename="./pytorch/08.자연어_처리의_전처리/IMDb_Reviews.csv")

df = pd.read_csv('./pytorch/08.자연어_처리의_전처리/IMDb_Reviews.csv', encoding='latin1')
df.head()

print('전체 샘플의 개수 : {}'.format(len(df)))

train_df = df[:25000]
test_df = df[25000:]

train_df.to_csv("./pytorch/08.자연어_처리의_전처리/train_data.csv", index=False)
test_df.to_csv("./pytorch/08.자연어_처리의_전처리/test_data.csv", index=False)

from torchtext.legacy import data # torchtext.legacy.data 임포트

# 필드 정의
TEXT = data.Field(sequential=True,
                  use_vocab=True,
                  tokenize=str.split,
                  lower=True,
                  batch_first=True,
                  fix_length=20)

LABEL = data.Field(sequential=False,
                   use_vocab=False,
                   batch_first=False,
                   is_target=True)

from torchtext.legacy.data import TabularDataset

train_data, test_data = TabularDataset.splits(
        path='.', train='./pytorch/08.자연어_처리의_전처리/train_data.csv', test='./pytorch/08.자연어_처리의_전처리/test_data.csv', format='csv',
        fields=[('text', TEXT), ('label', LABEL)], skip_header=True)

print('훈련 샘플의 개수 : {}'.format(len(train_data)))
print('테스트 샘플의 개수 : {}'.format(len(test_data)))
print(vars(train_data[0]))
# 필드 구성 확인.
print(train_data.fields.items())

TEXT.build_vocab(train_data, min_freq=10, max_size=10000)

print('단어 집합의 크기 : {}'.format(len(TEXT.vocab)))
print(TEXT.vocab.stoi)

from torchtext.legacy.data import Iterator

batch_size = 5
train_loader = Iterator(dataset=train_data, batch_size = batch_size)
test_loader = Iterator(dataset=test_data, batch_size = batch_size)

print('훈련 데이터의 미니 배치 수 : {}'.format(len(train_loader)))
print('테스트 데이터의 미니 배치 수 : {}'.format(len(test_loader)))

batch = next(iter(train_loader)) # 첫번째 미니배치
print(type(batch))

batch = next(iter(train_loader)) # 첫번째 미니배치
print(batch.text[0]) # 첫번째 미니배치 중 첫번째 샘플