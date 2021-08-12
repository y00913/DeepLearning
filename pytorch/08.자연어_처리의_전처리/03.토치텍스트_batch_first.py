import urllib.request
import pandas as pd

urllib.request.urlretrieve("https://raw.githubusercontent.com/LawrenceDuan/IMDb-Review-Analysis/master/IMDb_Reviews.csv", filename="./pytorch/08.자연어_처리의_전처리/IMDb_Reviews.csv")

df = pd.read_csv('./pytorch/08.자연어_처리의_전처리/IMDb_Reviews.csv', encoding='latin1')

print('전체 샘플의 개수 : {}'.format(len(df)))

train_df = df[:25000]
test_df = df[25000:]

train_df.to_csv("./pytorch/08.자연어_처리의_전처리/train_data.csv", index=False)
test_df.to_csv("./pytorch/08.자연어_처리의_전처리/test_data.csv", index=False)

from torchtext.legacy import data # torchtext.data 임포트

# 필드 정의
TEXT = data.Field(sequential=True,
                  use_vocab=True,
                  tokenize=str.split,
                  lower=True,
                  batch_first=True, # <== 이 부분을 True로 합니다.
                  fix_length=20)

LABEL = data.Field(sequential=False,
                   use_vocab=False,
                   batch_first=False,
                   is_target=True)

from torchtext.legacy.data import TabularDataset
from torchtext.legacy.data import Iterator

# TabularDataset은 데이터를 불러오면서 필드에서 정의했던 토큰화 방법으로 토큰화를 수행합니다.
train_data, test_data = TabularDataset.splits(
        path='.', train='./pytorch/08.자연어_처리의_전처리/train_data.csv', test='./pytorch/08.자연어_처리의_전처리/test_data.csv', format='csv',
        fields=[('text', TEXT), ('label', LABEL)], skip_header=True)

# 정의한 필드에 .build_vocab() 도구를 사용하면 단어 집합을 생성합니다.
TEXT.build_vocab(train_data, min_freq=10, max_size=10000) # 10,000개의 단어를 가진 단어 집합 생성

# 배치 크기를 정하고 첫번째 배치를 출력해보겠습니다.
batch_size = 5
train_loader = Iterator(dataset=train_data, batch_size = batch_size)
batch = next(iter(train_loader)) # 첫번째 미니배치

print(batch.text)
print(batch.text.shape)

# 필드 정의
TEXT = data.Field(sequential=True,
                  use_vocab=True,
                  tokenize=str.split,
                  lower=True,
                  fix_length=20)

LABEL = data.Field(sequential=False,
                   use_vocab=False,
                   batch_first=False,
                   is_target=True)

# 첫번째 미니 배치 출력
print(batch.text)
print(batch.text.shape)