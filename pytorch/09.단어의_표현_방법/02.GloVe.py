# 영어
import gensim

# 구글의 사전 훈련된 Word2Vec 모델을 로드합니다.
result = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin 파일 경로', binary=True)  

from glove import Corpus, Glove

corpus = Corpus()
corpus.fit(result, window=5)
# 훈련 데이터로부터 GloVe에서 사용할 동시 등장 행렬 생성

glove = Glove(no_components=100, learning_rate=0.05)
glove.fit(corpus.matrix, epochs=20, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)
# 학습에 이용할 쓰레드의 개수는 4로 설정, 에포크는 20.

model_result1=glove.most_similar("man")
print(model_result1)

model_result2=glove.most_similar("boy")
print(model_result2)

model_result3=glove.most_similar("university")
print(model_result3)

model_result4=glove.most_similar("water")
print(model_result4)

model_result5=glove.most_similar("physics")
print(model_result5)

model_result6=glove.most_similar("muscle")
print(model_result6)

model_result7=glove.most_similar("clean")
print(model_result7)