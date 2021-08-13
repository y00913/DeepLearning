# 영어
import gensim

# 구글의 사전 훈련된 Word2Vec 모델을 로드합니다.
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin 파일 경로', binary=True)  

print(model.vectors.shape) # 모델의 크기 확인

print (model.similarity('this', 'is')) # 두 단어의 유사도 계산하기
print (model.similarity('post', 'book'))

print(model['book']) # 단어 'book'의 벡터 출력



# 한국어
import gensim
model = gensim.models.Word2Vec.load('./pytorch/09.단어의_표현_방법/ko.bin')

result = model.wv.most_similar("강아지")
print(result)