import nltk
# nltk.download('punkt')
import urllib.request
import zipfile
from lxml import etree
import re
from nltk.tokenize import word_tokenize, sent_tokenize

# 데이터 다운로드
urllib.request.urlretrieve("https://raw.githubusercontent.com/GaoleMeng/RNN-and-FFNN-textClassification/master/ted_en-20160408.xml", filename="./pytorch/09.단어의_표현_방법/ted_en-20160408.xml")

targetXML=open('./pytorch/09.단어의_표현_방법/ted_en-20160408.xml', 'r', encoding='UTF8')
# 저자의 경우 윈도우 바탕화면에서 작업하여서 'C:\Users\USER\Desktop\ted_en-20160408.xml'이 해당 파일의 경로.  
target_text = etree.parse(targetXML)
parse_text = '\n'.join(target_text.xpath('//content/text()'))
# xml 파일로부터 <content>와 </content> 사이의 내용만 가져온다.

content_text = re.sub(r'\([^)]*\)', '', parse_text)
# 정규 표현식의 sub 모듈을 통해 content 중간에 등장하는 (Audio), (Laughter) 등의 배경음 부분을 제거.
# 해당 코드는 괄호로 구성된 내용을 제거.

sent_text = sent_tokenize(content_text)
# 입력 코퍼스에 대해서 NLTK를 이용하여 문장 토큰화를 수행.

normalized_text = []
for string in sent_text:
     tokens = re.sub(r"[^a-z0-9]+", " ", string.lower())
     normalized_text.append(tokens)
# 각 문장에 대해서 구두점을 제거하고, 대문자를 소문자로 변환.

result = []
result = [word_tokenize(sentence) for sentence in normalized_text]
# 각 문장에 대해서 NLTK를 이용하여 단어 토큰화를 수행.

print('총 샘플의 개수 : {}'.format(len(result)))
for line in result[:3]: # 샘플 3개만 출력
    print(line)



from gensim.models import Word2Vec, KeyedVectors
model = Word2Vec(sentences=result, vector_size=100, window=5, min_count=5, workers=4, sg=0)

model_result = model.wv.most_similar("man")
print(model_result)

model.wv.save_word2vec_format('./pytorch/09.단어의_표현_방법/eng_w2v') # 모델 저장
loaded_model = KeyedVectors.load_word2vec_format("./pytorch/09.단어의_표현_방법/eng_w2v") # 모델 로드

model_result = loaded_model.most_similar("man")
print(model_result)