import pandas as pd

data = pd.read_csv('gpascore.csv')

data = data.dropna()
# data = data.fillna(100)

y데이터 = data['admit'].values
x데이터 = []

for i, rows in data.iterrows() :
    x데이터.append([ rows['gre'], rows['gre'], rows['gre'] ])

import numpy as np
import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(np.array(x데이터), np.array(y데이터) ,epochs=1000) # fit(학습데이터, 실제정답, 학습횟수)

# 예측
예측값 = model.predict( [ [750, 3.70, 3], [400, 2.2, 1] ] ) # predict(x데이터)
print(예측값)

# 1. 모델 만들기
# 2. 데이터 넣고 학습
# 3. 새로운 데이터 예측
# +데이터 전처리, 모델 튜닝