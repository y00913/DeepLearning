import numpy as np

t = np.array([0., 1., 2., 3., 4., 5., 6.])
# 파이썬으로 설명하면 List를 생성해서 np.array로 1차원 array로 변환함.
print(t) 
# [0. 1. 2. 3. 4. 5. 6.]

print('Rank of t: ', t.ndim)
print('Shape of t: ', t.shape)
# Rank of t:  1
# Shape of t:  (7,)

print('t[0] t[1] t[-1] = ', t[0],t[1],t[-1]) # 인덱스를 통한 원소 접근
# t[0] t[1] t[-1] =  0.0 1.0 6.0

print('t[2:5] t[4:-1]  = ', t[2:5], t[4:-1]) # [시작 번호 : 끝 번호]로 범위 지정을 통해 가져옴.
# t[2:5] t[4:-1]  =  [2. 3. 4.] [4. 5.]

print('t[:2] t[3:]  = ', t[:2],t[3:]) # 시작 번호를 생략한 경우와 끝 번호를 생략한 경우
# t[:2] t[3:]     =  [0. 1.] [3. 4. 5. 6.]