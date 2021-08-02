import torch

m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print('Shape of Matrix 1: ', m1.shape) # 2 x 2
print('Shape of Matrix 2: ', m2.shape) # 2 x 1
print(m1.matmul(m2)) # 2 x 1
# Shape of Matrix 1:  torch.Size([2, 2])
# Shape of Matrix 2:  torch.Size([2, 1])
# tensor([[ 5.],
#         [11.]])

print(m1 * m2) # 2 x 2
print(m1.mul(m2))
# Shape of Matrix 1:  torch.Size([2, 2])
# Shape of Matrix 2:  torch.Size([2, 1])
# tensor([[1., 2.],
#         [6., 8.]])
# tensor([[1., 2.],
#         [6., 8.]])





t = torch.FloatTensor([1, 2])
print(t.mean()) # 평균
# tensor(1.5000)

t = torch.FloatTensor([[1, 2], [3, 4]])
print(t.mean(dim=0))
# tensor([2., 3.])
# dim=0은 첫번째 차원(행)을 의미. 인자로 dim을 준다면 해당 차원 제거를 의미함. 즉 '행'을 지우고 '열'만을 남긴다는 의미.
# (2,2) -> (1,2)

print(t.mean(dim=1)) # dim=1은 '열'을 지우고 '행'만을 남김
# tensor([1.5000, 3.5000])

print(t.mean(dim=-1)) # dim=-1 == dim=1
# tensor([1.5000, 3.5000])




t = torch.FloatTensor([[1, 2], [3, 4]])
print(t.sum()) # 단순히 원소 전체의 덧셈을 수행
print(t.sum(dim=0)) # 행을 제거
print(t.sum(dim=1)) # 열을 제거
print(t.sum(dim=-1)) # 열을 제거
# tensor(10.)
# tensor([4., 6.])
# tensor([3., 7.])
# tensor([3., 7.])




t = torch.FloatTensor([[1, 2], [3, 4]])
print(t.max()) # 최곳값 반환
# tensor(4.)

print(t.max(dim=0))
# (tensor([3., 4.]), tensor([1, 1]))
# [3,4]는 max, [1,1]은 argmax
# argmax는 해당 열의 인덱스 값

print('Max: ', t.max(dim=0)[0])
print('Argmax: ', t.max(dim=0)[1])
# Max:  tensor([3., 4.])
# Argmax:  tensor([1, 1])

print(t.max(dim=1))
print(t.max(dim=-1))
# (tensor([2., 4.]), tensor([1, 1]))
# (tensor([2., 4.]), tensor([1, 1]))