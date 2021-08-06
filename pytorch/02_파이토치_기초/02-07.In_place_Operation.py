import torch

x = torch.FloatTensor([[1, 2], [3, 4]])

print(x.mul(2.)) # 곱하기 2를 수행한 결과를 출력
print(x) # 기존의 값 출력
# tensor([[2., 4.],
#         [6., 8.]])
# tensor([[1., 2.],
#         [3., 4.]])

print(x.mul_(2.))  # 곱하기 2를 수행한 결과를 변수 x에 값을 저장하면서 결과를 출력
print(x) # 기존의 값 출력
# tensor([[2., 4.],
#         [6., 8.]])
# tensor([[2., 4.],
#         [6., 8.]])