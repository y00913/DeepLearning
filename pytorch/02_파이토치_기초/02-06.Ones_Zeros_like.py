import torch

x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])

print(torch.ones_like(x)) # 입력 텐서와 크기를 동일하게 하면서 값을 1로 채움
# tensor([[1., 1., 1.],
#         [1., 1., 1.]])

print(torch.zeros_like(x)) # 입력 텐서와 크기를 동일하게 하면서 값을 0으로 채우기
# tensor([[0., 0., 0.],
#         [0., 0., 0.]])