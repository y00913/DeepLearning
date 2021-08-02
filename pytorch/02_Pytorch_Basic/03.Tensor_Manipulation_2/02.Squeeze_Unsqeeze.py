import torch

ft = torch.FloatTensor([[0], [1], [2]])
print(ft)
print(ft.shape)
# tensor([[0.],
#         [1.],
#         [2.]])
# torch.Size([3, 1])

print(ft.squeeze()) # 1인 차원을 제거. [3,1] -> [3,]
print(ft.squeeze().shape)
# tensor([0., 1., 2.])
# torch.Size([3])




ft = torch.Tensor([0, 1, 2])
print(ft.shape)
# torch.Size([3])

print(ft.unsqueeze(0)) # 인덱스가 0부터 시작하므로 0은 첫번째 차원을 의미함. 특정 위치에 1인 차원 추가. [3,] -> [1,3]
print(ft.unsqueeze(0).shape)
# tensor([[0., 1., 2.]])
# torch.Size([1, 3])

print(ft.view(1, -1)) # unsqueeze와 같은 결과를 보임.
print(ft.view(1, -1).shape)
# tensor([[0., 1., 2.]])
# torch.Size([1, 3])

print(ft.unsqueeze(1)) # [3,] -> [3,1]
print(ft.unsqueeze(1).shape)
# tensor([[0.],
#         [1.],
#         [2.]])
# torch.Size([3, 1])

print(ft.unsqueeze(-1)) # [3,] -> [3,1]
print(ft.unsqueeze(-1).shape)
# tensor([[0.],
#         [1.],
#         [2.]])
# torch.Size([3, 1])