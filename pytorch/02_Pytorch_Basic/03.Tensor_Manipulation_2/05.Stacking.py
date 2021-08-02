import torch

x = torch.FloatTensor([1, 4])
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])

print(torch.stack([x, y, z]))
# tensor([[1., 4.],
#         [2., 5.],
#         [3., 6.]])

print(torch.cat([x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)], dim=0)) # torch.stack()과 결과 같음.
# tensor([[1., 4.],
#         [2., 5.],
#         [3., 6.]])

print(torch.stack([x, y, z], dim=1)) # dim=1 차원 증가
# tensor([[1., 2., 3.],
#         [4., 5., 6.]])