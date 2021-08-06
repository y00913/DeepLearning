import torch

x = torch.FloatTensor([[1, 2], [3, 4]])
y = torch.FloatTensor([[5, 6], [7, 8]])

print(torch.cat([x, y], dim=0)) # torch.cat은 어느 차원을 늘릴지 결정 가능. 여기서는 dim=0을 늘림. 두개의 [2,2] -> 한개의 [4,2]
# tensor([[1., 2.],
#         [3., 4.],
#         [5., 6.],
#         [7., 8.]])

print(torch.cat([x, y], dim=1)) # 두개의 [2,2] -> 한개의 [2,4]
# tensor([[1., 2., 5., 6.],
#         [3., 4., 7., 8.]])