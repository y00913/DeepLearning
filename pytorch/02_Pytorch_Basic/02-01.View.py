import torch
import numpy as np

t = np.array([[[0, 1, 2],
               [3, 4, 5]],
              [[6, 7, 8],
               [9, 10, 11]]])
ft = torch.FloatTensor(t)

print(ft.shape)
# torch.Size([2, 2, 3])

print(ft.view([-1, 3])) # ft라는 텐서를 (?, 3)의 크기로 변경, 3D -> 2D
print(ft.view([-1, 3]).shape)
# tensor([[ 0.,  1.,  2.],
#         [ 3.,  4.,  5.],
#         [ 6.,  7.,  8.],
#         [ 9., 10., 11.]])
# torch.Size([4, 3])

# view는 기본적으로 변경 전과 변경 후의 텐서 안의 원소의 개수가 유지되어야함
# 파이토치의 view는 사이즈가 -1로 설정되면 다른 차원으로부터 해당 값을 유추함

print(ft.view([-1, 1, 3])) # 3D 크기 변경
print(ft.view([-1, 1, 3]).shape)
# tensor([[[ 0.,  1.,  2.]],

#         [[ 3.,  4.,  5.]],

#         [[ 6.,  7.,  8.]],

#         [[ 9., 10., 11.]]])
# torch.Size([4, 1, 3])