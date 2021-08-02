import torch

t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])
print(t)

print(t.dim())  # rank 차원
print(t.shape)  # shape
print(t.size()) # shape
# 1
# torch.Size([7])
# torch.Size([7])

print(t[0], t[1], t[-1])  # 인덱스로 접근
print(t[2:5], t[4:-1])    # 슬라이싱
print(t[:2], t[3:])       # 슬라이싱
# tensor(0.) tensor(1.) tensor(6.)
# tensor([2., 3., 4.]) tensor([4., 5.])
# tensor([0., 1.]) tensor([3., 4., 5., 6.])