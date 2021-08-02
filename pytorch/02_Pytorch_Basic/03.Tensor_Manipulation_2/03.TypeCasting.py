import torch

lt = torch.LongTensor([1, 2, 3, 4])
print(lt.float())
# tensor([1., 2., 3., 4.])

bt = torch.ByteTensor([True, False, False, True])
print(bt.long())
print(bt.float())
# tensor([1, 0, 0, 1])
# tensor([1., 0., 0., 1.])