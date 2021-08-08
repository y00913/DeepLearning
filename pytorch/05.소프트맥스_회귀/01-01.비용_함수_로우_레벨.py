import torch
import torch.nn.functional as F

torch.manual_seed(1)

z = torch.FloatTensor([1, 2, 3])
hypothesis = F.softmax(z, dim=0)
print(hypothesis)
# tensor([0.0900, 0.2447, 0.6652])

print(hypothesis.sum())
# tensor(1.)

z = torch.rand(3, 5, requires_grad=True)
hypothesis = F.softmax(z, dim=1)
print(hypothesis)
# tensor([[0.2645, 0.1639, 0.1855, 0.2585, 0.1277],
#         [0.2430, 0.1624, 0.2322, 0.1930, 0.1694],
#         [0.2226, 0.1986, 0.2326, 0.1594, 0.1868]], grad_fn=<SoftmaxBackward>)

y = torch.randint(5, (3,)).long()
print(y)
# tensor([0, 2, 1])

# 모든 원소가 0의 값을 가진 3 × 5 텐서 생성
y_one_hot = torch.zeros_like(hypothesis) 
print(y_one_hot.scatter_(1, y.unsqueeze(1), 1))
# tensor([[1., 0., 0., 0., 0.],
#         [0., 0., 1., 0., 0.],
#         [0., 1., 0., 0., 0.]])

cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
print(cost)
# tensor(1.4689, grad_fn=<MeanBackward1>)