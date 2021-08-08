import torch
import torch.nn.functional as F

torch.manual_seed(1)

# Low level
z = torch.rand(3, 5, requires_grad=True)
hypothesis = F.softmax(z, dim=1)
print(torch.log(hypothesis))

# High level. F.softmax() + torch.log() = F.log_softmax()
print(F.log_softmax(z, dim=1))

y = torch.randint(5, (3,)).long()
y_one_hot = torch.zeros_like(hypothesis)
y_one_hot.scatter_(1, y.unsqueeze(1), 1)

# Low level
# 첫번째 수식
print((y_one_hot * -torch.log(F.softmax(z, dim=1))).sum(dim=1).mean())

# 두번째 수식
print((y_one_hot * - F.log_softmax(z, dim=1)).sum(dim=1).mean())

# High level
# 세번째 수식. F.nll_loss()
print(F.nll_loss(F.log_softmax(z, dim=1), y))

# 네번째 수식
print(F.cross_entropy(z, y))