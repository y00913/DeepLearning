import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

W = torch.zeros((2, 1), requires_grad=True) # 크기는 2 x 1
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # Cost 계산
    hypothesis = 1 / (1 + torch.exp(-(x_train.matmul(W) + b)))
    losses = -(y_train * torch.log(hypothesis) + (1 - y_train) * torch.log(1 - hypothesis))
    cost = losses.mean()

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

# 훈련 된 W, b로 예측값 출력
hypothesis = torch.sigmoid(x_train.matmul(W) + b)
print(hypothesis)
# tensor([[2.7648e-04],
#         [3.1608e-02],
#         [3.8977e-02],
#         [9.5622e-01],
#         [9.9823e-01],
#         [9.9969e-01]], grad_fn=<SigmoidBackward>)

# 0.5를 넘기면 True, 아니면 False
prediction = hypothesis >= torch.FloatTensor([0.5])
print(prediction)
# tensor([[False],
#         [False],
#         [False],
#         [ True],
#         [ True],
#         [ True]])