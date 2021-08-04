import torch
w = torch.tensor(2.0, requires_grad=True)

nb_epochs = 20
for epoch in range(nb_epochs + 1):

    z = 2*w

    z.backward()
    print("수식을 w로 미분한 값 : {}".format(w.grad))

# 미분값인 2가 누적이 되므로 optimizer.zero_grad를 이용해 0으로 초기화 해주어야함.