import torch
import torch.nn as nn

# 배치 크기 × 채널 × 높이(height) × 너비(widht)의 크기의 텐서를 선언
inputs = torch.Tensor(1, 1, 28, 28)
print('텐서의 크기 : {}'.format(inputs.shape))

conv1 = nn.Conv2d(1, 32, 3, padding=1)
print(conv1)
# Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
print(conv2)
# Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
pool = nn.MaxPool2d(2)
print(pool)
# MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")

out = conv1(inputs)
print(out.shape)
# torch.Size([1, 32, 28, 28])
out = pool(out)
print(out.shape)
# torch.Size([1, 32, 14, 14])
out = conv2(out)
print(out.shape)
# torch.Size([1, 64, 14, 14])
out = pool(out)
print(out.shape)
# torch.Size([1, 64, 7, 7])

# 첫번째 차원인 배치 차원은 그대로 두고 나머지는 펼쳐라
out = out.view(out.size(0), -1) 
print(out.shape)
torch.Size([1, 3136])

fc = nn.Linear(3136, 10) # input_dim = 3,136, output_dim = 10
out = fc(out)
print(out.shape)
# torch.Size([1, 10])