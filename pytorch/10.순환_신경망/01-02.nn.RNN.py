import torch
import torch.nn as nn

input_size = 5 # 입력의 크기
hidden_size = 8 # 은닉 상태의 크기

# (batch_size, time_steps, input_size)
inputs = torch.Tensor(1, 10, 5)

cell = nn.RNN(input_size, hidden_size, batch_first=True)
outputs, _status = cell(inputs)

print(outputs.shape) # 모든 time-step의 hidden_state
print(_status.shape) # 최종 time-step의 hidden_state



# (batch_size, time_steps, input_size)
inputs = torch.Tensor(1, 10, 5)

cell = nn.RNN(input_size = 5, hidden_size = 8, num_layers = 2, batch_first=True)
outputs, _status = cell(inputs)

print(outputs.shape) # 모든 time-step의 hidden_state
print(_status.shape) # (층의 개수, 배치 크기, 은닉 상태의 크기)



# (batch_size, time_steps, input_size)
inputs = torch.Tensor(1, 10, 5)

cell = nn.RNN(input_size = 5, hidden_size = 8, num_layers = 2, batch_first=True, bidirectional = True)
outputs, _status = cell(inputs)

print(outputs.shape) # (배치 크기, 시퀀스 길이, 은닉 상태의 크기 x 2)
print(_status.shape) # (층의 개수 x 2, 배치 크기, 은닉 상태의 크기)