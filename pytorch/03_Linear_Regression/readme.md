# 경사 하강법

- 경사 하강법은 비용 함수를 미분하여 이 함수의 기울기(gradient)를 구해서 비용이 최소화 되는 방향을 찾아내는 알고리즘.
- 비용 함수 = 손실 함수 = 오차 함수
- 파이토치에서 자동 미분을 지원하여 경사 하강법을 쉽게 사용할 수 있음.

# forword

- H(x) 식에 입력 x로부터 예측된 y를 얻는 것을 forward 연산이라고 함.
- 학습 전, prediction = model(x_train)은 x_train으로부터 예측값을 리턴하므로 forward 연산.
- 학습 후, pred_y = model(new_var)는 임의의 값 new_var로부터 예측값을 리턴하므로 forward 연산.

# backward

- 학습 과정에서 비용 함수를 미분하여 기울기를 구하는 것을 backward 연산이라고 함.
- cost.backward()는 비용 함수로부터 기울기를 구하라는 의미이며 backward 연산임.
