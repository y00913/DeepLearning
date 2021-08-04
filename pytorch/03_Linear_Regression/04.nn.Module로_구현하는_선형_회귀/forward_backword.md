# forword
- H(x) 식에 입력 x로부터 예측된 y를 얻는 것을 forward 연산이라고 함.
- 학습 전, prediction = model(x_train)은 x_train으로부터 예측값을 리턴하므로 forward 연산.
- 학습 후, pred_y = model(new_var)는 임의의 값 new_var로부터 예측값을 리턴하므로 forward 연산.
  
# backward
- 학습 과정에서 비용 함수를 미분하여 기울기를 구하는 것을 backward 연산이라고 함.
- cost.backward()는 비용 함수로부터 기울기를 구하라는 의미이며 backward 연산임.