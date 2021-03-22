def solution(a, b):
    week=["THU","FRI","SAT","SUN","MON","TUE","WED"]
    sum = 0
    
    for i in range(1,a):
      if i<=7:
        if i==2:
          sum += 29
        elif i%2==1:
          sum += 31
        elif i%2==0:
          sum += 30
      else:
        if i%2==0:
          sum += 31
        elif i%2==1:
          sum += 30
          
    sum += b
    sum %= 7
    
    return week[sum]
    
a=5
b=24
print(solution(a,b))
