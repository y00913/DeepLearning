def solution(n, lost, reserve):
    for i in range(1,n+1):
        if i in lost and i in reserve:
            lost.remove(i)
            reserve.remove(i)
    
    for i in range(1,n+1):
        if i in lost:
            if i-1 in reserve:
                lost.remove(i)
                reserve.remove(i-1)
            elif i+1 in reserve:
                lost.remove(i)
                reserve.remove(i+1)
    
    return n - len(lost)
    
    
n=7
lost=[1,2,3,4,5,6,7]
reserve=[1,2,3]
print(solution(n,lost,reserve))
