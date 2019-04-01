def prefixesDivBy5(self, A):
    """
    :type A: List[int]
    :rtype: List[bool]
    """
    k=0;
    result=[]
    for a in A:
        k=(k<<1|a)%5
        result.append(k==0)
    return result

def prefixesDivBy5(self, A):
        for i in range(1, len(A)):
            A[i] += A[i - 1] * 2 % 5
        return [a % 5 == 0 for a in A]