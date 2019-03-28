



print(bin(1))



# The solution can be improve a half by checking from N to N/2.
# The reason is simply for every i < N/2, the binary string of 2*i will contain binary string of i. Thus we don't need to check for i < N/2
 # python 大法好
def queryString(self, S, N):
    return all(bin(i)[2:] in S for i in range(N, N / 2, -1))
