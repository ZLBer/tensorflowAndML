import itertools


def clumsy(self, N):
    op = itertools.cycle("*/+-")
    return eval("".join(str(x) + next(op) for x in range(N, 0, -1)))


#太牛逼了
def clumsy(self, N):
    return [0, 1, 2, 6, 7][N] if N < 5 else N + [1, 2, 2, - 1][N % 4]
#  [0, 1, 2, 6, 7][N] if N < 5
#   N=0 return 0