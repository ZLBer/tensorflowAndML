def dfs(self, A, i, j):
    if i >= 0 and j >= 0 and i < len(A) and j < len(A[0]) and A[i][j]==1:
        A[i][j] = 0
        self.dfs(A, i + 1, j)
        self.dfs(A, i - 1, j)
        self.dfs(A, i, j + 1)
        self.dfs(A, i, j - 1)


def numEnclaves(self, A):
    """
    :type A: List[List[int]]
    :rtype: int
    """

    for i in range(len(A)):
        for j in range(len(A[0])):
            if i == 0 or i == len(A) - 1 or j == 0 or j == len(A[0]) - 1:
                self.dfs(A, i, j)
    res = 0
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j] == 1:
                res += 1
    return res


