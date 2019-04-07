def baseNeg2(self, x):
    res = []
    while x:
        res.append(x & 1)
        x = -(x >> 1)
    return "".join(map(str, res[::-1] or [0]))   #map()函数将一个全部为int的列表，转化为全部为str的列表

# 所以res[::-1]相当于 res[-1:-len(res)-1:-1]，也就是从最后一个元素到第一个元素复制一遍。所以你看到一个倒序的。

def baseNeg2(self, N):
    if N == 0 or N == 1: return str(N)
    return self.baseNeg2(-(N >> 1)) + str(N & 1)