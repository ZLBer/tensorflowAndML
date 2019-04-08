import re


def camelMatch(self, queries, pattern):
    """
    :type queries: List[str]
    :type pattern: str
    :rtype: List[bool]
    """
    def isMatch(query,pattern):
        index=0
        for i in query:
            if index<len(pattern) and pattern[index]==i:
                index+=1
            elif i==i.lower():
                pass
            else:
                return False
        if index==len(pattern):
            return True
        else:
            return False
    res=[]
    for query in queries:
        res.append(isMatch(query,pattern))
    return res


#一行代码
def camelMatch(self, qs, p):
    return [re.match("^[a-z]*" + "[a-z]*".join(p) + "[a-z]*$", q) != None for q in qs]