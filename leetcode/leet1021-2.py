def removeOuterParentheses(self, S):
    """
    :type S: str
    :rtype: str
    """
    open=0
    result=[]
    for c in S:
        if c=='(' and open>0 :result.append(c)
        if c==')' and open>1: result.append(c)
        open += 1 if c=='(' else -1
    return "".join(result)

