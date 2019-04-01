def nextLargerNodes(self, head):
    """
    :type head: ListNode
    :rtype: List[int]
    """
    list = []
    while(head!=None):
        list.append(head.val)
        head=head.next

    stack = []

    res = []
    for i,val in enumerate(list):

        while stack and val >list[stack[-1]]:
            res[stack.pop()] = val

        stack.append(i)
        res.append(0)
    return  res
