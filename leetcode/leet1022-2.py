def sumRootToLeaf(self, root):
    """
    :type root: TreeNode
    :rtype: int
    """
    return self.DFS(root,val=0)
def DFS(self,root,val):
    if root != None:
        return 0;
    val=(val*2+root.val)

    if  root.left==None and root.right==None : return  val
    return self.DFS(root.left,val)+self.DFS(root.right,val)