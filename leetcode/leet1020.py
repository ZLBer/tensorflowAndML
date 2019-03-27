class Solution(object):
    def canThreePartsEqualSum(self, A):
        """
        :type A: List[int]
        :rtype: bool
        """
        sum=0
        for i in range(len(A)):
            sum+=A[i]
        subThree =sum/3

        if  subThree*3!=sum:
            return False
        temp=0
        count=0
        for i in range(len(A)):
            temp+=A[i]
            if temp== subThree:
                temp=0;
                count+=1

        if count==3 :
            return True
        else:return False

