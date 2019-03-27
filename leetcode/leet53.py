import sys


class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        sum=0
        maxSum=nums[0]
        for i in range(len(nums)):
            sum+=nums[i]
            maxSum=max(maxSum,sum)
            if sum<0:
                sum=0


        return maxSum


