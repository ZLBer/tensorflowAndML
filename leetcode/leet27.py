class Solution:
    def removeElement(self, nums, val):
        if not nums:
            return 0
        index=0
        for i in range(len(nums)):
            if nums[i]!= val:
                nums[index],nums[i]=nums[i],nums[index]
                index+=1
        return index

    def removeElement(self, nums, val):
        start, end = 0, len(nums) - 1
        while start <= end:
            if nums[start] == val:
                nums[start], nums[end], end = nums[end], nums[start], end - 1
            else:
                start += 1
        return start