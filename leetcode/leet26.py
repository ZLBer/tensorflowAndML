class leet26(object):
    def removeDuplicates(self, nums):
        back=0
        for i in range(1,len(nums)):
            if(nums[i]!=nums[back]):
                back+=1
                nums[back]=nums[i]

        return back+1
