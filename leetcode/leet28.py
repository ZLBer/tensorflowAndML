class Solution(object):
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        for i in range(len(haystack) - len(needle)+1):
            if haystack[i:i+len(needle)] == needle:
                return i
        return -1

    def strStr(self, haystack, needle):
        if (needle not in haystack):
            return (-1)
        return (haystack.index(needle))