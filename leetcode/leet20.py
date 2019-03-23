class leet20(object):
    def isValid(self, s):
        stack = []
        for i in s:
            if i == '(':
                stack.append(i)
            elif i == '[':
                stack.append(i)
            elif i == '{':
                stack.append(i)
            elif i == ')':
                if len(stack) == 0 or stack.pop() != '(':
                    return False
            elif i == ']':
                if len(stack) == 0 or stack.pop() != '[':
                    return False
            else:
                if len(stack) == 0 or stack.pop() != '{':
                    return False
        if len(stack) == 0:
            return True
        else:
            return False
            # @return a boolean

  # 优雅的做法
    def isValid(self, s):
        stack = []
        dict = {"]": "[", "}": "{", ")": "("}
        for char in s:
            if char in dict.values():
                stack.append(char)
            elif char in dict.keys():
                if stack == [] or dict[char] != stack.pop():
                    return False
            else:
                return False
        return stack == []
