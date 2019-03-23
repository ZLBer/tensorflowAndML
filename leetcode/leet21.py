from leetcode.ListNode import ListNode


class leet21(object):
    def mergeTwoLists(self, l1, l2):
        begin=cur=ListNode(0)
        while l1 and l2:
            if l1.val<l2.val:
                cur.next=l1
                l1=l1.next
            else:
                cur.next=l2
                l2=l2.next
            cur = cur.next
        if l1:
            cur.next=l1
        if l2:
            cur.next=l2
        return  begin.next

    def mergeTwoLists1(self, l1, l2):
        dummy = cur = ListNode(0)
        while l1 and l2:
            if l1.val < l2.val:
                cur.next = l1
                l1 = l1.next
            else:
                cur.next = l2
                l2 = l2.next
            cur = cur.next
        cur.next = l1 or l2  # and or 均有返回值
        return dummy.next


  # 递归
    def mergeTwoLists(self, l1, l2):
        if not l1 or not l2:
            return l1 or l2
        if l1.val<l2.val:
            l1.next=self.mergeTwoLists(l1.next,l2)
            return l1
        else:
            l2.next= self.mergeTwoLists(l1,l2.next)
            return l2

