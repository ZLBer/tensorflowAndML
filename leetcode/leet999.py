def numRookCaptures(self, board):
    for i in range(8):
        for j in range(8):
            if board[i][j]=='R':
                x0,y0=i,j


    res=0

    for i,j in [[1,0],[0,1],[-1,0],[0,-1]]:
        x,y=x0+i,y0+y
        while 0<=x<8 and 0<=y<8:
            if board[x][y]=='p':
                res+=1;
                break;
            if board[x][y]!='.':
                break;
            x,y=x+1,y+j
    return res