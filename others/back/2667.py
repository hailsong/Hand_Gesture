'''
7
0110100
0110101
1110101
0000111
0100000
0111110
0111000
'''

import sys

dx=[-1,0,1,0]
dy=[0,1,0,-1]
n=int(sys.stdin.readline())
a=[list(sys.stdin.readline()) for _ in range(n)]
cnt=0
apt=[]

def