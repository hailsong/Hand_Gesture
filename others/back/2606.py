'''
7
6
1 2
2 3
1 5
5 2
5 6
4 7
'''

N = int(input())
M = int(input())
V = 1
matrix=[[0]*(N+1) for i in range(N+1)]
#print(matrix)

for i in range(M):
    a, b = map(int, input().split(' '))
    matrix[a][b] = 1
    matrix[b][a] = 1

from collections import deque

def DFS(m, start):
    visited = []
    stack = deque([start])

    while stack:
        #print(stack)
        now = stack.pop()
        if not now in visited:
            visited.append(now)
            for i in range(N+1)[::-1]:
                if m[now][i] == 1:
                    stack.append(i)
    return visited

def BFS(m, start):
    visited = []
    queue = deque([start])

    while queue:
        #print(stack)
        now = queue.popleft()
        if not now in visited:
            visited.append(now)
            for i in range(N+1):
                if m[now][i] == 1:
                    queue.append(i)
    return visited

print(len(DFS(matrix, V)) - 1)
