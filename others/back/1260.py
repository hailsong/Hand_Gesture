'''
4 5 1
1 2
1 3
1 4
2 4
3 4
'''

N,M,V=map(int,input().split()) # 노
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

print(*DFS(matrix, V))
print(*BFS(matrix, V))
