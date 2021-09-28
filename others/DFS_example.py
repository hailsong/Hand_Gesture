graph_list = {1: set([3, 4]),
              2: set([3, 4, 5]),
              3: set([1, 5]),
              4: set([1]),
              5: set([2, 6]),
              6: set([3, 5])}
root_node = 1

from collections import deque

def DFS(graph, root):
    visited = []
    stack = deque([root])

    while stack:
        now = stack.pop()
        if now in visited:
            pass
        else:
            visited.append(now)
            stack += graph[now]
    return visited

print(DFS(graph_list, root_node))