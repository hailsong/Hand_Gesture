graph_list = {1: set([3, 4]),
              2: set([3, 4, 5]),
              3: set([1, 5]),
              4: set([1]),
              5: set([2, 6]),
              6: set([3, 5])}
root_node = 1

from collections import deque

def BFS(graph, root):
    visited = []
    queue = deque([root])

    while queue:
        now = queue.popleft()
        if now in visited:
            pass
        else:
            visited.append(now)
            queue += graph[now]
    return visited

print(BFS(graph_list, root_node))




# from collections import deque
#
#
# def BFS_with_adj_list(graph, root):
#     visited = []
#     queue = deque([root])
#
#     while queue:
#         m = queue.popleft()
#         if not m in visited:
#             visited.append(m)
#             queue += graph[m]
#     return visited
#
#
# print(BFS_with_adj_list(graph_list, root_node))
#
