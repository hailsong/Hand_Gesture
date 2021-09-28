N = int(input())
li = []
for i in range(N):
    a, b = map(str, input().split(' '))
    li.append([int(a), i, b])

li.sort(key = lambda x:(x[0], x[1]))
for l in li:
    print(l[0], l[2])